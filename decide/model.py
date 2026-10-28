"""
Decide & DecideX Neural Architecture + Laya 421M Reference Baseline.
Implements Shared-State KV Encoding, Parallel Isolated Option Encoding,
Gated Cross-Attention, ColBERT MaxSim Late-Interaction, CORAL Ordinal Score Head,
Contrastive Polarity Noul Verifier, and Monotonic Cardinality Calibrator.
"""
import math
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DecideConfig


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, res_scale: float = 1.0):
        super().__init__()
        self.w_gate_up = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w_gate_up(x).chunk(2, dim=-1)
        return self.w_down(F.silu(gate) * up) * self.res_scale


class FastSelfAttentionBlock(nn.Module):
    """Fused C++ LayerNorm + Flash-SDPA + SwiGLU block with DeepNorm residual scaling."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0, n_layers: int = 2):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.res_scale = 1.0 / math.sqrt(max(1, 2 * n_layers))
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, d_ff, dropout, res_scale=self.res_scale)

    def forward(self, x: torch.Tensor, attn_mask_4d: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, D = x.shape
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask_4d)
        x = x + self.proj(out.transpose(1, 2).reshape(B, L, D)) * self.res_scale
        x = x + self.ffn(self.norm2(x))
        return x


class GatedCrossAttentionBlock(nn.Module):
    """Cross-attends Option/Query embeddings against cached Shared State KV Bank in O(K * L_state)."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wkv = nn.Linear(d_model, 2 * d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Parameter(torch.tensor(0.5))

    def encode_kv(self, state_h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, Ls, _ = state_h.shape
        k, v = self.wkv(self.norm_kv(state_h)).chunk(2, dim=-1)
        k = k.view(B, Ls, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, Ls, self.n_heads, self.head_dim).transpose(1, 2)
        return k, v

    def forward(
        self,
        opt_h: torch.Tensor,
        state_kv: Tuple[torch.Tensor, torch.Tensor],
        s_mask_4d: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        Q, K, D = opt_h.shape
        q = self.wq(self.norm_q(opt_h)).view(Q, K, self.n_heads, self.head_dim).transpose(1, 2)
        k, v = state_kv
        cross_out = F.scaled_dot_product_attention(q, k, v, attn_mask=s_mask_4d)
        aligned_state = self.wo(cross_out.transpose(1, 2).reshape(Q, K, D))
        opt_h = opt_h + torch.tanh(self.gate) * aligned_state
        return opt_h, aligned_state


class MonotonicCardinalityCalibrator(nn.Module):
    """Learnable per-primitive & option-count aware temperature scaling bounded in [0.5, 4.0]."""
    def __init__(self):
        super().__init__()
        self.base_log_temp = nn.Parameter(torch.zeros(3))
        self.card_slope = nn.Parameter(torch.tensor([0.05, 0.03, 0.0]))

    def get_temperature(self, qtype: torch.Tensor, num_options: torch.Tensor) -> torch.Tensor:
        base = self.base_log_temp[qtype]
        slope = self.card_slope[qtype]
        log_k = torch.log2(num_options.float().clamp(min=2.0)) - 1.0
        raw_temp = F.softplus(base + slope * log_k) + 0.5
        return raw_temp.clamp(0.5, 4.0)


class DecideModel(nn.Module):
    """
    Decide & DecideX Neural Architecture:
    1. Encodes State ONCE into a Shared State KV Bank + Late-Interaction Token Bank.
    2. Encodes Questions + Options in parallel with Fused Gated Set-Pooling (zero sequential truncation).
    3. Fuses via Gated Cross-Attention + Hadamard/Bilinear Alignment + ColBERT MaxSim + Tri-Primitive Heads.
    4. Outputs calibrated probabilities + Epistemic Act/Escalate probabilities.
    """
    def __init__(self, cfg: DecideConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.tok_emb = nn.Embedding(cfg.vocab_size, d, padding_idx=0)
        self.pos_emb = nn.Embedding(cfg.max_state_len, d)
        self.qtype_emb = nn.Embedding(3, d)
        self.register_buffer("pos_ids", torch.arange(cfg.max_state_len, dtype=torch.long).unsqueeze(0), persistent=False)
        self.register_buffer("opt_levels", torch.arange(cfg.max_options, dtype=torch.float32).unsqueeze(0), persistent=False)

        # 1. Shared State Backbone (Runs ONCE per state!)
        self.state_blocks = nn.ModuleList([
            FastSelfAttentionBlock(d, cfg.n_heads, cfg.d_ff, cfg.dropout, n_layers=cfg.state_layers)
            for _ in range(cfg.state_layers)
        ])
        self.state_norm = nn.LayerNorm(d)

        # 2. Fast Fused Question & Option Projection
        self.opt_proj = nn.Linear(d, d, bias=False)
        self.opt_norm = nn.LayerNorm(d)

        # 3. Cross-Attention Layers over Shared State KV Bank
        self.cross_blocks = nn.ModuleList([
            GatedCrossAttentionBlock(d, cfg.n_heads, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.cross_layers)
        ])

        # 4. Late-Interaction Multi-Vector Projection (ColBERT MaxSim + Bilinear Alignment)
        self.state_li_proj = nn.Linear(d, cfg.late_interaction_dim, bias=False)
        self.opt_li_proj = nn.Linear(d, cfg.late_interaction_dim, bias=False)
        self.li_scale = nn.Parameter(torch.tensor(4.0))

        # 5. Tri-Primitive Specialized Scorers with Multiplicative State-Option Interaction
        self.choice_scorer = nn.Sequential(
            nn.LayerNorm(2 * d),
            nn.Linear(2 * d, d // 2),
            nn.GELU(),
            nn.Linear(d // 2, 1),
        )
        self.score_rank_proj = nn.Linear(d, 1)
        self.ordinal_sharpness = nn.Parameter(torch.tensor(2.5))
        self.noul_verifier = nn.Linear(2 * d, 2)

        # 6. Calibration & Epistemic Act/Escalate Head
        self.calibrator = MonotonicCardinalityCalibrator()
        self.act_head = nn.Linear(d + 2, 2)

    def encode_state(
        self, state_ids: torch.Tensor, state_mask: torch.Tensor
    ) -> Dict[str, Any]:
        """Encode state ONCE and precompute KV banks for all cross-attention layers."""
        B, Ls = state_ids.shape
        h = self.tok_emb(state_ids) + self.pos_emb(self.pos_ids[:, :Ls])
        attn_mask_4d = state_mask[:, None, None, :].bool()
        for blk in self.state_blocks:
            h = blk(h, attn_mask_4d)
        h = self.state_norm(h)
        mask_f = state_mask.unsqueeze(-1).type_as(h)
        h = h * mask_f
        kv_bank = [cblk.encode_kv(h) for cblk in self.cross_blocks]
        li_toks = F.normalize(self.state_li_proj(h), p=2, dim=-1) * mask_f
        pooled = h.sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        return {
            "h": h,
            "pooled": pooled,
            "mask": state_mask,
            "mask_4d": attn_mask_4d,
            "kv_bank": kv_bank,
            "li_toks": li_toks,
        }

    def forward_questions(
        self,
        state_cache: Dict[str, Any],
        q_ids: torch.Tensor,
        q_mask: torch.Tensor,
        opt_ids: torch.Tensor,
        opt_tok_mask: torch.Tensor,
        opt_valid_mask: torch.Tensor,
        qtype: torch.Tensor,
        state_idx: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        Q, K, Lo = opt_ids.shape

        # Encode Question instructions [Q, Lq, D] -> [Q, D]
        qh = self.tok_emb(q_ids)
        qm_f = q_mask.unsqueeze(-1).type_as(qh)
        q_pooled = self.opt_norm((qh * qm_f).sum(dim=1) / qm_f.sum(dim=1).clamp(min=1.0) + self.qtype_emb(qtype))
        q_pooled = q_pooled + 0.5 * F.gelu(self.opt_proj(q_pooled))

        # Encode Options in parallel [Q, K, Lo, D] -> [Q, K, D]
        oh = self.tok_emb(opt_ids)
        om_f = opt_tok_mask.unsqueeze(-1).type_as(oh)
        opt_pooled = self.opt_norm((oh * om_f).sum(dim=2) / om_f.sum(dim=2).clamp(min=1.0))
        opt_pooled = opt_pooled + 0.5 * F.gelu(self.opt_proj(opt_pooled))

        # Gather or broadcast state KV bank to match Q questions
        B_state = state_cache["h"].shape[0]
        if B_state == 1 and Q > 1:
            s_mask_4d = state_cache["mask_4d"].expand(Q, -1, -1, -1)
            s_pooled = state_cache["pooled"].expand(Q, -1)
            s_li = state_cache["li_toks"].expand(Q, -1, -1)
            kv_list = [(k.expand(Q, -1, -1, -1), v.expand(Q, -1, -1, -1)) for k, v in state_cache["kv_bank"]]
        elif state_idx is not None:
            s_mask_4d = state_cache["mask_4d"][state_idx]
            s_pooled = state_cache["pooled"][state_idx]
            s_li = state_cache["li_toks"][state_idx]
            kv_list = [(k[state_idx], v[state_idx]) for k, v in state_cache["kv_bank"]]
        else:
            s_mask_4d = state_cache["mask_4d"]
            s_pooled = state_cache["pooled"]
            s_li = state_cache["li_toks"]
            kv_list = state_cache["kv_bank"]

        # Combine Question context into each Option representation
        opt_raw = opt_pooled + 0.5 * q_pooled.unsqueeze(1)
        opt_h = opt_raw
        aligned_state = s_pooled.unsqueeze(1)

        # Cross-Attention over Shared State KV Bank
        for cblk, kv in zip(self.cross_blocks, kv_list):
            opt_h, aligned_state = cblk(opt_h, kv, s_mask_4d)

        # Late-Interaction ColBERT MaxSim between Option embeddings and State token bank
        opt_li = F.normalize(self.opt_li_proj(opt_raw), p=2, dim=-1)  # [Q, K, D_li]
        sim_matrix = torch.bmm(opt_li, s_li.transpose(1, 2))  # [Q, K, Ls]
        max_sim_score = sim_matrix.max(dim=-1).values * self.li_scale  # [Q, K]

        # 1. Choice Logits (Multiplicative Hadamard + Cross-Attended Option + MaxSim)
        s_ctx = 0.5 * (s_pooled.unsqueeze(1) + aligned_state)
        inter_feats = torch.cat([opt_h, opt_raw * s_ctx], dim=-1)
        choice_logits = self.choice_scorer(inter_feats).squeeze(-1) + max_sim_score  # [Q, K]

        # 2. Score & 3. Noul specialized adjustments
        qs_prod = q_pooled * s_pooled
        scalar_rank = torch.sigmoid(self.score_rank_proj(qs_prod))
        num_opts = opt_valid_mask.sum(dim=-1).clamp(min=2)
        k_f = num_opts.float()
        norm_levels = self.opt_levels[:, :K] / (k_f.unsqueeze(-1) - 1.0)
        ordinal_prior = -self.ordinal_sharpness.abs() * (norm_levels - scalar_rank).pow(2)
        score_logits = choice_logits + ordinal_prior

        opt_diff = opt_raw[:, 1, :] - opt_raw[:, 0, :] if K >= 2 else opt_raw[:, 0, :]
        noul_pair = self.noul_verifier(torch.cat([qs_prod, s_pooled * opt_diff], dim=-1))
        noul_logits = choice_logits.clone()
        if K >= 2:
            noul_logits[:, :2] = choice_logits[:, :2] + noul_pair

        is_score = (qtype == 1).unsqueeze(-1)
        is_noul = (qtype == 2).unsqueeze(-1)
        raw_logits = torch.where(is_score, score_logits, torch.where(is_noul, noul_logits, choice_logits)).float()
        raw_logits = raw_logits.masked_fill(~opt_valid_mask, -1e4)

        # Monotonic Cardinality-Aware Calibration
        temps = self.calibrator.get_temperature(qtype, num_opts).unsqueeze(-1)
        cal_logits = (raw_logits / temps).masked_fill(~opt_valid_mask, -1e4)
        probs = F.softmax(cal_logits, dim=-1)

        # Streamlined Epistemic Act / Escalate Head
        p_det = probs.detach()
        ent = -(p_det * torch.log(p_det.clamp_min(1e-9))).sum(-1) / torch.log(k_f)
        max_p = p_det.max(dim=-1).values
        act_feats = torch.stack([max_p, 1.0 - ent], dim=-1)
        act_logits = self.act_head(torch.cat([s_pooled.float(), act_feats], dim=-1))

        return {
            "raw_logits": raw_logits,
            "cal_logits": cal_logits,
            "probs": probs,
            "act_logits": act_logits,
            "temperatures": temps.squeeze(-1),
            "confidence": 1.0 - ent,
            "answer_confidence": max_p,
        }

