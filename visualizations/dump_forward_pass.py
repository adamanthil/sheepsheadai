"""Run a forward pass through the trained PPO network on a synthetic
pick-decision state and dump all intermediate activations to JSON for
the 3D architecture visualization."""

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from ppo import PPOAgent  # noqa: E402
from sheepshead import (  # noqa: E402
    ACTION_IDS,
    ACTION_LOOKUP,
    ACTIONS,
    DECK_IDS,
    TRUMP,
)

CHECKPOINT = ROOT / "pfsp_checkpoints_swish" / "pfsp_swish_checkpoint_30000000.pt"
OUT_JSON = HERE / "ppo_forward_pass.json"


def build_sample_state():
    """Pick-decision moment: not picker, hand has QC + 2 other trump."""
    hand_cards = ["QC", "JC", "AD", "AS", "KH", "7C"]  # 6 cards, 3 trump
    hand_ids = [DECK_IDS[c] for c in hand_cards] + [0, 0]  # pad to 8

    state = {
        "partner_mode": np.uint8(0),  # JD partner
        "is_leaster": np.uint8(0),
        "play_started": np.uint8(0),
        "current_trick": np.uint8(0),
        "alone_called": np.uint8(0),
        "called_card_id": np.uint8(0),
        "called_under": np.uint8(0),
        "picker_rel": np.uint8(0),
        "partner_rel": np.uint8(0),
        "leader_rel": np.uint8(0),
        "picker_position": np.uint8(0),
        "hand_ids": np.array(hand_ids, dtype=np.uint8),
        "blind_ids": np.array([0, 0], dtype=np.uint8),
        "bury_ids": np.array([0, 0], dtype=np.uint8),
        "trick_card_ids": np.array([0, 0, 0, 0, 0], dtype=np.uint8),
        "trick_is_picker": np.array([0, 0, 0, 0, 0], dtype=np.uint8),
        "trick_is_partner_known": np.array([0, 0, 0, 0, 0], dtype=np.uint8),
    }
    valid_actions = {ACTION_IDS["PICK"], ACTION_IDS["PASS"]}
    return state, valid_actions, hand_cards


def tensor_to_py(t: torch.Tensor):
    if t is None:
        return None
    return t.detach().cpu().tolist()


def main():
    print(f"Loading checkpoint: {CHECKPOINT}")
    agent = PPOAgent(action_size=len(ACTIONS))
    agent.load(str(CHECKPOINT), load_optimizers=False)
    agent.encoder.eval()
    agent.actor.eval()
    agent.critic.eval()

    state, valid_actions, hand_cards = build_sample_state()
    device = next(agent.encoder.parameters()).device

    # ---- Manual forward through encoder so we can capture intermediates ----
    enc = agent.encoder

    with torch.no_grad():
        # zero memory
        memory_in = torch.zeros((1, 256), device=device)

        # Replicate encoder.encode_batch with intermediate captures
        batch = [state]

        header_fields = [
            "partner_mode",
            "is_leaster",
            "play_started",
            "current_trick",
            "alone_called",
            "called_under",
            "picker_rel",
            "partner_rel",
            "leader_rel",
            "picker_position",
        ]
        header_cols = [enc._stack_scalar(batch, k) for k in header_fields]
        header_scalar = torch.cat(header_cols, dim=1).to(device)
        norm = torch.tensor(
            [1.0, 1.0, 1.0, 6.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0],
            dtype=header_scalar.dtype,
            device=device,
        )
        header_scalar_n = header_scalar / norm

        called_ids = torch.tensor(
            [int(state["called_card_id"])], device=device, dtype=torch.long
        )
        called_emb = enc.card(called_ids)
        context_tok = enc.context_mlp(torch.cat([header_scalar_n, called_emb], dim=1))
        memory_tok = enc.memory_in_proj(memory_in)

        picker_rel_raw = torch.tensor(
            [int(state["picker_rel"])], device=device, dtype=torch.long
        )
        partner_rel_raw = torch.tensor(
            [int(state["partner_rel"])], device=device, dtype=torch.long
        )
        actor_role_id = (
            picker_rel_raw.eq(1).long() * 1 + partner_rel_raw.eq(1).long() * 2
        )

        hand_ids = torch.as_tensor(
            state["hand_ids"], dtype=torch.long, device=device
        ).view(1, 8)
        blind_ids = torch.as_tensor(
            state["blind_ids"], dtype=torch.long, device=device
        ).view(1, 2)
        bury_ids = torch.as_tensor(
            state["bury_ids"], dtype=torch.long, device=device
        ).view(1, 2)
        trick_card_ids = torch.as_tensor(
            state["trick_card_ids"], dtype=torch.long, device=device
        ).view(1, 5)
        trick_is_picker = (
            torch.as_tensor(state["trick_is_picker"], dtype=torch.long, device=device)
            .view(1, 5)
            .bool()
        )
        trick_is_partner_known = (
            torch.as_tensor(
                state["trick_is_partner_known"], dtype=torch.long, device=device
            )
            .view(1, 5)
            .bool()
        )

        hand_tok, hand_mask = enc._embed_hand(hand_ids, actor_role_id)
        blind_tok, blind_mask = enc._embed_simple_bag(blind_ids)
        bury_tok, bury_mask = enc._embed_simple_bag(bury_ids)
        trick_tok, trick_mask = enc._embed_trick(
            trick_card_ids, trick_is_picker, trick_is_partner_known
        )

        all_tokens_pre = torch.cat(
            [
                context_tok.unsqueeze(1),
                memory_tok.unsqueeze(1),
                hand_tok,
                trick_tok,
                blind_tok,
                bury_tok,
            ],
            dim=1,
        )  # (1, 19, 64)
        all_mask = torch.cat(
            [
                torch.ones((1, 1), dtype=torch.bool, device=device),
                torch.ones((1, 1), dtype=torch.bool, device=device),
                hand_mask,
                trick_mask,
                blind_mask,
                bury_mask,
            ],
            dim=1,
        )

        type_ids = torch.cat(
            [
                torch.zeros((1, 1), dtype=torch.long, device=device),
                torch.ones((1, 1), dtype=torch.long, device=device),
                torch.full((1, 8), 2, dtype=torch.long, device=device),
                torch.full((1, 5), 3, dtype=torch.long, device=device),
                torch.full((1, 2), 4, dtype=torch.long, device=device),
                torch.full((1, 2), 5, dtype=torch.long, device=device),
            ],
            dim=1,
        )
        all_tokens_in = all_tokens_pre + enc.card_type(type_ids)

        # Run transformer manually, capturing attention weights from last layer
        tokens = all_tokens_in
        attn_mask = ~all_mask
        attn_weights_last = None
        for li, (attn, ffn, ln1, ln2) in enumerate(
            zip(
                enc.card_reasoner.attn_layers,
                enc.card_reasoner.ffn_layers,
                enc.card_reasoner.ln_attn,
                enc.card_reasoner.ln_ffn,
            )
        ):
            attn_out, attn_w = attn(
                tokens,
                tokens,
                tokens,
                key_padding_mask=attn_mask,
                need_weights=True,
                average_attn_weights=True,
            )
            tokens = ln1(tokens + attn_out)
            ffn_out = ffn(tokens)
            tokens = ln2(tokens + ffn_out)
            if li == len(enc.card_reasoner.attn_layers) - 1:
                attn_weights_last = attn_w  # (1, 19, 19)

        all_tokens_post = tokens
        context_out = all_tokens_post[:, 0, :]
        hand_tok_out = all_tokens_post[:, 2:10, :]
        trick_tok_out = all_tokens_post[:, 10:15, :]
        blind_tok_out = all_tokens_post[:, 15:17, :]
        bury_tok_out = all_tokens_post[:, 17:19, :]

        hand_vec = enc.pool_hand(hand_tok_out, hand_mask)
        trick_vec = enc.pool_trick(trick_tok_out, trick_mask)
        blind_vec = enc.pool_blind(blind_tok_out, blind_mask)
        bury_vec = enc.pool_bury(bury_tok_out, bury_mask)

        memory_out = enc.memory_gru(context_out, memory_in)
        features = enc.feature_proj(
            torch.cat([hand_vec, trick_vec, blind_vec, bury_vec, context_out], dim=1)
        )

        # ---- Actor ----
        actor = agent.actor
        feat = actor.actor_adapter(features)

        # pick / pass
        pick_logits = actor.pick_head(feat)  # (1, 2)
        partner_basic = actor.partner_basic_head(feat)  # (1, 2)
        play_under_logit = actor.play_under_head(feat).squeeze(-1)  # (1,)

        # two-tower
        q_tw = actor.tw_Wg(feat)  # (1, 64)
        K_tw = actor.tw_We(enc.card.weight)  # (34, 64)
        card_scores = torch.matmul(q_tw, K_tw.t())  # (1, 34)

        # pointer
        slot_scores = actor._score_hand_pointer(feat, hand_tok_out)  # (1, 8)

        # full action logits via the model
        action_mask_t = (
            agent.get_action_mask(valid_actions, agent.action_size)
            .unsqueeze(0)
            .to(device)
        )
        probs, full_logits = actor.forward_with_logits(
            {"features": features, "hand_tokens": hand_tok_out},
            action_mask_t,
            hand_ids,
            enc.card,
        )

        # ---- Critic ----
        critic = agent.critic
        crit_feat = critic.critic_adapter(features)
        value = critic.value_head(crit_feat).squeeze(-1)
        win_logit = critic.win_head(crit_feat).squeeze(-1)
        return_pred = critic.return_head(crit_feat).squeeze(-1)
        secret_logit = critic.secret_partner_head(crit_feat).squeeze(-1)
        points_pred = critic.points_head(crit_feat)
        seen_trump_mask_logits = critic.seen_trump_mask_logits(crit_feat, enc.card)
        unseen_higher_logits = critic.unseen_trump_higher_than_hand_logits(crit_feat)

    # ---------- Extract & format ----------
    # Card embedding rows used
    used_card_ids = sorted(set([cid for cid in hand_ids[0].tolist() if cid != 0]))
    trump_card_ids = [DECK_IDS[c] for c in TRUMP]

    out = {
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "sample": {
            "hand_cards": hand_cards,
            "hand_ids": hand_ids[0].tolist(),
            "valid_actions": [ACTION_LOOKUP[a] for a in sorted(valid_actions)],
            "phase": "pick-decision (no one has picked yet)",
        },
        "card_embedding": {
            "shape": list(enc.card.weight.shape),
            "rows_used": {
                str(cid): enc.card.weight[cid].detach().cpu().tolist()
                for cid in used_card_ids
            },
            # Also expose a couple of stats so JS can normalize displays
            "min": float(enc.card.weight.min().item()),
            "max": float(enc.card.weight.max().item()),
        },
        "tokens_pre_attn": {
            "context": tensor_to_py(context_tok[0]),
            "memory": tensor_to_py(memory_tok[0]),
            "hand": tensor_to_py(hand_tok[0]),  # (8, 64)
            "trick": tensor_to_py(trick_tok[0]),  # (5, 64)
            "blind": tensor_to_py(blind_tok[0]),  # (2, 64)
            "bury": tensor_to_py(bury_tok[0]),  # (2, 64)
            "mask": all_mask[0].tolist(),
        },
        "tokens_post_attn": {
            "context": tensor_to_py(context_out[0]),
            "hand": tensor_to_py(hand_tok_out[0]),
            "trick": tensor_to_py(trick_tok_out[0]),
            "blind": tensor_to_py(blind_tok_out[0]),
            "bury": tensor_to_py(bury_tok_out[0]),
        },
        "attention_weights_last_layer": tensor_to_py(attn_weights_last[0]),  # (19, 19)
        "pools": {
            "hand_vec": tensor_to_py(hand_vec[0]),
            "trick_vec": tensor_to_py(trick_vec[0]),
            "blind_vec": tensor_to_py(blind_vec[0]),
            "bury_vec": tensor_to_py(bury_vec[0]),
        },
        "memory": {
            "memory_in": memory_in[0].tolist(),
            "memory_out": memory_out[0].tolist(),
        },
        "features": tensor_to_py(features[0]),
        "actor": {
            "adapter_out": tensor_to_py(feat[0]),
            "pick_logits": tensor_to_py(pick_logits[0]),  # [PICK, PASS]
            "partner_basic_logits": tensor_to_py(
                partner_basic[0]
            ),  # [ALONE, JD PARTNER]
            "play_under_logit": float(play_under_logit.item()),
            "two_tower_q": tensor_to_py(q_tw[0]),
            "card_scores_all": tensor_to_py(card_scores[0]),  # (34,)
            "slot_scores": tensor_to_py(slot_scores[0]),  # (8,)
            "full_probs": tensor_to_py(probs[0]),
            "full_logits": tensor_to_py(full_logits[0]),
            "valid_action_indices": (action_mask_t[0].nonzero().squeeze(-1).tolist()),
        },
        "critic": {
            "value": float(value.item()),
            "win_prob": float(torch.sigmoid(win_logit).item()),
            "return_pred": float(return_pred.item()),
            "secret_prob": float(torch.sigmoid(secret_logit).item()),
            "points_pred": tensor_to_py(points_pred[0]),
            "seen_trump_mask_logits": tensor_to_py(seen_trump_mask_logits[0]),
            "seen_trump_mask_probs": tensor_to_py(
                torch.sigmoid(seen_trump_mask_logits[0])
            ),
            "unseen_higher_logit": float(unseen_higher_logits.item()),
            "unseen_higher_prob": float(torch.sigmoid(unseen_higher_logits).item()),
        },
        "constants": {
            "TRUMP": TRUMP,
            "trump_card_ids": trump_card_ids,
            "DECK_IDS": DECK_IDS,
            "ACTIONS": ACTIONS,
            "PICK_index_0based": ACTION_IDS["PICK"] - 1,
            "PASS_index_0based": ACTION_IDS["PASS"] - 1,
        },
    }

    with open(OUT_JSON, "w") as f:
        json.dump(out, f)
    size_mb = OUT_JSON.stat().st_size / (1024 * 1024)
    print(f"Wrote {OUT_JSON.name} ({size_mb:.2f} MB)")

    # Quick console summary
    pick_idx = ACTION_IDS["PICK"] - 1
    pass_idx = ACTION_IDS["PASS"] - 1
    print(f"  P(PICK) = {out['actor']['full_probs'][pick_idx]:.4f}")
    print(f"  P(PASS) = {out['actor']['full_probs'][pass_idx]:.4f}")
    print(f"  V(s)    = {out['critic']['value']:.3f}")
    print(f"  win     = {out['critic']['win_prob']:.3f}")
    print(f"  return  = {out['critic']['return_pred']:.3f}")


if __name__ == "__main__":
    main()
