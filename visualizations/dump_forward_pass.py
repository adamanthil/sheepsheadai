"""Play one deterministic hand with a trained perceiver PPO agent and dump
the forward-pass activations at several decision points to JSON for the 3D
architecture visualization.

The hand is played in called-ace mode with the same agent in all five seats
(recurrent memory keyed per seat), so every captured scenario has a genuine
GRU memory state and trick context. Captured decision types:

  pick    - the eventual picker's PICK/PASS decision
  call    - the picker's partner-call decision (two-tower head live)
  bury    - the picker's first bury decision (pointer head live)
  lead    - the opening lead of trick 0
  follow  - a late-trick follow with cards on the table

A seed is scanned so the hand contains all five decision types (someone
picks, calls a partner card, and the hand plays out to completion).

Unlike the old `full`-architecture dump (last-layer head-averaged attention
only), this captures the whole transformer: per-layer, per-head attention
maps, per-layer token norms, FFN hidden-activation norms, and the actor's
and critic's readout cross-attention (learned queries over the 19
post-reasoning tokens).
"""

import argparse
import copy
import json
import sys
from datetime import date
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from architectures import PerceiverEncoder  # noqa: E402
from ppo import PerceiverActorNetwork, PerceiverCriticNetwork, load_agent  # noqa: E402
from sheepshead import (  # noqa: E402
    ACTION_IDS,
    ACTION_LOOKUP,
    ACTIONS,
    DECK_IDS,
    PARTNER_BY_CALLED_ACE,
    TRUMP,
    Game,
)

DEFAULT_CHECKPOINT = ROOT / "runs" / "ablate_perceiver_s42" / "best_perceiver.pt"
OUT_JSON = HERE / "ppo_forward_pass.json"

ID_TO_CODE = {v: k for k, v in DECK_IDS.items()}
ID_TO_CODE[0] = "PAD"
ID_TO_CODE[33] = "UNDER"

SCENARIO_ORDER = ["pick", "call", "bury", "lead", "follow"]
SCENARIO_LABELS = {
    "pick": "Pick or pass",
    "call": "Call a partner",
    "bury": "Bury two cards",
    "lead": "Lead trick 1",
    "follow": "Follow late in the hand",
}


def tensor_to_py(t: torch.Tensor):
    if t is None:
        return None
    return t.detach().cpu().tolist()


def token_norms(t: torch.Tensor):
    """Per-token L2 norms of a (1, N, d) tensor -> list of N floats."""
    return t[0].norm(dim=-1).tolist()


def classify_decision(action_names, state):
    """Map a decision point to a scenario kind (or None to skip)."""
    if "PICK" in action_names:
        return "pick"
    if any(n.startswith("CALL ") or n in ("ALONE", "JD PARTNER") for n in action_names):
        return "call"
    if any(n.startswith("BURY ") for n in action_names):
        return "bury"
    if any(n.startswith("PLAY ") for n in action_names):
        cards_on_table = sum(1 for c in state["trick_card_ids"] if int(c) != 0)
        if cards_on_table == 0 and int(state["current_trick"]) == 0:
            return "lead"
        if cards_on_table >= 2 and int(state["current_trick"]) >= 2:
            return "follow"
    return None


def play_hand(agent, seed, force_pick=False):
    """Play one full deterministic hand, snapshotting candidate decisions.

    Returns (game, snapshots) where snapshots maps kind -> list of dicts with
    the state, valid actions, pre-decision memory, and chosen action.

    With force_pick=True, the last seat to see a PICK opportunity is forced
    to PICK (mid-training checkpoints can PASS every hand into a leaster, and
    then the call/bury phases never occur). Every captured forward pass is
    still the genuine policy on a real game state; only that one action is
    overridden, and its snapshot is marked "forced".
    """
    game = Game(partner_selection_mode=PARTNER_BY_CALLED_ACE, seed=seed)
    agent.reset_recurrent_state()
    snapshots = {k: [] for k in SCENARIO_ORDER}

    while not game.is_done():
        for player in game.players:
            valid_actions = player.get_valid_action_ids()
            while valid_actions:
                state = player.get_state_dict()
                names = [ACTION_LOOKUP[a] for a in sorted(valid_actions)]
                kind = classify_decision(names, state)
                snap = None
                if kind is not None and len(valid_actions) >= 2:
                    snap = {
                        "kind": kind,
                        "seat": player.position,
                        "trick": int(game.current_trick),
                        "state": copy.deepcopy(state),
                        "valid_actions": set(valid_actions),
                        "memory_in": agent.get_recurrent_memory(
                            player.position
                        ).clone(),
                    }

                forced = (
                    force_pick
                    and "PICK" in names
                    and player.position == 5  # last seat: no earlier picker
                )
                if forced:
                    action = ACTION_IDS["PICK"]
                    if snap is not None:
                        snap["forced"] = True
                else:
                    action, _, _ = agent.act(
                        state, valid_actions, player.position, deterministic=True
                    )
                player.act(action)
                if snap is not None:
                    snap["chosen_action"] = ACTION_LOOKUP[action]
                    snapshots[kind].append(snap)
                valid_actions = player.get_valid_action_ids()

                # Propagate end-of-trick observation to every seat's memory.
                if game.was_trick_just_completed:
                    for seat in game.players:
                        agent.observe(
                            seat.get_last_trick_state_dict(),
                            player_id=seat.position,
                        )
    return game, snapshots


def select_snapshots(game, snapshots):
    """Pick one snapshot per scenario kind, or None if the hand lacks one."""
    if game.is_leaster or not game.picker or game.called_card is None:
        return None
    picked = {}
    # The eventual picker's own PICK/PASS moment.
    picked["pick"] = next(
        (s for s in snapshots["pick"] if s["seat"] == game.picker), None
    )
    picked["call"] = snapshots["call"][0] if snapshots["call"] else None
    picked["bury"] = snapshots["bury"][0] if snapshots["bury"] else None
    picked["lead"] = snapshots["lead"][0] if snapshots["lead"] else None
    # Prefer a defender follow with the most live tokens on screen;
    # tie-break toward later tricks (richer memory state).
    def follow_richness(s):
        hand_n = sum(1 for c in s["state"]["hand_ids"] if int(c) != 0)
        table_n = sum(1 for c in s["state"]["trick_card_ids"] if int(c) != 0)
        return (hand_n + table_n, s["trick"])

    follows = snapshots["follow"]
    defender_follows = [
        s for s in follows if s["seat"] not in (game.picker, game.partner)
    ]
    pool = defender_follows or follows
    picked["follow"] = max(pool, key=follow_richness) if pool else None
    if any(picked[k] is None for k in SCENARIO_ORDER):
        return None
    return picked


def find_hand(agent, max_seeds=500):
    for seed in range(max_seeds):
        game, snapshots = play_hand(agent, seed)
        picked = select_snapshots(game, snapshots)
        if picked is not None:
            return seed, game, picked, False
    print(
        "  no natural hand had all five decision types (mid-training "
        "checkpoints can pass every hand); re-scanning with a forced pick..."
    )
    for seed in range(max_seeds):
        game, snapshots = play_hand(agent, seed, force_pick=True)
        picked = select_snapshots(game, snapshots)
        if picked is not None:
            return seed, game, picked, True
    raise RuntimeError(f"No seed in [0, {max_seeds}) produced all scenario kinds")


def capture_forward(agent, state, valid_actions, memory_in):
    """Manually replicate PerceiverEncoder.encode_batch (mirrors encoder.py /
    architectures.py) so every intermediate — per-layer per-head attention,
    FFN hidden activations, both readout cross-attentions — can be captured,
    then run the actor and critic heads. Returns the per-scenario payload."""
    enc = agent.encoder
    device = next(enc.parameters()).device

    with torch.no_grad():
        memory_in = memory_in.view(1, enc.d_model).to(device)
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
        )  # (1, 19, d_token)
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

        # Run the transformer manually, capturing every layer's per-head
        # attention, post-attention/post-FFN token norms, and the FFN's
        # 128-d hidden activation norms.
        tokens = all_tokens_in
        attn_mask = ~all_mask
        layers_out = []
        for attn, ffn, ln1, ln2 in zip(
            enc.card_reasoner.attn_layers,
            enc.card_reasoner.ffn_layers,
            enc.card_reasoner.ln_attn,
            enc.card_reasoner.ln_ffn,
        ):
            attn_out, attn_w = attn(
                tokens,
                tokens,
                tokens,
                key_padding_mask=attn_mask,
                need_weights=True,
                average_attn_weights=False,
            )  # attn_w: (1, H, 19, 19)
            tokens = ln1(tokens + attn_out)
            norms_attn = token_norms(tokens)
            ffn_hidden = ffn[1](ffn[0](tokens))  # Linear -> SiLU, (1, 19, 2*d)
            ffn_out = ffn[2](ffn_hidden)
            tokens = ln2(tokens + ffn_out)
            layers_out.append(
                {
                    "attn": tensor_to_py(attn_w[0]),  # (H, 19, 19)
                    "token_norms_attn": norms_attn,
                    "token_norms_ffn": token_norms(tokens),
                    "ffn_hidden_norms": token_norms(ffn_hidden),
                }
            )

        all_tokens_post = tokens
        context_out = all_tokens_post[:, 0, :]
        memory_tok_out = all_tokens_post[:, 1, :]
        hand_tok_out = all_tokens_post[:, 2:10, :]
        trick_tok_out = all_tokens_post[:, 10:15, :]
        blind_tok_out = all_tokens_post[:, 15:17, :]
        bury_tok_out = all_tokens_post[:, 17:19, :]

        # Memory write: the post-reasoning MEMORY token (index 1), not the
        # context token — the perceiver's changed recurrence wiring.
        memory_out = enc.memory_gru(memory_tok_out, memory_in)

        # ---- Actor: readout cross-attention over the token set ----
        actor = agent.actor
        M = actor.readout_n_queries
        q = actor.readout_query.unsqueeze(0)  # (1, M, d_token)
        actor_ro_out, actor_ro_w = actor.readout_mha(
            q,
            all_tokens_post,
            all_tokens_post,
            key_padding_mask=~all_mask,
            need_weights=True,
            average_attn_weights=False,
        )  # (1, M, d_token), (1, H, M, 19)
        readout = actor.readout_proj(
            actor_ro_out.reshape(1, M * actor._d_token)
        )  # (1, d_model)
        feat = actor.actor_adapter(readout)

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
            {
                "features": memory_out,  # vestigial for the perceiver actor
                "hand_tokens": hand_tok_out,
                "all_tokens": all_tokens_post,
                "all_mask": all_mask,
            },
            action_mask_t,
            hand_ids,
            enc.card,
        )

        # ---- Critic: its own readout cross-attention ----
        critic = agent.critic
        Mc = critic.readout_n_queries
        qc = critic.readout_query.unsqueeze(0)
        critic_ro_out, critic_ro_w = critic.readout_mha(
            qc,
            all_tokens_post,
            all_tokens_post,
            key_padding_mask=~all_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        critic_readout = critic.readout_proj(
            critic_ro_out.reshape(1, Mc * critic._d_token)
        )
        value = critic.value_head(critic.value_trunk(critic_readout)).squeeze(-1)

    return {
        "tokens_pre_attn": {
            "context": tensor_to_py(context_tok[0]),
            "memory": tensor_to_py(memory_tok[0]),
            "hand": tensor_to_py(hand_tok[0]),  # (8, d_token)
            "trick": tensor_to_py(trick_tok[0]),  # (5, d_token)
            "blind": tensor_to_py(blind_tok[0]),  # (2, d_token)
            "bury": tensor_to_py(bury_tok[0]),  # (2, d_token)
            "mask": all_mask[0].tolist(),
        },
        "tokens_post_attn": {
            "context": tensor_to_py(context_out[0]),
            "memory": tensor_to_py(memory_tok_out[0]),
            "hand": tensor_to_py(hand_tok_out[0]),
            "trick": tensor_to_py(trick_tok_out[0]),
            "blind": tensor_to_py(blind_tok_out[0]),
            "bury": tensor_to_py(bury_tok_out[0]),
        },
        "transformer": {"layers": layers_out},
        "memory": {
            "memory_in": tensor_to_py(memory_in[0]),
            "memory_out": tensor_to_py(memory_out[0]),
        },
        "actor": {
            "readout": {
                "attn": tensor_to_py(actor_ro_w[0]),  # (H, M, 19)
                "vec": tensor_to_py(readout[0]),  # (d_model,)
            },
            "adapter_out": tensor_to_py(feat[0]),
            "pick_logits": tensor_to_py(pick_logits[0]),  # [PICK, PASS]
            "partner_basic_logits": tensor_to_py(partner_basic[0]),
            "play_under_logit": float(play_under_logit.item()),
            "two_tower_q": tensor_to_py(q_tw[0]),
            "card_scores_all": tensor_to_py(card_scores[0]),  # (34,)
            "slot_scores": tensor_to_py(slot_scores[0]),  # (8,)
            "full_probs": tensor_to_py(probs[0]),
            "full_logits": tensor_to_py(full_logits[0]),
            "valid_action_indices": (action_mask_t[0].nonzero().squeeze(-1).tolist()),
        },
        "critic": {
            "readout_attn": tensor_to_py(critic_ro_w[0]),  # (H, M, 19)
            "readout_vec": tensor_to_py(critic_readout[0]),
            "value": float(value.item()),
        },
    }


def describe_scenario(kind, snap, game):
    seat = snap["seat"]
    role = (
        "the picker"
        if seat == game.picker
        else ("the partner" if seat == game.partner else "a defender")
    )
    hand_codes = [
        ID_TO_CODE[int(c)] for c in snap["state"]["hand_ids"] if int(c) != 0
    ]
    hand_str = ", ".join(hand_codes)
    if kind == "pick":
        forced_note = (
            " (Forced to pick for this capture — the checkpoint preferred "
            "PASS here.)"
            if snap.get("forced")
            else " (This seat goes on to pick.)"
        )
        return f"Seat {seat} holds {hand_str} and must PICK or PASS.{forced_note}"
    if kind == "call":
        return (
            f"Seat {seat} picked and now holds {hand_str} (8 cards after the "
            f"blind). It must choose a partner card — the two-tower CALL "
            f"scorer is the live head here."
        )
    if kind == "bury":
        return (
            f"Seat {seat} (the picker) holds {hand_str} and must bury its "
            f"first card — the pointer head scores every hand slot."
        )
    if kind == "lead":
        return (
            f"Seat {seat} ({role}) leads trick 1 with {hand_str}. The trick "
            f"is empty; memory now carries the pick phase."
        )
    return (
        f"Trick {snap['trick'] + 1}: seat {seat} ({role}) follows with "
        f"{hand_str} after several cards hit the table. Trick tokens are "
        f"live and the memory token carries every completed trick."
    )


def build_sample_block(kind, snap, game):
    state = snap["state"]
    valid_sorted = sorted(snap["valid_actions"])
    hand_ids = [int(c) for c in state["hand_ids"]]
    callable_cards = [
        {"code": n[len("CALL "):].replace(" UNDER", ""), "under": n.endswith(" UNDER")}
        for n in (ACTION_LOOKUP[a] for a in valid_sorted)
        if n.startswith("CALL ")
    ]
    return {
        "kind": kind,
        "seat": snap["seat"],
        "trick": snap["trick"],
        "hand_cards": [ID_TO_CODE[c] for c in hand_ids if c != 0],
        "hand_ids": hand_ids,
        "trick_card_codes": [
            ID_TO_CODE[int(c)] for c in state["trick_card_ids"]
        ],
        "valid_actions": [ACTION_LOOKUP[a] for a in valid_sorted],
        "chosen_action": snap["chosen_action"],
        "callable_cards": callable_cards,
        "is_picker": snap["seat"] == game.picker,
        "is_partner": snap["seat"] == game.partner,
        "called_card": game.called_card,
        "phase": SCENARIO_LABELS[kind],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Perceiver-arch checkpoint to visualize (loaded via ppo.load_agent).",
    )
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    agent = load_agent(str(args.checkpoint), load_optimizers=False)
    enc = agent.encoder
    if not isinstance(enc, PerceiverEncoder):
        raise SystemExit(
            f"Checkpoint arch is not a perceiver variant (encoder is "
            f"{type(enc).__name__}); this dump captures the perceiver "
            f"forward pass only."
        )
    assert isinstance(agent.actor, PerceiverActorNetwork)
    assert isinstance(agent.critic, PerceiverCriticNetwork)
    enc.eval()
    agent.actor.eval()
    agent.critic.eval()

    n_layers = len(enc.card_reasoner.attn_layers)
    n_heads = enc.card_reasoner.attn_layers[0].num_heads

    print("Scanning seeds for a hand with all five decision types...")
    seed, game, picked, forced_pick = find_hand(agent)
    print(
        f"  seed={seed} picker=seat {game.picker} partner=seat {game.partner} "
        f"called={game.called_card}"
    )

    trump_card_ids = [DECK_IDS[c] for c in TRUMP]
    scenarios = []
    for kind in SCENARIO_ORDER:
        snap = picked[kind]
        payload = capture_forward(
            agent, snap["state"], snap["valid_actions"], snap["memory_in"]
        )
        payload["id"] = kind
        payload["label"] = SCENARIO_LABELS[kind]
        payload["description"] = describe_scenario(kind, snap, game)
        payload["sample"] = build_sample_block(kind, snap, game)
        scenarios.append(payload)
        probs = payload["actor"]["full_probs"]
        top_idx = max(
            payload["actor"]["valid_action_indices"], key=lambda i: probs[i]
        )
        print(
            f"  [{kind:6s}] seat {snap['seat']} chose {snap['chosen_action']:12s} "
            f"argmax={ACTIONS[top_idx]} ({probs[top_idx]:.3f}) "
            f"V={payload['critic']['value']:+.3f}"
        )

    out = {
        "checkpoint": str(
            args.checkpoint.resolve().relative_to(ROOT)
            if args.checkpoint.resolve().is_relative_to(ROOT)
            else args.checkpoint
        ),
        "arch": "perceiver",
        "generated": date.today().isoformat(),
        "seed": seed,
        "dims": {
            "d_card": enc.d_card_dim,
            "d_token": enc.d_token_dim,
            "d_model": enc.d_model,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "n_actor_queries": agent.actor.readout_n_queries,
            "n_critic_queries": agent.critic.readout_n_queries,
        },
        "hand_summary": {
            "picker": int(game.picker),
            "forced_pick": bool(forced_pick),
            "partner": int(game.partner),
            "called_card": game.called_card,
            "picker_points": int(game.get_final_picker_points()),
            "defender_points": int(game.get_final_defender_points()),
        },
        "card_embedding": {
            "shape": list(enc.card.weight.shape),
            "table": tensor_to_py(enc.card.weight),  # full 34 x d_card
            "min": float(enc.card.weight.min().item()),
            "max": float(enc.card.weight.max().item()),
        },
        "scenarios": scenarios,
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
    print(f"Wrote {OUT_JSON.name} ({size_mb:.2f} MB, {len(scenarios)} scenarios)")


if __name__ == "__main__":
    main()
