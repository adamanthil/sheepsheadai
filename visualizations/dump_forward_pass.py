"""Play one deterministic hand with a trained perceiver-shared-v2 PPO agent
and dump the forward-pass activations at several decision points to JSON for
the 3D architecture visualization.

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

Captured per decision: the whole transformer (per-layer, per-head attention
maps, per-layer token norms, FFN hidden-activation norms), the encoder's
SHARED readout cross-attention (16 learned queries x 4 heads over the 19
post-reasoning tokens -> one 256-d `features` vector both networks consume),
the actor's opened-up heads (two-tower CALL scores, Bahdanau pointer
intermediates over the hand tokens), and the critic's value + full auxiliary
stack (win/return/secret-partner/points and the trump tracker).

The dump ALSO captures the ORACLE CRITIC (oracle.py: OracleValueNetwork, the
CTDE privileged critic) on the same five decision states: its 51-token
full-information forward pass (per-layer attention stored sparse top-K),
memory-token GRU recurrence threaded over the same per-seat event stream the
agent's memory sees, its 4-query readout, and U(h,s). With no trained oracle
checkpoint yet, the default is a fresh seeded random init (--oracle-seed);
the manual capture is cross-checked against the module's own encode_batch,
so the dump doubles as an architecture smoke test. Pass --oracle-checkpoint
to use a league checkpoint's `oracle_state_dict` once one exists.
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

from architectures import SharedReadoutEncoder  # noqa: E402
from oracle import OracleValueNetwork  # noqa: E402
from ppo import (  # noqa: E402
    MultiHeadRecurrentActorNetwork,
    RecurrentCriticNetwork,
    load_agent,
)
from sheepshead import (  # noqa: E402
    ACTION_IDS,
    ACTION_LOOKUP,
    ACTIONS,
    DECK_IDS,
    PARTNER_BY_CALLED_ACE,
    TRUMP,
    Game,
)

DEFAULT_CHECKPOINT = (
    ROOT
    / "runs"
    / "ablate_perceiver-shared-v2_s42"
    / "perceiver-shared-v2_checkpoint_175000.pt"
)
OUT_JSON = HERE / "ppo_forward_pass.json"

# Oracle critic (untrained smoke test until the league run produces one).
ORACLE_INIT_SEED = 20260709
ORACLE_ATTN_TOPK = 400
# The oracle runs on CPU end-to-end: 5 captures + ~40 memory advances are
# trivial, and a fixed device keeps the seeded random init reproducible.
ORACLE_DEVICE = torch.device("cpu")

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


def round4(seq):
    """Round a flat float sequence to 4 dp (keeps the oracle JSON compact)."""
    return [round(float(v), 4) for v in seq]


def sparse_attn_topk(attn_hnn: torch.Tensor, mask, k=ORACLE_ATTN_TOPK):
    """Top-k directed attention triples of a (H, N, N) map.

    Returns [[h, i, j, w], ...] sorted by weight descending, restricted to
    i != j with both tokens live. Dense 4x51x51 maps would add ~2.5 MB per
    dump; the tunnel only draws the strongest chords anyway.
    """
    H, N, _ = attn_hnn.shape
    live = [i for i in range(N) if mask[i]]
    triples = []
    for h in range(H):
        layer = attn_hnn[h]
        for i in live:
            row = layer[i]
            for j in live:
                if j != i:
                    triples.append((float(row[j]), h, i, j))
    triples.sort(reverse=True)
    return [[h, i, j, round(w, 4)] for w, h, i, j in triples[:k]]


def load_oracle(args):
    """Instantiate the oracle critic: a league checkpoint's oracle_state_dict
    when --oracle-checkpoint is given, else a fresh seeded random init."""
    if args.oracle_checkpoint is not None:
        ckpt = torch.load(
            args.oracle_checkpoint, map_location="cpu", weights_only=False
        )
        if "oracle_state_dict" not in ckpt:
            raise SystemExit(
                f"{args.oracle_checkpoint} has no 'oracle_state_dict' — it was "
                f"not saved by a --critic-mode oracle trainer."
            )
        net = OracleValueNetwork()
        net.load_state_dict(ckpt["oracle_state_dict"])
        untrained = False
    else:
        # Seed immediately before construction so the init is independent of
        # whatever RNG the policy load consumed.
        torch.manual_seed(args.oracle_seed)
        net = OracleValueNetwork()
        untrained = True
    net.to(ORACLE_DEVICE).eval()
    return net, untrained


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


def play_hand(agent, seed, force_pick=False, oracle=None):
    """Play one full deterministic hand, snapshotting candidate decisions.

    Returns (game, snapshots) where snapshots maps kind -> list of dicts with
    the state, valid actions, pre-decision memory, and chosen action.

    With force_pick=True, the last seat to see a PICK opportunity is forced
    to PICK (mid-training checkpoints can PASS every hand into a leaster, and
    then the call/bury phases never occur). Every captured forward pass is
    still the genuine policy on a real game state; only that one action is
    overridden, and its snapshot is marked "forced".

    With oracle set (an OracleValueNetwork), per-seat oracle memory is
    threaded over the same event stream the agent's memory sees — each seat's
    decision states plus its end-of-trick observations, zero-init per hand,
    matching the training recurrence protocol — and every snapshot gains the
    decision-time privileged state + pre-decision oracle memory.
    """
    game = Game(partner_selection_mode=PARTNER_BY_CALLED_ACE, seed=seed)
    agent.reset_recurrent_state()
    snapshots = {k: [] for k in SCENARIO_ORDER}
    oracle_mem = None
    if oracle is not None:
        d_model = oracle.encoder.d_model
        oracle_mem = {
            p.position: torch.zeros(1, d_model, device=ORACLE_DEVICE)
            for p in game.players
        }

    while not game.is_done():
        for player in game.players:
            valid_actions = player.get_valid_action_ids()
            while valid_actions:
                state = player.get_state_dict()
                # Privileged state must be captured pre-action (hands shrink).
                ostate = (
                    player.get_oracle_state_dict() if oracle is not None else None
                )
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
                    if oracle is not None:
                        snap["oracle_state"] = copy.deepcopy(ostate)
                        # Clone BEFORE the decision advance below.
                        snap["oracle_memory_in"] = oracle_mem[
                            player.position
                        ].clone()

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
                # Oracle decision event: advance this seat's oracle memory
                # (forced picks included — the event stream is defined by
                # decisions, not by who chose the action).
                if oracle is not None:
                    with torch.no_grad():
                        out = oracle.encoder.encode_batch(
                            [ostate],
                            memory_in=oracle_mem[player.position],
                            device=ORACLE_DEVICE,
                        )
                    oracle_mem[player.position] = out["memory_out"].detach()
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
                        if oracle is not None:
                            with torch.no_grad():
                                oo = oracle.encoder.encode_batch(
                                    [seat.get_last_trick_oracle_state_dict()],
                                    memory_in=oracle_mem[seat.position],
                                    device=ORACLE_DEVICE,
                                )
                            oracle_mem[seat.position] = oo["memory_out"].detach()
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


def capture_transformer(card_reasoner, tokens, all_mask):
    """Manually run a TransformerCardReasoning stack, capturing every layer's
    per-head attention, post-attention/post-FFN token norms, and the FFN's
    hidden-activation norms at the 128-d bottleneck.

    Both the policy encoder and the oracle encoder inherit the same module
    lists (attn_layers/ffn_layers/ln_attn/ln_ffn), so one capture loop serves
    both. Returns (tokens_out, layers) where layers[l]["attn"] is the raw
    (H, N, N) tensor — callers serialize dense (policy) or sparse (oracle).
    """
    attn_mask = ~all_mask
    layers = []
    for attn, ffn, ln1, ln2 in zip(
        card_reasoner.attn_layers,
        card_reasoner.ffn_layers,
        card_reasoner.ln_attn,
        card_reasoner.ln_ffn,
    ):
        attn_out, attn_w = attn(
            tokens,
            tokens,
            tokens,
            key_padding_mask=attn_mask,
            need_weights=True,
            average_attn_weights=False,
        )  # attn_w: (1, H, N, N)
        tokens = ln1(tokens + attn_out)
        norms_attn = token_norms(tokens)
        ffn_hidden = ffn[1](ffn[0](tokens))  # Linear -> SiLU, (1, N, 2*d)
        ffn_out = ffn[2](ffn_hidden)
        tokens = ln2(tokens + ffn_out)
        layers.append(
            {
                "attn": attn_w[0],  # (H, N, N) tensor
                "token_norms_attn": norms_attn,
                "token_norms_ffn": token_norms(tokens),
                "ffn_hidden_norms": token_norms(ffn_hidden),
            }
        )
    return tokens, layers


def capture_forward(agent, state, valid_actions, memory_in):
    """Manually replicate PerceiverEncoder.encode_batch (mirrors encoder.py /
    the architectures package) so every intermediate — per-layer per-head attention,
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
        # 128-d hidden activation norms. Policy maps stay dense on disk
        # (19x19 is small).
        all_tokens_post, cap_layers = capture_transformer(
            enc.card_reasoner, all_tokens_in, all_mask
        )
        layers_out = [
            {
                "attn": tensor_to_py(L["attn"]),  # (H, 19, 19)
                "token_norms_attn": L["token_norms_attn"],
                "token_norms_ffn": L["token_norms_ffn"],
                "ffn_hidden_norms": L["ffn_hidden_norms"],
            }
            for L in cap_layers
        ]
        context_out = all_tokens_post[:, 0, :]
        memory_tok_out = all_tokens_post[:, 1, :]
        hand_tok_out = all_tokens_post[:, 2:10, :]
        trick_tok_out = all_tokens_post[:, 10:15, :]
        blind_tok_out = all_tokens_post[:, 15:17, :]
        bury_tok_out = all_tokens_post[:, 17:19, :]

        # Memory write: v2 drives the GRU from the post-reasoning CONTEXT
        # token (index 0, as in `full`); the original perceiver used the
        # MEMORY token. Honor the encoder's flag so both wirings dump right.
        driver_idx = 1 if enc.memory_token_driver else 0
        memory_out = enc.memory_gru(all_tokens_post[:, driver_idx, :], memory_in)

        # ---- Shared encoder readout: one 16-query x 4-head cross-attention
        # over the token set produces the 256-d `features` vector that BOTH
        # the actor and the critic consume (no per-network readouts).
        M = enc.readout_n_queries
        q = enc.readout_query.unsqueeze(0)  # (1, M, d_token)
        ro_out, ro_w = enc.readout_mha(
            q,
            all_tokens_post,
            all_tokens_post,
            key_padding_mask=~all_mask,
            need_weights=True,
            average_attn_weights=False,
        )  # (1, M, d_token), (1, H, M, 19)
        features = enc.readout_proj(
            ro_out.reshape(1, M * enc.d_token_dim)
        )  # (1, d_model); v2 readout_proj = Linear + LayerNorm

        # ---- Actor: adapter + heads on the shared features ----
        actor = agent.actor
        feat = actor.actor_adapter(features)

        pick_logits = actor.pick_head(feat)  # (1, 2)
        partner_basic = actor.partner_basic_head(feat)  # (1, 2)
        play_under_logit = actor.play_under_head(feat).squeeze(-1)  # (1,)

        # two-tower CALL scorer: query from features vs the whole card table
        q_tw = actor.tw_Wg(feat)  # (1, 64)
        K_tw = actor.tw_We(enc.card.weight)  # (34, 64)
        card_scores = torch.matmul(q_tw, K_tw.t())  # (1, 34)

        # Pointer over hand slots, opened up so the viz can show the
        # Bahdanau combine: score_i = v . tanh(Wg(feat) + Wt(token_i))
        ptr_g = actor.pointer_Wg(feat)  # (1, h)
        ptr_t = actor.pointer_Wt(hand_tok_out)  # (1, 8, h)
        ptr_hidden = torch.tanh(ptr_g.unsqueeze(1) + ptr_t)  # (1, 8, h)
        slot_scores = actor.pointer_v(ptr_hidden).squeeze(-1)  # (1, 8)

        # full action logits via the model
        action_mask_t = (
            agent.get_action_mask(valid_actions, agent.action_size)
            .unsqueeze(0)
            .to(device)
        )
        probs, full_logits = actor.forward_with_logits(
            {
                "features": features,
                "hand_tokens": hand_tok_out,
                "all_tokens": all_tokens_post,
                "all_mask": all_mask,
            },
            action_mask_t,
            hand_ids,
            enc.card,
        )

        # ---- Critic: value trunk + auxiliary stack on the same features ----
        critic = agent.critic
        value_feat = critic.value_trunk(features)
        value = critic.value_head(value_feat).squeeze(-1)
        aux_feat = critic.critic_adapter(features)
        win_prob = torch.sigmoid(critic.win_head(aux_feat)).squeeze(-1)
        expected_return = critic.return_head(aux_feat).squeeze(-1)
        secret_prob = torch.sigmoid(critic.secret_partner_head(aux_feat)).squeeze(-1)
        points_pred = torch.clamp(critic.points_head(aux_feat), min=0.0, max=120.0)
        seen_trump_probs = torch.sigmoid(
            critic.seen_trump_mask_logits(aux_feat, enc.card)
        )  # (1, 14)
        unseen_higher_prob = torch.sigmoid(
            critic.unseen_trump_higher_than_hand_logits(aux_feat)
        )  # (1,)

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
            "driver": "memory" if enc.memory_token_driver else "context",
        },
        "encoder_readout": {
            "attn": tensor_to_py(ro_w[0]),  # (H, M, 19)
            "features": tensor_to_py(features[0]),  # (d_model,)
        },
        "actor": {
            "adapter_out": tensor_to_py(feat[0]),
            "pick_logits": tensor_to_py(pick_logits[0]),  # [PICK, PASS]
            "partner_basic_logits": tensor_to_py(partner_basic[0]),
            "play_under_logit": float(play_under_logit.item()),
            "two_tower_q": tensor_to_py(q_tw[0]),
            "card_scores_all": tensor_to_py(card_scores[0]),  # (34,)
            "pointer": {
                "g_norm": float(ptr_g[0].norm().item()),
                "t_norms": ptr_t[0].norm(dim=-1).tolist(),  # (8,)
                "hidden_norms": ptr_hidden[0].norm(dim=-1).tolist(),  # (8,)
                "slot_scores": tensor_to_py(slot_scores[0]),  # (8,)
            },
            "full_probs": tensor_to_py(probs[0]),
            "full_logits": tensor_to_py(full_logits[0]),
            "valid_action_indices": (action_mask_t[0].nonzero().squeeze(-1).tolist()),
        },
        "critic": {
            "value": float(value.item()),
            "value_trunk_norm": float(value_feat[0].norm().item()),
            "aux": {
                "win_prob": float(win_prob.item()),
                "expected_return": float(expected_return.item()),
                "secret_partner_prob": float(secret_prob.item()),
                "points": tensor_to_py(points_pred[0]),  # (5,) relative seats
                "seen_trump_probs": tensor_to_py(seen_trump_probs[0]),  # (14,)
                "unseen_higher_prob": float(unseen_higher_prob.item()),
            },
        },
    }


def capture_oracle_forward(oracle, ostate, memory_in):
    """Manually replicate OracleCriticEncoder.encode_batch (mirrors
    oracle.py) over the 51-token full-information layout, capturing per-layer
    attention (serialized sparse top-K), the 4-query readout cross-attention,
    and U(h,s) — then cross-check every stage against the module's own
    forward. With an untrained oracle this doubles as the architecture
    smoke test."""
    enc = oracle.encoder
    device = ORACLE_DEVICE

    with torch.no_grad():
        memory_in = memory_in.view(1, enc.d_model).to(device)
        batch = [ostate]

        # Context token: base header (normalized) + privileged scalars +
        # called AND under card embeddings.
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

        points_rel = (
            torch.as_tensor(
                ostate["points_taken_rel"], dtype=torch.float32, device=device
            ).view(1, 5)
            / 120.0
        )
        secret_rel_raw = torch.tensor(
            [int(ostate["secret_partner_rel"])], device=device, dtype=torch.long
        )
        secret_scalar = secret_rel_raw.float().view(1, 1) / 5.0
        called_ids = torch.tensor(
            [int(ostate["called_card_id"])], device=device, dtype=torch.long
        )
        under_ids = torch.tensor(
            [int(ostate["under_card_id"])], device=device, dtype=torch.long
        )
        context_tok = enc.context_mlp(
            torch.cat(
                [
                    header_scalar_n,
                    points_rel,
                    secret_scalar,
                    enc.card(called_ids),
                    enc.card(under_ids),
                ],
                dim=1,
            )
        )
        memory_tok = enc.memory_in_proj(memory_in)

        picker_rel_raw = torch.tensor(
            [int(ostate["picker_rel"])], device=device, dtype=torch.long
        )
        partner_rel_raw = torch.tensor(
            [int(ostate["partner_rel"])], device=device, dtype=torch.long
        )
        actor_role_id = (
            picker_rel_raw.eq(1).long() * 1 + partner_rel_raw.eq(1).long() * 2
        )

        hand_ids = torch.as_tensor(
            ostate["hand_ids"], dtype=torch.long, device=device
        ).view(1, 8)
        # TRUE blind/bury for every seat — the privileged fields.
        blind_ids = torch.as_tensor(
            ostate["blind_ids"], dtype=torch.long, device=device
        ).view(1, 2)
        bury_ids = torch.as_tensor(
            ostate["bury_ids"], dtype=torch.long, device=device
        ).view(1, 2)
        trick_card_ids = torch.as_tensor(
            ostate["trick_card_ids"], dtype=torch.long, device=device
        ).view(1, 5)
        trick_is_picker = (
            torch.as_tensor(ostate["trick_is_picker"], dtype=torch.long, device=device)
            .view(1, 5)
            .bool()
        )
        trick_is_partner_known = (
            torch.as_tensor(
                ostate["trick_is_partner_known"], dtype=torch.long, device=device
            )
            .view(1, 5)
            .bool()
        )
        opp_ids = torch.as_tensor(
            ostate["opp_hand_ids"], dtype=torch.long, device=device
        ).view(1, 32)

        hand_tok, hand_mask = enc._embed_hand(hand_ids, actor_role_id)
        blind_tok, blind_mask = enc._embed_simple_bag(blind_ids)
        bury_tok, bury_mask = enc._embed_simple_bag(bury_ids)
        trick_tok, trick_mask = enc._embed_trick(
            trick_card_ids, trick_is_picker, trick_is_partner_known
        )
        opp_tok, opp_mask = enc._embed_opp_hands(
            opp_ids, picker_rel_raw, secret_rel_raw
        )

        all_tokens_pre = torch.cat(
            [
                context_tok.unsqueeze(1),
                memory_tok.unsqueeze(1),
                hand_tok,
                trick_tok,
                blind_tok,
                bury_tok,
                opp_tok,
            ],
            dim=1,
        )  # (1, 51, d_token)
        ones = torch.ones((1, 1), dtype=torch.bool, device=device)
        all_mask = torch.cat(
            [ones, ones, hand_mask, trick_mask, blind_mask, bury_mask, opp_mask],
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
                torch.full((1, 32), enc.OPP_TYPE_ID, dtype=torch.long, device=device),
            ],
            dim=1,
        )
        all_tokens_in = all_tokens_pre + enc.card_type(type_ids)

        tokens_out, cap_layers = capture_transformer(
            enc.card_reasoner, all_tokens_in, all_mask
        )
        # Memory write: the post-reasoning MEMORY token (index 1) — the
        # oracle keeps the original perceiver wiring.
        memory_out = enc.memory_gru(tokens_out[:, 1, :], memory_in)

        # Readout re-run with weights (oracle._readout hardcodes
        # need_weights=False).
        q = oracle.readout_query.unsqueeze(0)
        ro_out, ro_w = oracle.readout_mha(
            q,
            tokens_out,
            tokens_out,
            key_padding_mask=~all_mask,
            need_weights=True,
            average_attn_weights=False,
        )  # (1, M, d_token), (1, H, M, 51)
        features = oracle.readout_proj(
            ro_out.reshape(1, oracle.readout_n_queries * oracle._d_token)
        )
        value_feat = oracle.value_trunk(features)
        value = oracle.value_head(value_feat).squeeze(-1)

        # ---- Smoke-test cross-check: the manual replication must match the
        # module's own forward exactly.
        ref = enc.encode_batch([ostate], memory_in=memory_in, device=device)
        assert bool((all_mask == ref["all_mask"]).all()), "oracle mask mismatch"
        assert torch.allclose(tokens_out, ref["all_tokens"], atol=1e-5), (
            "oracle capture: post-reasoning tokens diverge from encode_batch"
        )
        assert torch.allclose(memory_out, ref["memory_out"], atol=1e-5), (
            "oracle capture: memory_out diverges from encode_batch"
        )
        ref_value = oracle.value_head(
            oracle.value_trunk(oracle._readout(ref["all_tokens"], ref["all_mask"]))
        ).squeeze(-1)
        assert torch.allclose(value, ref_value, atol=1e-5), (
            "oracle capture: U(h,s) diverges from the module forward"
        )

    mask_list = all_mask[0].tolist()

    def codes(ids):
        return [ID_TO_CODE[int(c)] for c in ids if int(c) != 0]

    opp_rows = [[int(c) for c in row] for row in ostate["opp_hand_ids"]]
    return {
        "sample": {
            "opp_hand_ids": opp_rows,
            "opp_hand_codes": [codes(row) for row in opp_rows],
            "blind_cards": codes(ostate["blind_ids"]),
            "bury_cards": codes(ostate["bury_ids"]),
            "under_card": (
                ID_TO_CODE[int(ostate["under_card_id"])]
                if int(ostate["under_card_id"])
                else None
            ),
            "secret_partner_rel": int(ostate["secret_partner_rel"]),
            "points_taken_rel": [int(p) for p in ostate["points_taken_rel"]],
            "picker_rel": int(ostate["picker_rel"]),
            "partner_rel": int(ostate["partner_rel"]),
        },
        "tokens_pre": {
            "norms": round4(all_tokens_pre[0].norm(dim=-1).tolist()),
            "mask": mask_list,
        },
        "transformer": {
            "layers": [
                {
                    "attn_topk": sparse_attn_topk(L["attn"], mask_list),
                    "k": ORACLE_ATTN_TOPK,
                    "token_norms_attn": round4(L["token_norms_attn"]),
                    "token_norms_ffn": round4(L["token_norms_ffn"]),
                    "ffn_hidden_norms": round4(L["ffn_hidden_norms"]),
                }
                for L in cap_layers
            ]
        },
        "memory": {
            "memory_in": round4(memory_in[0].tolist()),
            "memory_out": round4(memory_out[0].tolist()),
            "driver": "memory",
        },
        "encoder_readout": {
            "attn": [
                [round4(row) for row in per_query]
                for per_query in ro_w[0].tolist()
            ],  # (H, M, 51)
            "features": round4(features[0].tolist()),
        },
        "value": {
            "value": float(value.item()),
            "value_trunk_norm": float(value_feat[0].norm().item()),
            "features_norm": float(features[0].norm().item()),
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
        help=(
            "perceiver-shared-v2-arch checkpoint to visualize "
            "(loaded via ppo.load_agent)."
        ),
    )
    parser.add_argument(
        "--oracle-checkpoint",
        type=Path,
        default=None,
        help=(
            "Checkpoint holding an 'oracle_state_dict' (saved by a "
            "--critic-mode oracle trainer). Default: fresh random init."
        ),
    )
    parser.add_argument(
        "--oracle-seed",
        type=int,
        default=ORACLE_INIT_SEED,
        help="torch seed for the random-init oracle (ignored with a checkpoint).",
    )
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    agent = load_agent(str(args.checkpoint), load_optimizers=False)
    enc = agent.encoder
    if not isinstance(enc, SharedReadoutEncoder):
        raise SystemExit(
            f"Checkpoint arch is not a shared-readout perceiver variant "
            f"(encoder is {type(enc).__name__}); this dump captures the "
            f"perceiver-shared-v2 forward pass only."
        )
    assert type(agent.actor) is MultiHeadRecurrentActorNetwork
    assert type(agent.critic) is RecurrentCriticNetwork
    assert agent.critic.has_aux_heads, "expected the aux-head critic stack"
    enc.eval()
    agent.actor.eval()
    agent.critic.eval()

    n_layers = len(enc.card_reasoner.attn_layers)
    n_heads = enc.card_reasoner.attn_layers[0].num_heads

    oracle_net, oracle_untrained = load_oracle(args)
    n_oracle_params = sum(p.numel() for p in oracle_net.parameters())
    oracle_src = (
        f"random init (untrained), seed {args.oracle_seed}"
        if oracle_untrained
        else str(args.oracle_checkpoint)
    )
    print(f"Oracle critic: {oracle_src} · {n_oracle_params:,} params")

    print("Scanning seeds for a hand with all five decision types...")
    seed, game, picked, forced_pick = find_hand(agent)
    print(
        f"  seed={seed} picker=seat {game.picker} partner=seat {game.partner} "
        f"called={game.called_card}"
    )

    # Replay the winning hand once with oracle memory threading (the scan
    # itself stays oracle-free — 500 seeds x 51-token encodes is waste).
    print("Replaying the hand with oracle-critic memory threading...")
    game2, snapshots2 = play_hand(
        agent, seed, force_pick=forced_pick, oracle=oracle_net
    )
    picked2 = select_snapshots(game2, snapshots2)
    assert (
        picked2 is not None
        and game2.picker == game.picker
        and game2.partner == game.partner
        and game2.called_card == game.called_card
    ), "oracle replay diverged from the scanned hand"
    game, picked = game2, picked2

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
        payload["oracle"] = capture_oracle_forward(
            oracle_net, snap["oracle_state"], snap["oracle_memory_in"]
        )
        scenarios.append(payload)
        probs = payload["actor"]["full_probs"]
        top_idx = max(
            payload["actor"]["valid_action_indices"], key=lambda i: probs[i]
        )
        print(
            f"  [{kind:6s}] seat {snap['seat']} chose {snap['chosen_action']:12s} "
            f"argmax={ACTIONS[top_idx]} ({probs[top_idx]:.3f}) "
            f"V={payload['critic']['value']:+.3f} "
            f"U={payload['oracle']['value']['value']:+.3f}"
        )

    out = {
        "checkpoint": str(
            args.checkpoint.resolve().relative_to(ROOT)
            if args.checkpoint.resolve().is_relative_to(ROOT)
            else args.checkpoint
        ),
        "arch": agent.arch_name,
        "generated": date.today().isoformat(),
        "seed": seed,
        "dims": {
            "d_card": enc.d_card_dim,
            "d_token": enc.d_token_dim,
            "d_model": enc.d_model,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "n_readout_queries": enc.readout_n_queries,
            "n_readout_heads": enc.readout_mha.num_heads,
            "pointer_hidden": agent.actor.pointer_hidden,
            "tw_latent": agent.actor.tw_latent,
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
        "oracle": {
            "untrained": oracle_untrained,
            "init_seed": args.oracle_seed if oracle_untrained else None,
            "checkpoint": (
                None
                if args.oracle_checkpoint is None
                else str(
                    args.oracle_checkpoint.resolve().relative_to(ROOT)
                    if args.oracle_checkpoint.resolve().is_relative_to(ROOT)
                    else args.oracle_checkpoint
                )
            ),
            "n_params": n_oracle_params,
            "dims": {
                "d_card": oracle_net.encoder.d_card_dim,
                "d_token": oracle_net.encoder.d_token_dim,
                "d_model": oracle_net.encoder.d_model,
                "n_layers": len(oracle_net.encoder.card_reasoner.attn_layers),
                "n_heads": oracle_net.encoder.card_reasoner.attn_layers[0].num_heads,
                "n_tokens": 51,
                "n_readout_queries": oracle_net.readout_n_queries,
                "n_readout_heads": oracle_net.readout_mha.num_heads,
                "attn_top_k": ORACLE_ATTN_TOPK,
            },
            # The oracle's OWN table — zero parameters shared with the policy.
            "card_embedding": {
                "shape": list(oracle_net.encoder.card.weight.shape),
                "table": [
                    round4(row) for row in oracle_net.encoder.card.weight.tolist()
                ],
                "min": float(oracle_net.encoder.card.weight.min().item()),
                "max": float(oracle_net.encoder.card.weight.max().item()),
            },
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
