#!/usr/bin/env python3
"""Offline distillation corpus generator (CE_Teacher_Design §17.3).

Plays frozen-theta_k self-play games (all five seats collected, per-seat
episode streams) and annotates every decision with its §16.9-addendum-6
policy-loss partition:

  override   searched, material (shrink w > 0): carries the §1.1 shrink-
             and-tilt CE target plus the top-2 pooled-Q gap for the
             trainer's omega evidence weight (AWR-family — Peng et al.
             2019, arXiv:1910.00177).
  endorsed   searched, abstained (w = 0): search spoke and endorsed the
             policy — the row anchors to theta_k in the trainer.
  retention  search cannot speak there in this corpus: bidding heads
             (pick/partner/bury — ALONE declaration included) and
             leaster play. Alone-game PLAY is searched by default
             (operator amendment 2026-08-21: same token-pointer play
             head as standard play, and the 1v4 determinization has no
             hidden-partner uncertainty; §17.6 records its noise floor).
  none       eligible-but-unsearched play (the p-schedule passed it
             over), forced nodes, and committee failures: no policy
             loss, still in the value-regression stream.

The invariant (§16.9 addendum 3): a row carries an anchor only if search
CANNOT speak there or SPOKE AND ENDORSED — never merely unasked, which
would teach the policy to distinguish searched twins from unsearched ones.

Offline phase purity (Expert Iteration — Anthony et al. 2017; AlphaGo
Zero / AlphaZero — Silver et al. 2017/2018): nothing updates during
generation, so the expert, the acting policy and the anchor are all the
SAME frozen network — the attempt-7 drifting-expert and attempt-11/12
CE-x-PG interaction mechanisms structurally cannot arise here.

State distribution: stochastic self-play acting = on-policy states
(DAgger — Ross et al. 2011). A pre-registered fraction of games are
COMMITTEE-ACTING (§16.9 addendum 5; AggreVaTe — Ross & Bagnell 2014,
scheduled sampling — Bengio et al. 2015): at material searched nodes the
seat acts the CE target's argmax, so states downstream of the expert's
improvements enter the corpus and every material search doubles as a
label row.

Anchors are theta_k's act-time probability vectors (the ``act()``
stash): a DIRECT forward output at the true recurrent state of the
realized trajectory, which the trainer's replayed unroll reproduces to
replay noise — the engine's forced replay (and its trick-4 recurrent
divergence artifact, §13 phase 1) is never used for anchors.

Node telemetry (``--node-telemetry``) writes one JSONL row per searched
node (class, regime, w, gap, spread, per-replicate top-pair Q diffs) —
the §17.6 alone-noise calibration instrument and the iteration-2
p-schedule refinement input.

Usage:
  uv run python -m sheepshead.training.distill_corpus \\
      --ckpt runs/league_retention_pg/checkpoints/..._checkpoint_8000000.pt \\
      --out-dir runs/distill_corpus_202608 --games 20000 --workers 8 \\
      --node-telemetry runs/distill_corpus_202608/nodes.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from multiprocessing import get_context

import numpy as np

from sheepshead import (
    PARTNER_BY_CALLED_ACE,
    PARTNER_BY_JD,
    TRUMP,
    Game,
)
from sheepshead.training.training_utils import RETURN_SCALE

_W: dict = {}  # per-worker state (agent, teacher, config)


# --------------------------------------------------------------------------- #
# p-schedule (§17.2)
# --------------------------------------------------------------------------- #
def lead_features(game, player) -> tuple[bool, bool]:
    """(is_lead, called_suit_eligible) for a standard-game play decision.

    called_suit_eligible mirrors the calibrated-instrument definition
    (ceiling study / E9): a non-partnerish seat leading while the called
    suit is unplayed, holding both a called-suit fail card and an
    alternative."""
    is_lead = all(c == "" for c in game.history[game.current_trick])
    if not is_lead:
        return False, False
    is_partnerish = player.is_picker or player.is_partner or player.is_secret_partner
    called = game.called_card
    cs = (
        not is_partnerish
        and bool(called)
        and not game.was_called_suit_played
        and any(c not in TRUMP and c[-1] == called[-1] for c in player.hand)
        and any(c in TRUMP or c[-1] != called[-1] for c in player.hand)
    )
    return True, cs


def schedule_p(
    is_lead: bool,
    called_suit_eligible: bool,
    *,
    p_base: float,
    boost_lead: float,
    boost_cs: float,
    p_min: float,
    p_max: float,
) -> float:
    """§17.2: p = clip(p0 * b_lead^[lead] * b_cs^[cs], p_min, p_max). The
    nonzero floor keeps every eligible class covered (annealed-bias
    rationale of prioritized sampling — Schaul et al. 2016)."""
    p = p_base
    if is_lead:
        p *= boost_lead
    if called_suit_eligible:
        p *= boost_cs
    return min(max(p, p_min), p_max)


def node_class(game, player, head: str) -> str:
    """Manifest/telemetry class label: bidding head name, or
    ``{regime}|{play_cell}`` for play decisions."""
    if head != "play":
        return head
    from sheepshead.training.pfsp_runtime import play_cell

    regime = "leaster" if game.is_leaster else ("alone" if game.alone_called else "std")
    return f"{regime}|{play_cell(game, player)}"


def _replicate_top_pair_diffs(replicates, valid_actions) -> tuple[list, list]:
    """(top2_actions, per-replicate Q diffs for that pair) — the paired-
    replicate statistic behind the §17.6 noise-floor calibration (the
    §12.8 instrument, re-emitted per corpus node)."""
    acts = sorted(valid_actions)
    usable = [r for r in replicates if r["ok"] and r.get("root_q") is not None]
    pooled = {}
    for a in acts:
        obs = [float(r["root_q"][a]) for r in usable if r["root_n"].get(a, 0.0) > 0.0]
        if obs:
            pooled[a] = float(np.mean(obs))
    if len(pooled) < 2:
        return [], []
    a1, a2 = sorted(pooled, key=pooled.get, reverse=True)[:2]
    diffs = [
        float(r["root_q"][a1]) - float(r["root_q"][a2])
        for r in usable
        if r["root_n"].get(a1, 0.0) > 0.0 and r["root_n"].get(a2, 0.0) > 0.0
    ]
    return [a1, a2], diffs


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #
def worker_init(init_args: dict) -> None:
    import torch

    torch.set_num_threads(int(init_args.get("torch_threads", 1)))
    routed = init_args.get("routed_encoder")
    if routed:
        # Throughput-only (§16.6: 1.50x on committee search); the shadow
        # never needs re-syncing here because the weights are frozen.
        from sheepshead.agent.compiled_encoder import enable_routed_encoder

        enable_routed_encoder(32, None, device=routed)

    from sheepshead.agent.ppo import load_agent
    from sheepshead.ismcts import ISMCTSConfig, ISMCTSTeacher

    agent = load_agent(init_args["ckpt"])
    agent.stash_action_probs = True
    # Terminal-reward regime: undiscounted search backups (matches the
    # league teacher's teacher_gamma=1.0).
    agent.gamma = 1.0
    iters = int(init_args["iters"])
    _W.clear()
    _W.update(
        {
            "agent": agent,
            "teacher": ISMCTSTeacher(
                agent,
                ISMCTSConfig(
                    iters={h: iters for h in ("pick", "partner", "bury", "play")}
                ),
            ),
            "args": init_args,
        }
    )


def _search_node(game, player, valid_actions, forced_public, det_rng, anchor):
    """Run the committee and build the CE target (base_prior = the act-time
    stash, the §16.6 zero-gradient abstention referent). Returns
    (target_list | None, info | None, telemetry_extras)."""
    from sheepshead.training.pfsp_runtime import build_ce_search_target

    init_args = _W["args"]
    rngs = [
        random.Random(det_rng.getrandbits(64))
        for _ in range(int(init_args["replicates"]))
    ]
    replicates = _W["teacher"].search_committee(
        game,
        player.position,
        list(forced_public),
        rngs,
        d_rollout=int(init_args["d_rollout"]),
    )
    built = build_ce_search_target(
        replicates,
        valid_actions,
        shrink_nu=float(init_args["shrink_nu"]),
        shrink_s2_global=float(init_args["shrink_s2_global"]),
        gumbel_c_visit=_W["teacher"].config.gumbel_c_visit,
        gumbel_c_scale=_W["teacher"].config.gumbel_c_scale,
        base_prior=anchor,
    )
    top_pair, pair_diffs = _replicate_top_pair_diffs(replicates, valid_actions)
    if built is None:
        return None, None, (top_pair, pair_diffs)
    target, info = built
    return [float(x) for x in target], info, (top_pair, pair_diffs)


def play_corpus_game(task: tuple) -> dict:
    """One self-play game, all five seats collected. Returns per-seat
    episode event lists (store_episode_events schema + distill keys),
    node-telemetry rows and per-class counters."""
    import torch

    from sheepshead.ismcts import infer_head, is_private_action
    from sheepshead.training.training_utils import (
        compute_any_unseen_trump_higher_than_hand,
        compute_known_points_rel,
        compute_seen_trump_mask,
    )

    game_idx, mode, committee_act = task
    init_args = _W["args"]
    agent = _W["agent"]
    base_seed = int(init_args["seed"])
    # Per-game deterministic streams, independent of pool scheduling.
    game_rng = random.Random((base_seed << 20) ^ game_idx)
    torch.manual_seed((base_seed ^ (game_idx * 0x9E3779B1)) & 0x7FFFFFFF)
    det_rng = random.Random(game_rng.getrandbits(64))
    collect_oracle = bool(init_args["collect_oracle"])

    game = Game(partner_selection_mode=mode, seed=game_rng.randint(0, 2**31 - 1))
    agent.reset_recurrent_state()

    seat_transitions: dict[int, list] = {pos: [] for pos in range(1, 6)}
    forced_public: list[tuple[int, int]] = []
    telemetry_rows: list[dict] = []
    counts: dict[str, dict] = defaultdict(
        lambda: {"nodes": 0, "searched": 0, "override": 0, "endorsed": 0, "failed": 0}
    )
    gaps: list[float] = []
    committee_acted = 0

    while not game.is_done():
        for player in game.players:
            valid_actions = player.get_valid_action_ids()
            while valid_actions:
                state = player.get_state_dict()
                action, log_prob, value = agent.act(
                    state, valid_actions, player.position
                )
                anchor = None
                probs = agent.last_action_probs
                if probs is not None:
                    acts = sorted(valid_actions)
                    a = np.clip(
                        np.array([float(probs[i - 1]) for i in acts], dtype=np.float64),
                        1e-12,
                        None,
                    )
                    anchor = a / a.sum()

                head = infer_head(valid_actions)
                cls = node_class(game, player, head)
                counts[cls]["nodes"] += 1
                dset = "none"
                target_list = None
                info = None
                if len(valid_actions) >= 2:
                    searchable_play = head == "play" and not game.is_leaster
                    if not searchable_play:
                        dset = "retention"
                    else:
                        if game.alone_called:
                            is_lead, cs_elig = (
                                all(c == "" for c in game.history[game.current_trick]),
                                False,
                            )
                        else:
                            is_lead, cs_elig = lead_features(game, player)
                        p = schedule_p(
                            is_lead,
                            cs_elig,
                            p_base=float(init_args["p_base"]),
                            boost_lead=float(init_args["boost_lead"]),
                            boost_cs=float(init_args["boost_cs"]),
                            p_min=float(init_args["p_min"]),
                            p_max=float(init_args["p_max"]),
                        )
                        # Calibration mode (§17.6): spend search on alone
                        # nodes only — the game is discarded otherwise, and
                        # the unsearched standard rows stay "none" (the
                        # sampling gate, not the regime gate, skips them).
                        if init_args.get("alone_only") and not game.alone_called:
                            p = 0.0
                        if det_rng.random() < p:
                            counts[cls]["searched"] += 1
                            target_list, info, (top_pair, pair_diffs) = _search_node(
                                game,
                                player,
                                valid_actions,
                                forced_public,
                                det_rng,
                                anchor,
                            )
                            if target_list is None:
                                dset = "none"
                                counts[cls]["failed"] += 1
                            elif info["w"] > 0.0:
                                dset = "override"
                                counts[cls]["override"] += 1
                                gaps.append(info["gap"])
                            else:
                                dset = "endorsed"
                                counts[cls]["endorsed"] += 1
                            telemetry_rows.append(
                                {
                                    "game": game_idx,
                                    "class": cls,
                                    "n_valid": len(valid_actions),
                                    "w": info["w"] if info else None,
                                    "gap": info["gap"] if info else None,
                                    "spread": info["spread"] if info else None,
                                    "top_pair": top_pair,
                                    "pair_diffs": pair_diffs,
                                }
                            )
                            if (
                                committee_act
                                and dset == "override"
                                and target_list is not None
                            ):
                                acts = sorted(valid_actions)
                                acted = acts[int(np.argmax(target_list))]
                                if acted != action:
                                    committee_acted += 1
                                    action = acted
                                    log_prob = float(np.log(anchor[acts.index(acted)]))

                transition = {
                    "kind": "action",
                    "state": state,
                    "action": action,
                    "log_prob": float(log_prob),
                    "value": float(value),
                    "valid_actions": set(valid_actions),
                    "player_id": player.position,
                    "secret_partner_label": 1.0 if player.is_secret_partner else 0.0,
                    "points_label": compute_known_points_rel(player),
                    "seen_trump_mask_label": compute_seen_trump_mask(player),
                    "unseen_trump_higher_than_hand_label": (
                        compute_any_unseen_trump_higher_than_hand(player)
                    ),
                    # Distill annotations (§17.3)
                    "distill_set": dset,
                    "node_class": cls,
                    "has_search_target": dset == "override",
                    "search_target": target_list if dset == "override" else None,
                    "search_gap": float(info["gap"]) if info else 0.0,
                    "search_w": float(info["w"]) if info else 0.0,
                    "anchor_probs": (
                        [float(x) for x in anchor]
                        if dset in ("endorsed", "retention") and anchor is not None
                        else None
                    ),
                }
                if collect_oracle:
                    transition["oracle_state"] = player.get_oracle_state_dict()
                seat_transitions[player.position].append(transition)

                if not is_private_action(action):
                    forced_public.append((player.position, action))
                player.act(action)

                if game.was_trick_just_completed and not game.is_done():
                    for seat in game.players:
                        obs_state = seat.get_last_trick_state_dict()
                        agent.observe(obs_state, player_id=seat.position)
                        obs = {"kind": "observation", "state": obs_state}
                        if collect_oracle:
                            obs["oracle_state"] = (
                                seat.get_last_trick_oracle_state_dict()
                            )
                        seat_transitions[seat.position].append(obs)

                valid_actions = player.get_valid_action_ids()

    final_scores = [p.get_score() for p in game.players]
    episodes = []
    for pos in range(1, 6):
        events = seat_transitions[pos]
        score = float(final_scores[pos - 1])
        last_action = max(
            (i for i, e in enumerate(events) if e["kind"] == "action"), default=None
        )
        for i, ev in enumerate(events):
            if ev["kind"] == "action":
                # Terminal-only reward on the seat's last action (the
                # trainer's MC value target reads final_return directly).
                ev["reward"] = score / RETURN_SCALE if i == last_action else 0.0
                ev["win_label"] = 1.0 if score > 0 else 0.0
                ev["final_return_label"] = score
        episodes.append(events)

    return {
        "game": game_idx,
        "mode": "called" if mode == PARTNER_BY_CALLED_ACE else "jd",
        "committee_act": committee_act,
        "committee_acted": committee_acted,
        "is_leaster": bool(game.is_leaster),
        "alone_called": bool(game.alone_called),
        "episodes": episodes,
        "telemetry": telemetry_rows,
        "counts": dict(counts),
        "gaps": gaps,
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _git_rev() -> str:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=os.path.dirname(os.path.abspath(__file__)),
            ).stdout.strip()
            or "unknown"
        )
    except OSError:
        return "unknown"


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def main() -> int:
    from sheepshead.training.config import SearchConfig

    sc = SearchConfig()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True, help="frozen theta_k checkpoint")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--games", type=int, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--torch-threads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shard-games", type=int, default=200)
    ap.add_argument(
        "--committee-act-frac",
        type=float,
        default=0.25,
        help="fraction of games where material searches ACT (§17.3)",
    )
    ap.add_argument(
        "--alone-only",
        action="store_true",
        help="calibration mode: keep only games where ALONE was called "
        "(size --games so the surviving fraction meets the target)",
    )
    ap.add_argument("--no-oracle", dest="collect_oracle", action="store_false")
    ap.add_argument("--iters", type=int, default=sc.teacher_iters)
    ap.add_argument("--replicates", type=int, default=sc.teacher_replicates)
    ap.add_argument("--d-rollout", type=int, default=sc.teacher_d_rollout)
    ap.add_argument("--shrink-nu", type=float, default=sc.shrink_nu)
    ap.add_argument("--shrink-s2-global", type=float, default=sc.shrink_s2_global)
    # §17.2 schedule
    ap.add_argument("--p-base", type=float, default=0.10)
    ap.add_argument("--boost-lead", type=float, default=1.25)
    ap.add_argument("--boost-cs", type=float, default=1.5)
    ap.add_argument("--p-min", type=float, default=0.05)
    ap.add_argument("--p-max", type=float, default=0.25)
    ap.add_argument("--node-telemetry", default=None)
    ap.add_argument(
        "--routed-encoder",
        nargs="?",
        const="mps",
        default=None,
        help="route >=16-row encodes to a compiled shadow (§16.6 throughput)",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    init_args = {
        "ckpt": args.ckpt,
        "seed": args.seed,
        "torch_threads": args.torch_threads,
        "collect_oracle": args.collect_oracle,
        "alone_only": args.alone_only,
        "iters": args.iters,
        "replicates": args.replicates,
        "d_rollout": args.d_rollout,
        "shrink_nu": args.shrink_nu,
        "shrink_s2_global": args.shrink_s2_global,
        "p_base": args.p_base,
        "boost_lead": args.boost_lead,
        "boost_cs": args.boost_cs,
        "p_min": args.p_min,
        "p_max": args.p_max,
        "routed_encoder": args.routed_encoder,
    }

    # Deterministic task schedule: modes alternate; committee-act games are
    # drawn by index hash at the pre-registered fraction.
    sched_rng = random.Random(args.seed ^ 0xD157)
    tasks = []
    for g in range(args.games):
        mode = PARTNER_BY_CALLED_ACE if g % 2 == 0 else PARTNER_BY_JD
        tasks.append((g, mode, sched_rng.random() < args.committee_act_frac))

    import torch

    telemetry_f = open(args.node_telemetry, "w") if args.node_telemetry else None
    manifest = {
        "ckpt": args.ckpt,
        "ckpt_sha256_16": _file_sha256(args.ckpt),
        "git_rev": _git_rev(),
        "config": {k: v for k, v in vars(args).items() if k != "out_dir"},
        "games": 0,
        "kept_games": 0,
        "episodes": 0,
        "committee_act_games": 0,
        "committee_acted_nodes": 0,
        "classes": {},
        "gap_percentiles": None,
        "shards": [],
    }
    all_gaps: list[float] = []
    class_totals: dict[str, dict] = defaultdict(
        lambda: {"nodes": 0, "searched": 0, "override": 0, "endorsed": 0, "failed": 0}
    )
    shard_episodes: list = []
    shard_meta: list = []
    shard_idx = 0
    kept = 0
    done = 0
    t_start = time.time()

    def flush_shard():
        nonlocal shard_idx, shard_episodes, shard_meta
        if not shard_episodes:
            return
        path = os.path.join(args.out_dir, f"corpus_{shard_idx:04d}.pt")
        torch.save({"episodes": shard_episodes, "games": shard_meta}, path + ".tmp")
        os.replace(path + ".tmp", path)
        manifest["shards"].append(
            {"path": os.path.basename(path), "episodes": len(shard_episodes)}
        )
        shard_idx += 1
        shard_episodes = []
        shard_meta = []

    def write_manifest():
        for cls, c in class_totals.items():
            manifest["classes"][cls] = dict(c)
        if all_gaps:
            qs = np.percentile(all_gaps, [10, 25, 50, 75, 90, 99])
            manifest["gap_percentiles"] = {
                p: float(v) for p, v in zip([10, 25, 50, 75, 90, 99], qs)
            }
        with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

    ctx = get_context("spawn")
    with ctx.Pool(
        processes=args.workers, initializer=worker_init, initargs=(init_args,)
    ) as pool:
        for res in pool.imap_unordered(play_corpus_game, tasks, chunksize=1):
            done += 1
            manifest["games"] += 1
            if args.alone_only and not res["alone_called"]:
                if done % 50 == 0:
                    print(f"[{done}/{len(tasks)}] alone-only: kept {kept}", flush=True)
                continue
            kept += 1
            manifest["kept_games"] = kept
            manifest["episodes"] += len(res["episodes"])
            if res["committee_act"]:
                manifest["committee_act_games"] += 1
            manifest["committee_acted_nodes"] += res["committee_acted"]
            for cls, c in res["counts"].items():
                tot = class_totals[cls]
                for k in tot:
                    tot[k] += c[k]
            all_gaps.extend(res["gaps"])
            if telemetry_f is not None:
                for row in res["telemetry"]:
                    telemetry_f.write(json.dumps(row) + "\n")
                telemetry_f.flush()
            shard_episodes.extend(res["episodes"])
            shard_meta.append(
                {
                    "game": res["game"],
                    "mode": res["mode"],
                    "committee_act": res["committee_act"],
                    "is_leaster": res["is_leaster"],
                    "alone_called": res["alone_called"],
                }
            )
            if len(shard_meta) >= args.shard_games:
                flush_shard()
                write_manifest()
            if done % 25 == 0 or done == len(tasks):
                searched = sum(c["searched"] for c in class_totals.values())
                override = sum(c["override"] for c in class_totals.values())
                rate = done / max(time.time() - t_start, 1e-9)
                print(
                    f"[{done}/{len(tasks)} games, {rate:.2f} g/s] "
                    f"searched {searched} override {override} "
                    f"endorsed {sum(c['endorsed'] for c in class_totals.values())} "
                    f"failed {sum(c['failed'] for c in class_totals.values())}",
                    flush=True,
                )
    flush_shard()
    write_manifest()
    if telemetry_f is not None:
        telemetry_f.close()
    print(
        f"DONE: {kept} games kept, {manifest['episodes']} episodes, "
        f"{shard_idx} shards -> {args.out_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
