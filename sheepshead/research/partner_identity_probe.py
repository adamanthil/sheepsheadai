"""Does each recall-ladder rung know it is the secret partner?
(Architecture_Ablation_202610 §4.4)

The observation never says "you are the partner": ``partner_rel`` stays 0
until the partner is revealed, so a secret partner must infer its role by
matching ``called_card_id`` against ``hand_ids`` (called-ace mode) or by
holding JD (JD mode). The informed card init gives same-suit fail cards and
adjacent trumps near-identical embeddings, which would make that exact
identity match hard to read without an aux head forcing it. Three checks,
per (rung, seed) at one checkpoint:

1. **Embedding geometry**: for each partner card (AC, AS, AH, JD), the
   cosine similarity of its card embedding to the nearest other card, at
   initialization (a fresh agent built under the cell's seed) and trained.
2. **Linear probe**: logistic regression for ``is_secret_partner`` on the
   encoder output (every 2-D tensor it returns, concatenated) at the play
   nodes of non-pickers while the partner is unrevealed; held out by game.
3. **Trump leads by role**: greedy lead rates at lead nodes of non-pickers
   holding trump and fail while the partner is unrevealed, split by secret
   partner vs true defender, partner mode and trick.

Greedy self-play with the agent in every seat, as the convention battery
(``eval.health_probe``), but on seeded deals shared by every cell.

    uv run python -m sheepshead.research.partner_identity_probe --jobs 8
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from sheepshead.training.run_recall_ablation import (
    EPISODES,
    LADDER,
    SEEDS,
    checkpoint,
    eval_path,
)

PARTNER_CARDS = ("AC", "AS", "AH", "JD")
N_GAMES = 3000
DEAL_SEED = 20261102
TRAIN_FRACTION = 0.7
OUT_CSV = Path("notebooks/arch_ablation_202610_partner_identity.csv")


def _cosine_rows(weight: np.ndarray) -> dict[str, tuple[float, str]]:
    """Nearest other real card (ids 1..32) to each partner card, by cosine."""
    from sheepshead.game import DECK, DECK_IDS

    w = weight / np.maximum(np.linalg.norm(weight, axis=1, keepdims=True), 1e-12)
    out = {}
    for card in PARTNER_CARDS:
        i = DECK_IDS[card]
        sims = [(float(w[i] @ w[DECK_IDS[c]]), c) for c in DECK if c != card]
        out[card] = max(sims)
    return out


def embedding_geometry(arch: str, seed: int, agent) -> dict:
    """Nearest-neighbour cosine of the partner cards, at init and trained."""
    import torch

    from sheepshead.agent.ppo import PPOAgent
    from sheepshead.game import ACTIONS

    card = getattr(agent.encoder, "card", None)
    if card is None:  # the one-hot rung has no card embedding
        return {}
    torch.manual_seed(seed)
    fresh = PPOAgent(len(ACTIONS), arch=arch)
    init = _cosine_rows(fresh.encoder.card.weight.detach().numpy())
    trained = _cosine_rows(card.weight.detach().numpy())
    row = {}
    for c in PARTNER_CARDS:
        row[f"init_nn_cos_{c}"], row[f"init_nn_{c}"] = init[c]
        row[f"trained_nn_cos_{c}"], row[f"trained_nn_{c}"] = trained[c]
    row["init_nn_cos_mean"] = float(np.mean([init[c][0] for c in PARTNER_CARDS]))
    row["trained_nn_cos_mean"] = float(np.mean([trained[c][0] for c in PARTNER_CARDS]))
    return row


def _features(encoder_out) -> np.ndarray:
    parts = [
        v[0].detach().numpy()
        for _, v in sorted(encoder_out.items())
        if hasattr(v, "dim") and v.dim() == 2
    ]
    return np.concatenate(parts).astype(np.float32)


def play_and_collect(agent, n_games: int) -> dict:
    """Greedy self-play; probe rows and lead tallies while the partner is
    unrevealed (non-picker seats, non-leaster, partner not called alone)."""
    import torch

    from sheepshead.agent.observation import (
        last_trick_observation_for,
        observation_for,
    )
    from sheepshead.game import (
        ACTIONS,
        PARTNER_BY_JD,
        TRUMP,
        Game,
        get_partner_selection_mode,
    )

    X, y, game_idx, modes = [], [], [], []
    # leads[(role, mode, trick)] = [trump leads, eligible leads]
    leads: dict[tuple[str, str, int], list[int]] = {}
    for g in range(n_games):
        game = Game(
            partner_selection_mode=get_partner_selection_mode(g), seed=DEAL_SEED + g
        )
        agent.reset_recurrent_state()
        mode = "jd" if game.partner_mode_flag == PARTNER_BY_JD else "called"
        while not game.is_done():
            for player in game.players:
                valid = player.get_valid_action_ids()
                while valid:
                    state = observation_for(player, agent)
                    is_play = game.play_started and all(
                        ACTIONS[x - 1].startswith("PLAY ") for x in valid
                    )
                    secret_phase = (
                        is_play
                        and not game.is_leaster
                        and not game.alone_called
                        and not game.partner
                        and not player.is_picker
                    )
                    if is_play:
                        with torch.no_grad():
                            probs, _, enc = (
                                agent.get_action_probs_logits_and_encoder_out(
                                    state, valid, player_id=player.position
                                )
                            )
                        a = int(torch.argmax(probs, dim=1).item()) + 1
                        if secret_phase:
                            X.append(_features(enc))
                            y.append(int(player.is_secret_partner))
                            game_idx.append(g)
                            modes.append(mode)
                    else:
                        a, _, _ = agent.act(
                            state, valid, player.position, deterministic=True
                        )
                    if (
                        secret_phase
                        and game.cards_played == 0
                        and game.leader == player.position
                        and any(c in TRUMP for c in player.hand)
                        and any(c not in TRUMP for c in player.hand)
                    ):
                        role = "partner" if player.is_secret_partner else "defender"
                        key = (role, mode, int(game.current_trick))
                        tally = leads.setdefault(key, [0, 0])
                        tally[0] += int(ACTIONS[a - 1][5:] in TRUMP)
                        tally[1] += 1
                    player.act(a)
                    valid = player.get_valid_action_ids()
                    if game.was_trick_just_completed:
                        for seat in game.players:
                            agent.observe(
                                last_trick_observation_for(seat, agent),
                                player_id=seat.position,
                            )
    return {
        "X": np.stack(X),
        "y": np.array(y),
        "game": np.array(game_idx),
        "mode": np.array(modes),
        "leads": leads,
    }


def linear_probe(data: dict) -> dict:
    """Held-out-by-game logistic regression for is_secret_partner."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y, games, modes = data["X"], data["y"], data["game"], data["mode"]
    train = games < int(TRAIN_FRACTION * N_GAMES)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
    model.fit(X[train], y[train])
    test = ~train
    p = model.predict_proba(X[test])[:, 1]
    row = {
        "probe_n_train": int(train.sum()),
        "probe_n_test": int(test.sum()),
        "probe_partner_frac": float(y[test].mean()),
        "probe_auc": float(roc_auc_score(y[test], p)),
        "probe_bal_acc": float(balanced_accuracy_score(y[test], p > 0.5)),
    }
    for mode in ("called", "jd"):
        m = modes[test] == mode
        row[f"probe_auc_{mode}"] = float(roc_auc_score(y[test][m], p[m]))
    return row


def lead_rates(leads: dict) -> dict:
    """Trump-lead % by role, overall, by mode, and trick 0 vs tricks 1+."""
    row = {}
    for role in ("partner", "defender"):
        groups = {
            "all": lambda k: True,
            "called": lambda k: k[1] == "called",
            "jd": lambda k: k[1] == "jd",
            "t0": lambda k: k[2] == 0,
            "t1plus": lambda k: k[2] >= 1,
        }
        for name, keep in groups.items():
            t = n = 0
            for key, (trump, total) in leads.items():
                if key[0] == role and keep(key):
                    t += trump
                    n += total
            row[f"{role}_trump_lead_{name}"] = 100.0 * t / n if n else None
            row[f"{role}_leads_{name}"] = n
    return row


def run_cell(arch: str, seed: int, stage: int) -> dict:
    import torch

    from sheepshead.agent.ppo import load_agent

    torch.set_num_threads(1)
    t0 = time.time()
    agent = load_agent(str(checkpoint(arch, seed, stage)))
    agent.encoder.eval()
    data = play_and_collect(agent, N_GAMES)
    row: dict = {"rung": LADDER.index(arch) + 1, "arch": arch, "seed": seed}
    row["episode"] = stage
    row.update(embedding_geometry(arch, seed, agent))
    row.update(linear_probe(data))
    row.update(lead_rates(data["leads"]))
    row["minutes"] = (time.time() - t0) / 60
    out = eval_path(arch, seed, f"partner_identity_{stage}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(row, indent=2))
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--episode", type=int, default=EPISODES)
    args = ap.parse_args(argv)
    cells = [(arch, seed) for seed in SEEDS for arch in LADDER]
    rows = []
    with ProcessPoolExecutor(
        max_workers=args.jobs, mp_context=get_context("spawn")
    ) as pool:
        futures = {
            pool.submit(run_cell, arch, seed, args.episode): (arch, seed)
            for arch, seed in cells
        }
        for fut in as_completed(futures):
            row = fut.result()
            rows.append(row)
            print(
                f"{row['arch']} s{row['seed']}: probe AUC {row['probe_auc']:.3f}, "
                f"partner trump {row['partner_trump_lead_all']}, "
                f"defender trump {row['defender_trump_lead_all']} "
                f"({row['minutes']:.1f} min)",
                flush=True,
            )
    rows.sort(key=lambda r: (r["seed"], r["rung"]))
    fields = list(dict.fromkeys(k for r in rows for k in r))
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: "" if r.get(k) is None else r.get(k) for k in fields})
    print(f"wrote {OUT_CSV}: {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
