# ExIt Validation Protocol + Baseline Reference (NOT committed)

May 2026. The validation protocol "defined up front" so a from-scratch ExIt run
can be judged against the frozen PPO baseline. All numbers CPU, dev Mac.
Baseline model = `final_pfsp_swish_ppo.pt` (the 30M PPO agent, incl. the critic
fix `b53496b`).

## t_full critic-calibration probe (`t_full_probe.py`)

Question: at which trick is the value head trustworthy enough to BOOTSTRAP the
ISMCTS rollout instead of rolling to terminal? Method: freeze encoder, play 3000
games, target = terminal `get_score()/12` (what the ExIt rollout bootstraps
toward), train a fresh DEEP head (production value_trunk shape) on a held-out-by-
game split, R^2 by trick.

R^2 by trick (fresh deep head, terminal target):
| trick | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| all play decisions | 0.26 | 0.40 | 0.54 | 0.62 | 0.73 | 0.82 |
| lead decisions | 0.23 | 0.36 | 0.52 | 0.63 | 0.74 | 0.83 |
| **defender leads (leak subset)** | **0.04** | 0.10 | 0.33 | 0.52 | 0.58 | 0.61 |
| leaster play | -0.03 | 0.05 | 0.13 | 0.15 | 0.16 | 0.21 |

Conclusions (acted on):
- **`t_full=1`, `d_short=2` validated.** A search at decision trick `t` bootstraps
  at ~`t+d_short`, so the first bootstrapped trick (2) lands its bootstrap at
  trick 4 (R^2 0.73), and the trick-0 leak states (R^2 0.04!) are always rolled to
  terminal (`0 <= t_full`). The `d_short=2` buffer is what makes `t_full=1` safe —
  the critic is never bootstrapped in its blind zone. Kept the defaults; updated
  the config comment from "placeholder" to evidence-based.
- **Leasters never calibrate** (R^2 <= 0.21 at trick 5) -> leaster searches now
  forced to terminal rollout regardless of `t_full` (pfsp_runtime change).
- Defender-lead leak states confirm the original partial-obs hypothesis (blind at
  trick 0, resolve as cards are revealed).

Caveat: probe uses the PPO critic (shaped-trained) as a proxy for "a trained
critic's calibration profile." Absolute R^2 may shift for a terminal-trained ExIt
critic, but the monotone rise + leaster/leak structure are partial-observability
properties that should hold. Re-run on an ExIt checkpoint to confirm.

## PPO baseline (`exit_validation.py`, greedy, 250 games)

The reference the ExIt agent must match or beat (run
`exit_validation.py -m <exit>.pt -b final_pfsp_swish_ppo.pt` later for real h2h).

**Bidding health (held WITHOUT epsilon-floor controllers):**
- PICK rate (of pick decisions): **32.9%**
- ALONE rate (of partner decisions): **6.6%**
- Leaster rate (of games): **9.2%**
- Per-position PICK rate: P1=28% P2=25% P3=39% P4=46% P5=48% (rises with seat
  order — later seats pick more, as expected)

**Trick-0 defender trump-lead (the original leak), 105 qualifying nodes:**
- trump-lead rate (greedy): **4.8%**
- conditional trump prob mass: **0.3%**
- i.e. the baseline (post critic-fix) already rarely leads trump as a trick-0
  defender. ExIt should hold this at/below baseline.

**Head-to-head sanity (baseline vs baseline, rotating seat, 250 games):**
- +0.208 +/- 0.169 points/game -> consistent with ~0 within noise (harness is
  unbiased). For a real exit-vs-PPO comparison, the signal must clear this noise
  floor; use >=1000 games to tighten the SE.

## Protocol items tracked DURING training (not in this harness)
- Distillation diagnostics: `update_stats['distill']` -> teacher_kl, ESS-abort
  fraction (gauge via `res['ok']` rate), pg_masked_fraction, pi' entropy.
- **PG-mask vs additive-form A/B**: two training configs (the hard PG-mask vs
  `PPO_clip + w_distill*CE` on searched states); promote whichever holds the
  guarded metrics better. Plan §4.
- Periodic exit-vs-PPO cross-eval (run the h2h above on checkpoints).

## How to reproduce
- `t_full_probe.py --games 3000` (≈ a few min collect + head train).
- `exit_validation.py --games 250 --h2h-games 250` (≈ 90s; self-play ~150 ms/game,
  lean mixed-agent h2h similar). Self-play can't batch (single trajectory), so it
  is inherently slow per game on CPU — keep game counts modest. NOTE: do NOT route
  h2h through play_population_game (the full training game w/ profiling is ~2-3
  s/game -> 400 games ~20 min); exit_validation uses a bare mixed-agent loop.
