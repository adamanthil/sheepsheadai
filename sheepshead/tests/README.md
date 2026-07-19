# Training test suite map

Run everything with `uv run pytest sheepshead/tests`; the fast loop is
`-m "not slow"`. Slow-marked suites play real episodes or run real
optimizer updates.

## Two tiers

- **Portable tests** (everything except the arch goldens) hold on every
  platform. Numerical tests either compare two in-process computations
  (equivalence suites) or assert hand-derived values with `pytest.approx`
  tolerances — never stored float hashes.
- **Machine-local bit-exact gates** (`test_arch_golden.py` numerical
  checks only): weight-init and forward-pass byte-identity against
  fixtures captured by `sheepshead/analysis/capture_arch_goldens.py`.
  These skip whenever torch or the platform differ from the fixture
  manifest. **A torch upgrade silences them without failing anything** —
  re-capture on the dev machine afterwards to restore the gate:

      uv run python -m sheepshead.analysis.capture_arch_goldens
      uv run python -m sheepshead.analysis.capture_arch_goldens --check

## File map

| File | Owns |
| --- | --- |
| `test_game_rules.py` / `test_game_scenarios.py` / `test_game_invariants.py` | Game engine rules, dealt scenarios, cross-deal invariants |
| `test_leaster_scoring.py` | Leaster scoring |
| `test_architectures.py` | Registry architectures: shapes, forward smoke, param groups |
| `test_arch_golden.py` | Registry consistency (portable) + bit-exact goldens (machine-local) + legacy checkpoint shim |
| `test_oracle_critic.py` | Privileged-critic mode: oracle encoding, dual GAE wiring |
| `test_ppo_event_storage.py` | `store_episode_events` raw-event → record mapping and defaults |
| `test_ppo_loss_math.py` | `_gae_1d` / `compute_gae` / `_actor_critic_losses` math on hand-derived values |
| `test_ppo_minibatch_equivalence.py` | `_build_minibatch_tensors` / `_flatten_action_steps` vs naive reference loops, exact equality |
| `test_training_reward_math.py` | `reward_shaping` pure functions (trick rewards, aux labels) |
| `test_trainer_output_contracts.py` | Trainer CSV headers, checkpoint payload keys, filename patterns |
| `test_exploiter_gate.py` | Exploiter gate decision + `gate_result.json` contract |
| `test_league.py` / `test_league_stopping.py` / `test_league_smoke.py` / `test_leaster_watchdog.py` | League runtime, stopping rules, end-to-end smoke |
| `test_ismcts_exit_regression.py` | ISMCTS teacher/search behavior |
| `test_scripted_agent.py` / `test_convention_wrapper.py` | Scripted baseline + convention wrappers |
| `test_trump_lead_probe.py` / `test_called_suit_probe.py` | Analysis probe plumbing |
| `ppo_test_helpers.py` / `game_test_utils.py` | Shared non-test machinery (seeded self-play episodes, update() preprocessing mirror) |

## When you change the algorithm, what do you update?

- **A loss term or GAE behavior** — re-derive the affected
  `test_ppo_loss_math` cases by hand; each test pins one term and carries
  its derivation in a comment. A failing hand-derived value after an
  intentional change means the new expected value must be worked out on
  paper, not copied from the test output.
- **A minibatch tensor column / new aux head** — the fan-out is:
  `MinibatchTensors` / `ForwardOutputs` / `FlattenedActionSteps` in
  `ppo.py`, the naive builders and `FLAT_FIELD_SOURCES` in
  `test_ppo_minibatch_equivalence.py`, `store_episode_events` and its
  tests in `test_ppo_event_storage.py`. The naive builders are the
  written spec of the layout — keep them obvious, not clever.
- **Event record fields or label defaults** — `test_ppo_event_storage.py`.
- **An architecture** — re-capture goldens (commands above); registry
  consistency tests parametrize automatically.
- **Trainer output files or checkpoint payloads** —
  `test_trainer_output_contracts.py` pins the contract on purpose;
  changing it is an interface change for the orchestrators, not a nit.
- **Reward shaping** — `test_training_reward_math.py`.
