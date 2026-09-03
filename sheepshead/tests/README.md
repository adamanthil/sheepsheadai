# Training test suite map

Run everything with `uv run pytest sheepshead/tests` (from the main
checkout; inside a worktree run
`PYTHONPATH=. <main>/.venv/bin/python -m pytest sheepshead/tests` so the
shared editable install is not re-pointed). The fast loop is
`-m "not slow"`; slow-marked suites play real episodes, run real
optimizer updates, or spawn worker pools.

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

  The probe states are observed through
  `sheepshead.agent.observation.observation_for`, so legacy architectures
  receive the picker memory exactly as when they were captured.

## Shared helpers (non-test modules)

| Module | Provides |
| --- | --- |
| `ppo_test_helpers.py` | `seed_all`, seeded self-play `play_episodes`, `prepare_minibatch_inputs` (the `update()` preprocessing mirror) |
| `game_test_utils.py` | `make_game` from fixed hands, `act` / `run_script` by action name, `sole_actor` |
| `distill_test_helpers.py` | `fresh_agent` (recall arch), `ScriptedCommittee` (deterministic ISMCTS stand-in), `worker_state` / `generate_game` for the corpus generator |
| `ismcts_test_helpers.py` | default `fresh_agent`, `drive_to_second_bury` deal driver |

Test modules never import from other test modules; anything two suites
share lives here, under a public name.

## File map

| File | Owns |
| --- | --- |
| `test_game_rules.py` / `test_game_scenarios.py` / `test_game_invariants.py` | Game engine rules (incl. the clean observation, the picker-memory and oracle interfaces), dealt scenarios, cross-deal invariants |
| `test_leaster_scoring.py` | Leaster scoring |
| `test_architectures.py` / `test_bilinear_pointer.py` | Registry architectures: shapes, forward smoke, param groups; the bilinear play pointer |
| `test_recall_architecture.py` | `perceiver-recall` and the observation contract: recall keys, `observation_for` merge rule, loud failures |
| `test_arch_golden.py` | Registry consistency (portable) + bit-exact goldens (machine-local) + legacy checkpoint shim |
| `test_search_encode_path.py` | Encoder split and the opt-in compiled encoder |
| `test_oracle_critic.py` / `test_oracle_aux_heads.py` / `test_oracle_init_guard.py` | Privileged-critic mode: oracle encoding, dual GAE, aux heads, deal-seeded games, `--oracle-init` guard |
| `test_pretrain_oracle.py` | Oracle pretraining: spawn-pool dataset + supervised fit (slow) |
| `test_ppo_event_storage.py` / `test_store_events_by_seat.py` | Raw-event → record mapping; per-seat league storage |
| `test_ppo_loss_math.py` | `_gae_1d` / `compute_gae` / `_actor_critic_losses` math on hand-derived values |
| `test_ppo_minibatch_equivalence.py` / `test_grad_accum.py` | Minibatch builders vs naive reference loops; gradient-accumulation update path |
| `test_training_reward_math.py` | `reward_shaping` pure functions (trick rewards, aux labels) |
| `test_trainer_output_contracts.py` | Trainer CSV headers, checkpoint payload keys, filename patterns, per-phase CLI defaults |
| `test_league.py` / `test_league_smoke.py` / `test_leaster_watchdog.py` | League roster/PFSP/ratings; `train_ppo.run_phase` end to end in all three phases (slow) incl. the worker pool and the controller sidecar; leaster watchdog |
| `test_parallel_stream_window.py` / `test_publish_weights.py` / `test_worker_inference_options.py` | Episode-stream generators, worker weight publishing, worker device/compile options |
| `test_entropy_controller.py` / `test_entropy_telemetry.py` / `test_step_telemetry.py` / `test_csv_truncate.py` | Target-entropy controller, its telemetry, boundary telemetry, crash-resume CSV dedupe |
| `test_stop_rules.py` / `test_program.py` | The program's decision rules on recorded numbers; the orchestrator (command wiring, config round-trip, generation loop with scripted h2h, resume) |
| `test_ismcts_exit_regression.py` / `test_ismcts_replay_equivalence.py` / `test_ismcts_committee.py` / `test_ismcts_oracle_leaves.py` / `test_committee_summary.py` | ISMCTS teacher: determinizer, forced replay, committee search, oracle leaves, committee summary |
| `test_distill_pipeline.py` / `test_policy_iteration.py` | Search-corpus generation; the policy-iteration stages (fit, target, distill, cert) |
| `test_scripted_agent.py` / `test_convention_wrapper.py` | Scripted baseline + convention wrappers |
| `test_trump_lead_probe.py` / `test_called_suit_probe.py` | Analysis probe plumbing (shared geometry in `sheepshead/analysis/conventions.py`) |

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
  consistency tests parametrize automatically. A new architecture must
  also declare `legacy_picker_memory` honestly: `test_recall_architecture`
  pins what each kind is fed.
- **The observation dict** — `test_game_rules` pins the exact key set;
  anything a human at the table cannot see stays out of it (the picker
  memory and the oracle view are separate interfaces).
- **Trainer output files, checkpoint payloads, or per-phase defaults** —
  `test_trainer_output_contracts.py` pins the contract on purpose;
  changing it is an interface change for the orchestrator, not a nit.
- **A program decision rule** — `test_stop_rules.py` replays recorded
  h2h series; a rule change must be re-derived against those numbers.
- **Reward shaping** — `test_training_reward_math.py`.
