#!/usr/bin/env python3
"""The release-candidate program's configuration (Training_Program_Redesign
§6): one dataclass tree that is both what the orchestrator runs and the
pre-registration artifact it writes next to its state.

Every number here is a decision recorded in the notebook; the defaults ARE
the pre-registered run. ``ProgramConfig.smoke()`` scales everything to a
minutes-long end-to-end check of the same code paths.
"""

from __future__ import annotations

import json
from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass

from sheepshead.analysis.panels import PANEL_A, PANEL_B


@dataclass
class BootstrapConfig:
    """Phase 0: shaped self-play from scratch (§4.1)."""

    episodes: int = 400_000
    update_interval: int | None = None  # None = BootstrapHyperparams
    save_interval: int = 50_000
    greedy_eval_interval: int = 50_000
    greedy_eval_games: int = 200
    # Health gate (no strength bar, §3.4): the last greedy probe must show
    # the all-PASS attractor escaped.
    max_final_leaster_rate: float = 50.0


@dataclass
class OracleConfig:
    """Phase 1: supervised oracle pretraining (§4.2)."""

    episodes: int = 40_000
    gamma: float = 1.0
    max_epochs: int = 25
    patience: int = 3


@dataclass
class LeagueConfig:
    """Phase 2: terminal-only league PG with the marginal-value handoff
    rule (§4.3, §5.1-§5.3)."""

    generation_episodes: int = 1_000_000
    min_generations: int = 3
    max_generations: int = 8
    seed_copies: int = 4  # sample_table draws without replacement
    update_interval: int | None = None  # None = LeagueHyperparams
    save_interval: int = 50_000
    snapshot_interval: int = 50_000
    greedy_eval_interval: int = 50_000
    greedy_eval_games: int = 200
    entropy_play_floor: float = 0.28
    # Worker inference device for the league generations (throughput only;
    # bit-exact vs CPU, Distributed_Inference_202608: MPS + torch.compile
    # 1.36x at 8 workers). The bootstrap and bidding phases keep the
    # program-level setting (CPU by default).
    worker_device: str | None = "mps"
    worker_compile: str | None = "default"
    # Handoff rule: continue while the duplicate h2h gain over the previous
    # generation is >= h2h_min_gain with the CI lower bound above zero.
    h2h_deals: int = 2_000  # per mode
    h2h_min_gain: float = 0.02
    h2h_ci_z: float = 2.0
    # Absolute yardstick (recorded; the gen-2 review gate reads it).
    panel: list[str] = field(default_factory=lambda: list(PANEL_A))
    panel_deals: int = 3996
    # PANEL-B (tentative, 09-16): the strong-skill / cross-ecology yardstick,
    # recorded per generation and at the final; no gate reads it. Empty
    # list = not run.
    panel_b: list[str] = field(default_factory=lambda: list(PANEL_B))
    # Convention battery (guards, not triggers) and the B2 hard bounds.
    convention_probe_games: int = 1000
    convention_probe_seeds: int = 4
    partner_trump_lead_min: float = 50.0
    defender_t0_trump_lead_max: float = 10.0


@dataclass
class PolicyIterationConfig:
    """Phase 3: search-Q policy iteration to convergence (§4.4; the
    compounding recipe of CE_Teacher_Design §20.14, pinned 2026-09-12)."""

    games: int = 8_000
    p_base: float = 1.0
    boost_lead: float = 1.0
    boost_cs: float = 1.5
    p_min: float = 0.05
    p_max: float = 1.0
    committee_act_frac: float = 1.0
    iters: int = 256
    # Trick-indexed lead budget (§20.13 add. 29b): t0 leads 1024, t1 leads
    # 512, everything else at ``iters`` (~+22% search time over all-256).
    iters_schedule: str | None = "t0-lead:1024,t1-lead:512"
    replicates: int = 3
    routed_encoder: str | None = "mps"
    corpus_seed_base: int = 20260902
    # Distill schedule (§20.14 step 4): six trunk epochs at 3e-5, then
    # bilinear-only head epochs at 1e-3; retention KL x10.
    trunk_epochs: int = 6
    lr: float = 3e-5
    head_epochs_default: int = 4
    lambda_ret: float = 10.0
    max_iterations: int = 5
    stop_se_multiple: float = 2.0
    stop_flat_iterations: int = 2
    bidding_episodes: int = 200_000
    cert_games: int = 1000
    cert_seeds: int = 4
    cert_h2h_deals: int = 8000
    # Head-routed reads in the cert (§20.14 step 5): the play-only route is
    # the compounding statistic; the bidding route is the drift guard.
    routed_reads: bool = True
    # Start the phase from an external theta_0 (validation on an existing
    # lineage) instead of the run's league handoff; the bidding phase then
    # samples opponents from ``league_dir``.
    theta_0: str | None = None
    league_dir: str | None = None
    # With start_phase="policy_iteration": run the bidding phase on theta_0
    # first (it is a certified distill candidate), then iterate from the
    # adopted checkpoint.
    bidding_first: bool = True
    # Smoke-only knobs for the fit/distill stages (None = module defaults).
    fit_epochs: int | None = None
    head_epochs: int | None = None
    holdout_frac: float | None = None
    batch_rows: int | None = None
    buffer_episodes: int | None = None
    batch_segments: int | None = None
    probe_games: int | None = None


@dataclass
class FinalConfig:
    """Phase 4: final certification and audit (§4.5)."""

    # Objective 2 (§1): beat the current best and the production 30M on the
    # deployment instrument. theta_3 = the last artifact of the
    # perceiver-shared-v2 lineage (its bidding phase's release, CE_Teacher
    # §21.4). Missing files are skipped with a log line.
    references: dict[str, str] = field(
        default_factory=lambda: {
            "v2_release": "runs/rc_validate_v2/final/release.pt",
            "iter11_p1": "runs/policy_iteration_202609/iter11/distill_epoch7.pt",
            "prod_30m": "final_pfsp_swish_ppo.pt",
        }
    )
    # 8000 deals/mode (SE ~0.0025): the final bars (30M positive at 2 SE,
    # iter11 excluding -0.02) are underpowered at 2000.
    h2h_deals: int = 8000
    exploit_episodes: int = 50_000
    exploit_gate_deals: int = 3000


@dataclass
class GateConfig:
    """Review gates (§5.3): operator review, never an automatic kill."""

    gen2_panel_min: float = 0.06
    handoff_reference: str = (
        "runs/league_retention_pg/checkpoints/"
        "pfsp_perceiver-shared-v2_checkpoint_8000000.pt"
    )
    handoff_h2h_lower_min: float = -0.02


@dataclass
class ProgramConfig:
    run_name: str = "rc_202609"
    arch: str = "perceiver-recall"
    seed: int = 42
    num_workers: int = 8
    worker_device: str | None = None
    worker_compile: str | None = None
    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)
    oracle: OracleConfig = field(default_factory=OracleConfig)
    league: LeagueConfig = field(default_factory=LeagueConfig)
    policy_iteration: PolicyIterationConfig = field(
        default_factory=PolicyIterationConfig
    )
    final: FinalConfig = field(default_factory=FinalConfig)
    gates: GateConfig = field(default_factory=GateConfig)
    smoke: bool = False
    # "bootstrap" runs every phase; "policy_iteration" skips to phase 3 from
    # ``policy_iteration.theta_0`` (validation of the final phases on an
    # existing lineage).
    start_phase: str = "bootstrap"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "ProgramConfig":
        return _from_dict(cls, d)

    @classmethod
    def load(cls, path: str) -> "ProgramConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def smoke_config(cls, run_name: str) -> "ProgramConfig":
        """A minutes-long end-to-end run of every phase."""
        cfg = cls(run_name=run_name, num_workers=2, smoke=True)
        # No PPO update inside the 40-episode bootstrap: a from-scratch
        # policy falls into the all-PASS attractor within a few updates and
        # then plays only leasters, which leaves the search corpus with
        # nothing to search (leaster play is a retention row). The init
        # policy still picks, so the downstream phases see standard games;
        # the update path itself is covered by test_league_smoke.
        cfg.bootstrap = BootstrapConfig(
            episodes=40,
            update_interval=1_000_000,
            save_interval=20,
            greedy_eval_interval=20,
            greedy_eval_games=4,
            max_final_leaster_rate=101.0,  # a 40-episode run never escapes
        )
        cfg.oracle = OracleConfig(episodes=16, max_epochs=2, patience=1)
        cfg.league = LeagueConfig(
            generation_episodes=20,
            min_generations=2,
            max_generations=3,
            seed_copies=4,
            # No PPO updates in the smoke's league generations either (same
            # attractor reason as the bootstrap); the phase's update path is
            # exercised by test_league_smoke and by the bidding phase below.
            update_interval=1_000_000,
            save_interval=10,
            snapshot_interval=10,
            greedy_eval_interval=10,
            greedy_eval_games=2,
            h2h_deals=3,
            worker_device=None,
            worker_compile=None,
            panel=[],  # the run's own bootstrap seeds
            panel_b=[],
            panel_deals=6,
            convention_probe_games=3,
            convention_probe_seeds=1,
            partner_trump_lead_min=0.0,
            defender_t0_trump_lead_max=100.0,
        )
        cfg.policy_iteration = PolicyIterationConfig(
            games=12,
            p_base=1.0,
            boost_lead=1.0,
            p_max=1.0,
            iters=8,
            iters_schedule=None,
            replicates=2,
            trunk_epochs=1,
            lr=1e-4,
            routed_reads=False,
            routed_encoder=None,
            max_iterations=1,
            bidding_episodes=20,
            cert_games=3,
            cert_seeds=1,
            cert_h2h_deals=3,
            fit_epochs=2,
            head_epochs=1,
            holdout_frac=0.34,
            batch_rows=32,
            buffer_episodes=10,
            batch_segments=4,
            probe_games=0,
        )
        cfg.final = FinalConfig(
            references={}, h2h_deals=3, exploit_episodes=20, exploit_gate_deals=4
        )
        cfg.gates = GateConfig(
            gen2_panel_min=-10.0, handoff_reference="", handoff_h2h_lower_min=-10.0
        )
        return cfg


def _from_dict(cls, d: dict):
    """Rebuild a config tree from its JSON dict (nested dataclasses are
    recognized by their default factories)."""
    kwargs = {}
    for f in fields(cls):
        if f.name not in d:
            continue
        v = d[f.name]
        default = f.default_factory() if f.default_factory is not MISSING else None
        if isinstance(v, dict) and default is not None and is_dataclass(default):
            kwargs[f.name] = _from_dict(type(default), v)
        else:
            kwargs[f.name] = v
    return cls(**kwargs)
