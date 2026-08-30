"""Tests for the CE_Teacher_Design §3 boundary cert gate in
train_league_ppo.py: run_boundary_cert's pass/fail contract, the exact
ABSOLUTE bars it is wired to by default, and the SystemExit(4) halt that
main() raises at its call site when a cert fails.

run_boundary_cert itself never raises: it returns a result dict with a
"passed" bool and a "failures" list, and persists that dict to
boundary_cert_gen<g>.json. The SystemExit(4) lives in main()'s call site
(around line ~1953: `if not cert["passed"]: ... raise SystemExit(4)`), not
inside run_boundary_cert — confirmed by reading the function body (it has
no `raise` statements) and the caller loop in main(). These tests therefore
pin the return-value contract directly against run_boundary_cert, and pin
the SystemExit(4) itself with a full (but heavily stubbed) main() call so
the actual conditional at the call site is what's under test.
"""

import json
import sys
from types import SimpleNamespace
from typing import cast

import pytest

from sheepshead import ACTIONS
from sheepshead.agent.ppo import PPOAgent
from sheepshead.training import league_gates, train_league_ppo


class _StubAgent:
    """Stand-in for PPOAgent inside run_boundary_cert: only the
    snapshot/restore memory hooks are exercised (greedy_health_probe and
    paired_edge are monkeypatched out, so nothing else touches the agent)."""

    def snapshot_player_memories(self):
        return {}

    def restore_player_memories(self, snapshot):
        pass


def _stub_agent() -> PPOAgent:
    """The stub, typed as the agent run_boundary_cert declares."""
    return cast(PPOAgent, _StubAgent())


def _make_cert_args(**overrides):
    defaults = dict(
        cert_seeds=3,
        cert_games=1000,
        cert_partner_floor=93.5,
        cert_t0_ceiling=5.0,
        cert_h2h_deals=1000,
        cert_anchor_resolved="anchor.pt",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _patch_probe_and_h2h(
    monkeypatch,
    partner_trump_lead_rate,
    t0_trump_lead_rate,
    called_suit_lead_rate,
    h2h_edge,
    h2h_se,
    h2h_n_deals=1000,
):
    def fake_probe(agent, n_games, seed):
        return {
            "partner_trump_lead_rate": partner_trump_lead_rate,
            "t0_trump_lead_rate": t0_trump_lead_rate,
            "called_suit_lead_rate": called_suit_lead_rate,
        }

    def fake_paired_edge(challenger, incumbent, field, n_deals, seed, log_every=0):
        return {
            "edge": h2h_edge,
            "se": h2h_se,
            "win_frac": 0.5,
            "n_deals": h2h_n_deals,
            "deviating_frac": 0.0,
        }

    # run_boundary_cert lives in league_gates; its collaborators resolve there.
    monkeypatch.setattr(league_gates, "greedy_health_probe", fake_probe)
    monkeypatch.setattr(league_gates, "paired_edge", fake_paired_edge)
    monkeypatch.setattr(league_gates, "load_agent", lambda path: _StubAgent())


# ----------------------------------------------------------------------------
# Default bars: pin the exact ABSOLUTE thresholds the cert is wired to.
# ----------------------------------------------------------------------------
def test_cert_default_bars_pinned():
    ap = train_league_ppo.build_arg_parser()
    args = ap.parse_args(["--resume", "x.pt", "--league-dir", "ld"])
    assert args.cert_seeds == 3
    assert args.cert_games == 1000
    assert args.cert_partner_floor == pytest.approx(93.5)
    assert args.cert_t0_ceiling == pytest.approx(5.0)
    assert args.cert_h2h_deals == 1000
    assert args.cert_anchor_ckpt is None


# ----------------------------------------------------------------------------
# run_boundary_cert: pass branch
# ----------------------------------------------------------------------------
def test_run_boundary_cert_pass(tmp_path, monkeypatch):
    _patch_probe_and_h2h(
        monkeypatch,
        partner_trump_lead_rate=95.0,  # >= 93.5 floor
        t0_trump_lead_rate=2.0,  # <= 5.0 ceiling
        called_suit_lead_rate=50.0,
        h2h_edge=0.05,
        h2h_se=0.01,  # edge + 2*se = 0.07 >= 0
    )
    args = _make_cert_args()
    result = train_league_ppo.run_boundary_cert(
        _stub_agent(), args, generation=3, checkpoint_dir=str(tmp_path)
    )

    assert result["passed"] is True
    assert result["failures"] == []
    assert result["adherence"]["partner_trump_mean"] == pytest.approx(95.0)
    assert result["adherence"]["t0_trump_mean"] == pytest.approx(2.0)
    # Cert result is persisted for the run record: boundary_cert_gen<g>.json.
    cert_path = tmp_path / "boundary_cert_gen3.json"
    assert cert_path.exists()
    on_disk = json.loads(cert_path.read_text())
    assert on_disk == result


# ----------------------------------------------------------------------------
# run_boundary_cert: fail branches (one per cert component)
# ----------------------------------------------------------------------------
def test_run_boundary_cert_fails_partner_floor(tmp_path, monkeypatch):
    _patch_probe_and_h2h(
        monkeypatch,
        partner_trump_lead_rate=80.0,  # < 93.5 floor
        t0_trump_lead_rate=2.0,
        called_suit_lead_rate=50.0,
        h2h_edge=0.05,
        h2h_se=0.01,
    )
    args = _make_cert_args()
    result = train_league_ppo.run_boundary_cert(
        _stub_agent(), args, generation=1, checkpoint_dir=str(tmp_path)
    )

    assert result["passed"] is False
    assert len(result["failures"]) == 1
    assert "partner trump-lead mean 80.0%" in result["failures"][0]
    assert "cert floor 93.5%" in result["failures"][0]


def test_run_boundary_cert_fails_t0_ceiling(tmp_path, monkeypatch):
    _patch_probe_and_h2h(
        monkeypatch,
        partner_trump_lead_rate=95.0,
        t0_trump_lead_rate=10.0,  # > 5.0 ceiling
        called_suit_lead_rate=50.0,
        h2h_edge=0.05,
        h2h_se=0.01,
    )
    args = _make_cert_args()
    result = train_league_ppo.run_boundary_cert(
        _stub_agent(), args, generation=1, checkpoint_dir=str(tmp_path)
    )

    assert result["passed"] is False
    assert len(result["failures"]) == 1
    assert "t0 trump-lead mean 10.0%" in result["failures"][0]
    assert "cert ceiling 5.0%" in result["failures"][0]


def test_run_boundary_cert_fails_h2h_significantly_negative(tmp_path, monkeypatch):
    _patch_probe_and_h2h(
        monkeypatch,
        partner_trump_lead_rate=95.0,
        t0_trump_lead_rate=2.0,
        called_suit_lead_rate=50.0,
        h2h_edge=-0.5,
        h2h_se=0.01,  # edge + 2*se = -0.48 < 0
    )
    args = _make_cert_args()
    result = train_league_ppo.run_boundary_cert(
        _stub_agent(), args, generation=1, checkpoint_dir=str(tmp_path)
    )

    assert result["passed"] is False
    assert len(result["failures"]) == 1
    assert "significantly negative" in result["failures"][0]
    assert "-0.500" in result["failures"][0]


def test_run_boundary_cert_h2h_edge_exactly_zero_passes(tmp_path, monkeypatch):
    # Boundary condition: edge + 2*se == 0.0 is NOT a failure (the check is
    # strictly `< 0.0`).
    _patch_probe_and_h2h(
        monkeypatch,
        partner_trump_lead_rate=95.0,
        t0_trump_lead_rate=2.0,
        called_suit_lead_rate=50.0,
        h2h_edge=-0.02,
        h2h_se=0.01,  # edge + 2*se == 0.0 exactly
    )
    args = _make_cert_args()
    result = train_league_ppo.run_boundary_cert(
        _stub_agent(), args, generation=1, checkpoint_dir=str(tmp_path)
    )
    assert result["passed"] is True


def test_run_boundary_cert_uses_configured_seed_count(tmp_path, monkeypatch):
    seen_seeds = []

    def fake_probe(agent, n_games, seed):
        seen_seeds.append(seed)
        return {
            "partner_trump_lead_rate": 95.0,
            "t0_trump_lead_rate": 2.0,
            "called_suit_lead_rate": 50.0,
        }

    monkeypatch.setattr(league_gates, "greedy_health_probe", fake_probe)
    monkeypatch.setattr(
        league_gates,
        "paired_edge",
        lambda *a, **k: {"edge": 0.0, "se": 0.0, "win_frac": 0.5, "n_deals": 1000},
    )
    monkeypatch.setattr(league_gates, "load_agent", lambda path: _StubAgent())

    args = _make_cert_args(cert_seeds=5, cert_games=17)
    train_league_ppo.run_boundary_cert(
        _stub_agent(), args, generation=1, checkpoint_dir=str(tmp_path)
    )
    assert len(seen_seeds) == 5
    assert seen_seeds == [league_gates.ADHERENCE_GUARD_SEED + i for i in range(5)]


# ----------------------------------------------------------------------------
# main(): SystemExit(4) at the boundary-cert call site.
#
# run_boundary_cert never raises (see module docstring above) — the
# SystemExit(4) is raised by main()'s generation loop when
# `not cert["passed"]`. To pin the actual call-site conditional (rather than
# re-deriving it), this drives a real main() invocation with run_main_phase
# and run_boundary_cert monkeypatched, and a real (tiny) PPOAgent/League so
# the surrounding setup code executes unmodified up to the cert call.
# ----------------------------------------------------------------------------
def test_main_raises_systemexit_4_on_cert_fail(tmp_path, monkeypatch):
    resume_ckpt = tmp_path / "resume.pt"
    PPOAgent(len(ACTIONS)).save(str(resume_ckpt))

    league_dir = tmp_path / "league"
    argv = [
        "train_league_ppo.py",
        "--resume",
        str(resume_ckpt),
        "--league-dir",
        str(league_dir),
        "--run-name",
        "cert_exit_test",
        "--generations",
        "1",
        "--main-episodes",
        "10",
        "--teacher",
        "--num-workers",
        "0",
        "--arch",
        "full",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.chdir(tmp_path)

    # run_main_phase would otherwise train for real; stand it up as a no-op
    # that just returns the phase boundary episode count.
    monkeypatch.setattr(
        train_league_ppo,
        "run_main_phase",
        lambda agent, league, ratings, args, start_episode, n_episodes, ckpt_dir, anchor_eval=None: (
            start_episode + n_episodes
        ),
    )
    monkeypatch.setattr(
        train_league_ppo,
        "run_boundary_cert",
        lambda agent, args, generation, checkpoint_dir: {
            "generation": generation,
            "passed": False,
            "failures": ["stubbed cert failure"],
        },
    )

    with pytest.raises(SystemExit) as excinfo:
        train_league_ppo.main()

    assert excinfo.value.code == 4


def test_main_continues_past_cert_pass_to_exploiter_phase(tmp_path, monkeypatch):
    # Complement of the fail-branch test: on a passing cert, main() must NOT
    # raise, and must proceed to the exploiter-generation call (the next
    # step after refreeze) rather than halting.
    resume_ckpt = tmp_path / "resume.pt"
    PPOAgent(len(ACTIONS)).save(str(resume_ckpt))

    league_dir = tmp_path / "league"
    argv = [
        "train_league_ppo.py",
        "--resume",
        str(resume_ckpt),
        "--league-dir",
        str(league_dir),
        "--run-name",
        "cert_exit_test_pass",
        "--generations",
        "1",
        "--main-episodes",
        "10",
        "--teacher",
        "--num-workers",
        "0",
        "--arch",
        "full",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(
        train_league_ppo,
        "run_main_phase",
        lambda agent, league, ratings, args, start_episode, n_episodes, ckpt_dir, anchor_eval=None: (
            start_episode + n_episodes
        ),
    )
    monkeypatch.setattr(
        train_league_ppo,
        "run_boundary_cert",
        lambda agent, args, generation, checkpoint_dir: {
            "generation": generation,
            "passed": True,
            "failures": [],
        },
    )

    exploiter_calls = []

    def fake_exploiter_generation(args, generation, main_ckpt):
        exploiter_calls.append((generation, main_ckpt))
        raise SystemExit(0)  # short-circuit the rest of main() cheaply

    monkeypatch.setattr(
        train_league_ppo, "run_exploiter_generation", fake_exploiter_generation
    )

    with pytest.raises(SystemExit) as excinfo:
        train_league_ppo.main()

    # Reached the exploiter call (i.e. did not halt at the cert gate) with
    # the expected exit code from our stub, not 4.
    assert excinfo.value.code == 0
    assert len(exploiter_calls) == 1
    assert exploiter_calls[0][0] == 1  # generation


# ----------------------------------------------------------------------------
# main(): first_gen derivation from a resumed episode count.
#
# first_gen = episode // main_ep + 1, where `episode` comes from parsing the
# "checkpoint_<N>.pt" suffix of --resume. Pinned here without running any
# training by stopping main() (via the run_main_phase stub) right after
# first_gen would be used to print/derive `boundary`, using the resume
# filename to control the resumed episode count.
# ----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "resumed_episode, main_episodes, expected_first_gen",
    [
        (0, 1_000_000, 1),
        (999_999, 1_000_000, 1),
        (1_000_000, 1_000_000, 2),
        (1_000_001, 1_000_000, 2),
        (2_500_000, 1_000_000, 3),
    ],
)
def test_main_first_gen_derivation(
    tmp_path, monkeypatch, resumed_episode, main_episodes, expected_first_gen
):
    resume_ckpt = tmp_path / f"pfsp_full_checkpoint_{resumed_episode}.pt"
    PPOAgent(len(ACTIONS)).save(str(resume_ckpt))

    league_dir = tmp_path / "league"
    argv = [
        "train_league_ppo.py",
        "--resume",
        str(resume_ckpt),
        "--league-dir",
        str(league_dir),
        "--run-name",
        f"firstgen_test_{resumed_episode}",
        "--generations",
        "1",
        "--main-episodes",
        str(main_episodes),
        "--num-workers",
        "0",
        "--arch",
        "full",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.chdir(tmp_path)

    seen_generations = []

    def fake_run_main_phase(
        agent,
        league,
        ratings,
        args,
        start_episode,
        n_episodes,
        ckpt_dir,
        anchor_eval=None,
    ):
        # The generation loop's `boundary - episode` argument to
        # run_main_phase, together with start_episode, lets us recover the
        # boundary the loop computed (boundary = generation * main_ep) —
        # and therefore the generation index — without instrumenting main()
        # itself.
        boundary = start_episode + n_episodes
        seen_generations.append(boundary // main_episodes)
        raise SystemExit(99)  # stop main() immediately after the call

    monkeypatch.setattr(train_league_ppo, "run_main_phase", fake_run_main_phase)

    with pytest.raises(SystemExit) as excinfo:
        train_league_ppo.main()

    assert excinfo.value.code == 99
    assert seen_generations == [expected_first_gen]
