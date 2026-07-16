#!/usr/bin/env python3
"""Report writers for run_extended_league.py's Orchestrator.

Extracted verbatim from Orchestrator._write_generations_csv,
_write_report_md, and _write_curve_png: these functions take the state the
Orchestrator used to read off ``self`` as explicit parameters (a dict, a
config dataclass, plain values) rather than the Orchestrator object itself,
so they can be tested/read independently of the training/eval machinery the
rest of the class carries. The Orchestrator's methods of the same name are
now thin delegating calls (see run_extended_league.py); ``write_reports()``
(the public entry point other methods call) is unchanged.
"""

from __future__ import annotations

import csv
import os
import time
from pathlib import Path
from typing import Dict

from sheepshead.analysis.league_progress_eval import PANEL_SEED, load_endpoint


def _fmt(v) -> str:
    if v is None:
        return ""
    try:
        if v != v:  # NaN
            return ""
        return f"{float(v):.4f}"
    except TypeError, ValueError:
        return ""


def write_generations_csv(
    orch_dir: str,
    generations: Dict[str, dict],
    anchor_coeff: float,
    main_episodes: int,
    arch: str,
) -> None:
    """Write ``generations.csv``. ``generations`` is ``Orchestrator.state
    ["generations"]``; ``anchor_coeff`` is ``Orchestrator.state
    ["anchor_coeff"]``; ``main_episodes``/``arch`` are the corresponding
    ``Orchestrator.args`` fields (used in place of the ``boundary``/
    ``boundary_ckpt`` methods, which are pure functions of exactly these)."""
    cols = [
        "generation",
        "boundary_episode",
        "ckpt",
        "panel_mean",
        "panel_lo",
        "panel_hi",
        "panel_se",
        "panel_called",
        "panel_jd",
        "gain_vs_best",
        "gain_lo",
        "gain_hi",
        "gain_p",
        "gain_best_gen",
        "h2h_edge",
        "h2h_se",
        "h2h_win_frac",
        "slope",
        "slope_lo",
        "slope_hi",
        "improving_A",
        "improving_B",
        "climbing_C",
        "flat",
        "flat_streak",
        "exploiter_edge",
        "exploiter_se",
        "exploiter_passed",
        "anchor_coeff",
        "t0_def_leads",
        "t0_forced",
        "t0_trump_lead_rate",
        "t0_trump_prob_mass",
        "t1_def_leads",
        "t1_forced",
        "t1_trump_lead_rate",
        "t1_trump_prob_mass",
        "train_hours",
        "eval_hours",
        "stop_verdict",
    ]
    path = os.path.join(orch_dir, "generations.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        streak = 0
        for g in sorted(int(k) for k in generations):
            rec = generations[str(g)]
            if "verdict" not in rec:
                continue
            v = rec["verdict"]
            streak = streak + 1 if v["flat"] else 0
            leak = rec["panel"]["trump_lead"]
            boundary_episode = g * main_episodes
            ckpt = f"pfsp_{arch}_checkpoint_{boundary_episode}.pt"
            row = {
                "generation": g,
                "boundary_episode": boundary_episode,
                "ckpt": ckpt,
                "panel_mean": f"{rec['panel']['mean']:.4f}",
                "panel_lo": f"{rec['panel']['lo']:.4f}",
                "panel_hi": f"{rec['panel']['hi']:.4f}",
                "panel_se": f"{rec['panel']['se']:.4f}",
                "panel_called": f"{rec['panel']['modes']['called']:.4f}",
                "panel_jd": f"{rec['panel']['modes']['jd']:.4f}",
                "gain_vs_best": f"{v['gain']['mean']:.4f}",
                "gain_lo": f"{v['gain']['lo']:.4f}",
                "gain_hi": f"{v['gain']['hi']:.4f}",
                "gain_p": f"{v['gain']['p_value']:.4f}",
                "gain_best_gen": v["gain"]["best_generation"],
                "h2h_edge": f"{v['h2h']['edge']:.4f}",
                "h2h_se": f"{v['h2h']['se']:.4f}",
                "h2h_win_frac": f"{rec['h2h']['win_frac']:.4f}",
                "slope": f"{v['slope']['mean']:.4f}" if v["slope"] else "",
                "slope_lo": f"{v['slope']['lo']:.4f}" if v["slope"] else "",
                "slope_hi": f"{v['slope']['hi']:.4f}" if v["slope"] else "",
                "improving_A": v["gain"]["improving"],
                "improving_B": v["h2h"]["improving"],
                "climbing_C": v["slope"]["climbing"] if v["slope"] else "",
                "flat": v["flat"],
                "flat_streak": streak,
                "anchor_coeff": anchor_coeff if g == 1 else 0.0,
                "t0_def_leads": leak.get("t0_def_leads", ""),
                "t0_forced": leak.get("t0_forced", ""),
                "t0_trump_lead_rate": _fmt(leak.get("t0_trump_lead_rate")),
                "t0_trump_prob_mass": _fmt(leak.get("t0_trump_prob_mass")),
                "t1_def_leads": leak.get("t1_def_leads", ""),
                "t1_forced": leak.get("t1_forced", ""),
                "t1_trump_lead_rate": _fmt(leak.get("t1_trump_lead_rate")),
                "t1_trump_prob_mass": _fmt(leak.get("t1_trump_prob_mass")),
                "train_hours": f"{rec.get('train_hours', 0.0):.2f}",
                "eval_hours": f"{rec.get('eval_hours', 0.0):.2f}",
                "stop_verdict": rec.get("stop_verdict", ""),
            }
            if "exploiter" in rec:
                row["exploiter_edge"] = f"{rec['exploiter']['edge']:.4f}"
                row["exploiter_se"] = f"{rec['exploiter']['se']:.4f}"
                row["exploiter_passed"] = rec["exploiter"]["passed"]
            w.writerow(row)


def write_report_md(state: dict, args, cfg, alone_limit: float, orch_dir: str) -> None:
    """Write ``report.md``. ``state`` is ``Orchestrator.state``, ``args`` is
    ``Orchestrator.args``, ``cfg`` is ``Orchestrator.cfg`` (a
    ``StopRuleConfig``), and ``alone_limit`` is the value of
    ``Orchestrator._alone_limit()`` at call time."""

    def _now() -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S")

    s = state
    lines = [
        "# Extended league run report",
        "",
        f"*Regenerated {_now()} — status: **{s['status']}***",
        "",
        "## Configuration",
        "",
        f"- arch `{args.arch}`, critic `{args.critic_mode}`, "
        f"resume `{args.resume}` (episode {s['resume_episode']:,})",
        f"- generation length {args.main_episodes:,} eps; "
        f"panel {args.panel_deals} deals (seed {PANEL_SEED}); "
        f"h2h {args.h2h_deals} deals",
        f"- stop rule: {cfg}",
    ]
    if s.get("baseline_health"):
        bh = s["baseline_health"]
        lines.append(
            f"- baseline greedy health: pick {bh['pick_rate']:.1f}%, alone "
            f"{bh['alone_rate']:.1f}% → effective alone halt limit "
            f"{alone_limit:.1f}% (baseline-relative)"
        )
    lines.append("")
    if s.get("calibration"):
        c = s["calibration"]
        lines += [
            "## Anchor calibration",
            "",
            f"Baseline greedy pick rate {c['baseline_pick_rate']:.1f}%. "
            f"Chosen coeff **{c['chosen']}** ({c['reason']}).",
            "",
            "| coeff | kl_last | kl_max | violations | pick % |",
            "|---|---|---|---|---|",
        ]
        for k, p in sorted(c["probes"].items(), key=lambda kv: float(kv[0])):
            lines.append(
                f"| {k} | {p['kl_last']:.4f} | {p['kl_max']:.4f} | "
                f"{p['gate_violations']} | {p['final_pick_rate']:.1f} |"
            )
        lines.append("")
    lines += [
        "## Generations",
        "",
        "| gen | panel score [95% CI] | gain vs best | h2h vs prev | slope | "
        "A/B/C | flat | gate edge | t0 leak | t1 leak |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for g in sorted(int(k) for k in s["generations"]):
        rec = s["generations"][str(g)]
        if "verdict" not in rec:
            continue
        v, p = rec["verdict"], rec["panel"]
        leak = p["trump_lead"]
        abc = (
            f"{'A' if v['gain']['improving'] else '·'}"
            f"{'B' if v['h2h']['improving'] else '·'}"
            f"{('C' if v['slope']['climbing'] else '·') if v['slope'] else '-'}"
        )
        slope_s = (
            f"{v['slope']['mean']:+.4f} [{v['slope']['lo']:+.4f},{v['slope']['hi']:+.4f}]"
            if v["slope"]
            else "—"
        )
        gate = rec.get("exploiter")
        lines.append(
            f"| {g} | {p['mean']:+.4f} [{p['lo']:+.4f},{p['hi']:+.4f}] "
            f"| {v['gain']['mean']:+.4f} (vs g{v['gain']['best_generation']}) "
            f"| {v['h2h']['edge']:+.3f}±{v['h2h']['se']:.3f} "
            f"| {slope_s} | {abc} | {v['flat']} "
            f"| {(f'{gate["edge"]:+.3f}' + (' ✅' if not gate['passed'] else ' ⚔️')) if gate else '—'} "
            f"| {_fmt(leak.get('t0_trump_lead_rate'))} "
            f"| {_fmt(leak.get('t1_trump_lead_rate'))} |"
        )
    lines += [
        "",
        "A = panel gain ≥ MDE vs previous best; B = h2h edge vs previous gen; "
        "C = 3-endpoint slope. Gate: ⚔️ = exploiter inserted (main exploitable), "
        "✅ = main survived (HOF). Leak columns: defender trump-lead rate at "
        "tricks 0/1 given a non-trump option (30M baseline t0 ≈ 0.048; "
        "scripted = 0).",
        "",
    ]
    if s.get("confirmation"):
        c = s["confirmation"]
        gg = c["gain_g_vs_gm2"]
        lines += [
            "## Confirmation (fresh deals, seed 20260706)",
            "",
            f"- last-two-generations gain {gg['mean']:+.4f} "
            f"[{gg['lo']:+.4f},{gg['hi']:+.4f}] p={gg['p_value']:.4f} — "
            + (
                "**CONTRADICTION** (training resumed)"
                if gg["contradiction"]
                else "plateau confirmed"
            ),
            f"- deploy candidate: generation {c['deploy_candidate']['generation']}"
            + (
                f", fresh-deal score {c['deploy_candidate']['score']['mean']:+.4f} "
                f"[{c['deploy_candidate']['score']['lo']:+.4f},"
                f"{c['deploy_candidate']['score']['hi']:+.4f}]"
                if "score" in c["deploy_candidate"]
                else ""
            ),
            "",
        ]
    caveats = ["## Caveats", ""]
    caveats.append(
        "- Trainer resets its OpenSkill training-rating to the prior on each "
        "per-generation invocation; it re-converges within a few thousand "
        "episodes, before the first snapshot inherits it."
    )
    if s["resume_episode"]:
        caveats.append(
            f"- Resume episode {s['resume_episode']:,} > 0: generation 1 "
            "trained fewer episodes than later generations."
        )
    for g in sorted(int(k) for k in s["generations"]):
        v = s["generations"][str(g)].get("verdict")
        if v and v.get("slope") and v["slope"]["small_but_significant"]:
            caveats.append(
                f"- Gen {g}: slope statistically positive but below "
                "SLOPE_MIN — learning may be continuing slowly."
            )
    last = max((int(k) for k in s["generations"]), default=0)
    gate = s["generations"].get(str(last), {}).get("exploiter")
    if s["status"] in ("stopped", "cap") and gate and gate["passed"]:
        caveats.append(
            "- **Stopped while still exploitable**: the final generation's "
            "exploiter cleared its gate."
        )
    lines += caveats + [""]
    if s["events"]:
        lines += ["## Event log", ""]
        lines += [f"- {e['time']}: {e['msg']}" for e in s["events"][-40:]]
        lines.append("")
    with open(os.path.join(orch_dir, "report.md"), "w") as f:
        f.write("\n".join(lines))


def write_curve_png(orch_dir: str, state: dict) -> None:
    """Write ``generations_curve.png``. ``state`` is ``Orchestrator.state``;
    the panel .npz paths are rebuilt from ``orch_dir`` (``panel_npz(g)`` is a
    pure function of exactly that plus ``g``)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gens, means, los, his, t0s, t1s = [], [], [], [], [], []
    panel0 = os.path.join(orch_dir, "panel_gen0.npz")
    base = load_endpoint(Path(panel0)) if os.path.exists(panel0) else None
    if base is not None:
        gens.append(0)
        means.append(base.score.mean)
        los.append(base.score.lo)
        his.append(base.score.hi)
        t0s.append(base.trump_lead.get("t0_trump_lead_rate", float("nan")))
        t1s.append(base.trump_lead.get("t1_trump_lead_rate", float("nan")))
    for g in sorted(int(k) for k in state["generations"]):
        rec = state["generations"][str(g)]
        if "panel" not in rec:
            continue
        gens.append(g)
        means.append(rec["panel"]["mean"])
        los.append(rec["panel"]["lo"])
        his.append(rec["panel"]["hi"])
        t0s.append(rec["panel"]["trump_lead"].get("t0_trump_lead_rate", float("nan")))
        t1s.append(rec["panel"]["trump_lead"].get("t1_trump_lead_rate", float("nan")))
    if len(gens) < 2:
        return

    ink, muted = "#333333", "#767672"
    blue, aqua = "#2a78d6", "#1baf7a"  # validated 2-series palette
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(7.2, 5.6),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 2]},
    )
    for ax in (ax1, ax2):
        ax.grid(True, linestyle=":", alpha=0.35)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=muted, labelsize=9)

    ax1.plot(gens, means, color=blue, lw=2, marker="o", ms=5)
    ax1.fill_between(gens, los, his, color=blue, alpha=0.15, lw=0)
    conf = state.get("confirmation")
    if conf and "score" in conf.get("deploy_candidate", {}):
        cg = conf["deploy_candidate"]["generation"]
        cs = conf["deploy_candidate"]["score"]
        ax1.plot([cg], [cs["mean"]], marker="D", ms=7, color=ink, zorder=5)
        ax1.annotate(
            "confirmation\n(fresh deals)",
            (cg, cs["mean"]),
            textcoords="offset points",
            xytext=(8, -4),
            fontsize=8,
            color=ink,
        )
    ax1.set_ylabel("PANEL-A score/hand\n(95% CI)", fontsize=9, color=ink)
    ax1.set_title(
        "Extended league run: strength and trump-lead leak by generation",
        fontsize=11,
        color=ink,
        loc="left",
    )

    ax2.plot(gens, t0s, color=blue, lw=2, marker="o", ms=5)
    ax2.plot(gens, t1s, color=aqua, lw=2, marker="s", ms=5)
    # Dodge the direct labels vertically when the line ends coincide.
    dodge = 9 if abs(t0s[-1] - t1s[-1]) < 0.004 else 0
    ax2.annotate(
        "trick 0",
        (gens[-1], t0s[-1]),
        textcoords="offset points",
        xytext=(6, dodge),
        fontsize=9,
        color=blue,
    )
    ax2.annotate(
        "trick 1",
        (gens[-1], t1s[-1]),
        textcoords="offset points",
        xytext=(6, -dodge),
        fontsize=9,
        color=aqua,
    )
    ax2.axhline(0.048, color=muted, lw=1, ls="--")
    ax2.annotate(
        "30M baseline t0",
        (gens[0], 0.048),
        fontsize=8,
        color=muted,
        textcoords="offset points",
        xytext=(0, 3),
    )
    ax2.set_ylabel(
        "defender trump-lead rate\n(non-trump option held)", fontsize=9, color=ink
    )
    ax2.set_xlabel("generation", fontsize=9, color=ink)
    ax2.set_ylim(bottom=0)
    ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    fig.tight_layout()
    fig.savefig(os.path.join(orch_dir, "generations_curve.png"), dpi=150)
    plt.close(fig)
