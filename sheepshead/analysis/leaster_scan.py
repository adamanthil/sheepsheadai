#!/usr/bin/env python3
"""Scan self-play run logs for the always-PASS (all-leaster) collapse.

The collapse: every seat learns to PASS every deal early in from-scratch
shaped-reward self-play, pinning the rollout leaster rate near 100% while
the bidding heads' entropy freezes at 0. Detected here from the trainer's
periodic "Leaster Rate: X%" log lines, positioned by the nearest preceding
"(Episode N)" marker.

Usage:
    PYTHONPATH=. python analysis/leaster_scan.py [glob ...]
Default glob: runs/ablate_*

Output: one line per run — peak rate, fraction of samples in hard collapse
(>=90%), the longest hard-collapse span, and any soft-dip (>=30%) spans.
Healthy from-scratch runs sit around 2-10% after the first few thousand
episodes; a >=30% span of any length is abnormal.
"""

import glob
import os
import re
import sys

EP_RE = re.compile(r"\(Episode ([\d,]+)\)")
LR_RE = re.compile(r"Leaster Rate: ([\d.]+)%")


def scan_log(path):
    ep = 0
    recs = []
    with open(path, errors="replace") as f:
        for line in f:
            m = EP_RE.search(line)
            if m:
                ep = int(m.group(1).replace(",", ""))
            m = LR_RE.search(line)
            if m:
                recs.append((ep, float(m.group(1))))
    return recs


def spans(recs, thresh):
    """Maximal consecutive spans with rate >= thresh -> [(start_ep, end_ep)]."""
    out = []
    cur = None
    for ep, r in recs:
        if r >= thresh:
            if cur is None:
                cur = [ep, ep]
            cur[1] = ep
        elif cur is not None:
            out.append(tuple(cur))
            cur = None
    if cur is not None:
        out.append(tuple(cur))
    return out


def fmt_span(s):
    return f"{s[0] // 1000}k-{s[1] // 1000}k"


def main(argv):
    patterns = argv[1:] or ["runs/ablate_*"]
    dirs = sorted(d for p in patterns for d in glob.glob(p))
    print(
        f"{'run':<38} {'last_ep':>8} {'peak%':>6} {'frac>=90':>8} "
        f"{'hard spans (>=90%)':<24} soft spans (>=30%)"
    )
    for d in dirs:
        log = os.path.join(d, "train.log")
        if not os.path.exists(log):
            continue
        recs = scan_log(log)
        name = os.path.basename(d)
        if not recs:
            print(f"{name:<38} (no leaster samples yet — log buffered?)")
            continue
        peak = max(r for _, r in recs)
        frac_hard = sum(1 for _, r in recs if r >= 90) / len(recs)
        hard = spans(recs, 90.0)
        soft = spans(recs, 30.0)
        print(
            f"{name:<38} {recs[-1][0]:>8} {peak:>6.1f} {frac_hard:>8.2f} "
            f"{', '.join(fmt_span(s) for s in hard) or '-':<24} "
            f"{', '.join(fmt_span(s) for s in soft) or '-'}"
        )


if __name__ == "__main__":
    main(sys.argv)
