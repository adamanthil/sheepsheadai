# Convention Optimality — Definitive Study (October 2026)

**Status (2026-10-31): C1 (trick-0 defender trump lead) DONE. C2
(called-suit lead) trick-0 rate and cost DONE; later tricks PENDING.
§10 (what search adds: league endpoint theta0 vs release theta6) DONE.**

This study supersedes `Convention_Optimality_202607.md` and its
2026-10-16 addendum as the reference for both conventions. Those remain
the early evidence. This one uses the fully trained release agent
(theta6, `models/recall-rc-theta6.pt`), the production 30M
(`models/pfsp-swish-30m.pt`), and the six policy-iteration search
corpora of run `202609_recall_rc`.

All values are **game score** (the leader's `get_score()`), not reward
units. Search Q is stored as score / 12 and is multiplied by 12 here.

Artifacts: `runs/convention_optimality_202610/t0_trump_lead/`
(gitignored; scripts, probe records, corpus extract, replay results).

## TL;DR (C1)

1. **Rates.** On identical seeded deals, theta6 leads trump at trick 0
   as a defender far less than the 30M in called-ace mode, slightly
   more in JD mode, and less overall.
2. **Where.** Theta6's trump leads are a pattern. More than half fall
   in two hand types: JD with two or more fail aces and a lone diamond
   trump or three trump, and called-ace with four or more trump, a
   called-suit void and a fail ace. The 30M's are scattered. Half of
   its JD trump leads come against a picker who went alone.
3. **Cost.** On true-deal replays with each agent's own continuation,
   theta6's leads cost −0.08 ± 0.03 per lead in both modes. The 30M's
   cost −0.19 ± 0.06 in JD and −0.40 ± 0.09 in called ace. Search
   prices the 30M's leads at about −0.47 per lead and theta6's at
   −0.04 (JD) and −0.15 (called ace).
4. **Rules.** Search and theta6 agree on a small set of hand types
   where a trump lead is acceptable. Search rates them +0.07 to +0.25.
   Policy replays rate the same hands −0.11 to +0.06. Both estimators
   put them inside or near the tie band; neither finds a large gain.
   The sign is unresolved (see §6).
5. Outside those hand types, both estimators agree that the trump lead
   is clearly wrong: −0.1 to −1.0 per lead.

## 1. Definition (C1 node)

The program's convention battery
(`sheepshead/eval/health_probe.py`, `is_t0_def_lead`). A node is an
opportunity when:

- play has started, the hand is not a leaster;
- it is trick 0, no card is on the table, and the player is the leader;
- the player is not the picker, not the partner, not the secret partner;
- the hand holds at least one trump and at least one fail.

The rate is greedy trump leads over opportunities. Hands where the
picker went alone are included (every non-picker is a defender).

"Fail aces" means aces of clubs, spades and hearts. "Void" means no
card of some fail suit. "Diamond trump" means a trump that is not a
queen or jack.

## 2. Instruments

| instrument | what it estimates | known bias |
|---|---|---|
| **Seeded probe** (`t0_probe.py`, `t0_probe_v2.py`) | greedy behavior in self-play, agent in all 5 seats; deal `g` uses `Game(seed=20261030·1000003+g)`, so both agents see identical deals | the opportunity set depends on each agent's own bidding |
| **Search corpora** (`extract_corpus.py`) | trump minus fail, with each side's card chosen by the act-time prior (avoids the winner's curse); ISMCTS committee, t0 leads at 1024 iterations × 3 replicates, iterations 1-6, 23,063 nodes | determinized search (strategy fusion), search continuation, prior of theta_{k−1} |
| **True-deal replay, greedy** (`replay.py`) | replay the seeded game to the node, force the agent's preferred trump vs its preferred fail, finish greedily with the same agent; unbiased for cell averages under the agent's own play | per-node noise (SD ≈ 1.6 score); measures value under the policy's continuation, not under best play |
| **True-deal replay, sampled** (`replay_sampled.py`) | as above, continuation sampled at T = 0.5 (deployed play temperature), 4 runs per branch with common seeds | same |

The program's own reads were unseeded (`Game()` draws from OS entropy),
which is why the program noted ~1 pp run-to-run noise. Its final theta6
read (0.84%, both modes pooled, about 1,900 opportunities) agrees with
this study's 0.72% within that noise.

The rich probe (`t0_probe_v2.py`) reproduced the first probe's choices
on all 188,959 records.

## 3. Rates

Theta6 on 600,000 games per mode, the 30M on 100,000. 95% Wilson
intervals.

| mode | 30M | theta6 | 30M − theta6 (first 100k games) |
|---|---|---|---|
| JD | 1.008% [0.920, 1.105] (453/44,936) | 1.169% [1.130, 1.209] (3,306/282,904) | −0.21 pp, z −3.0 |
| called ace | 0.790% [0.715, 0.873] (379/47,979) | 0.296% [0.277, 0.316] (871/294,631) | +0.47 pp, z 9.7 |
| overall | 0.895% [0.837, 0.958] (832/92,915) | 0.723% [0.702, 0.745] (4,177/577,535) | +0.13 pp, z 3.2 |

On the 41,279 JD nodes both agents reached with the same deal, seat and
hand, the JD gap is not significant (McNemar z −1.7). The two agents
lead trump in almost disjoint places: of about 84,000 shared nodes,
both led trump at 33.

## 4. Where the leads fall

Trump-lead rate by hand type, both agents on the same first 100,000
games per mode. "Share" is the hand type's share of that agent's trump
leads in that mode. §5 gives theta6's rates over all 600,000 games.

| hand type | theta6 rate | theta6 share | 30M rate | 30M share |
|---|---|---|---|---|
| JD: 2+ fail aces, no Q/J, no void | 29% | 42% | 0.2% | 0.4% |
| JD: 4+ trump | 9% | 11% | 28% | 18% |
| JD: picker alone | 0.4% | 5% | 9.1% | 51% |
| JD: everything else | 0.5-0.9% | 42% | 0.0-0.4% | 30% |
| CA: 4+ trump, called void, fail ace | 42% | 63% | 8.1% | 8% |
| CA: 4+ trump, called void, no fail ace | 3% | 17% | 6.4% | 17% |
| CA: <4 trump, called void | 0.2% | 13% | 1.1% | 28% |
| CA: holds a called-suit card | 0.0% | 0% | 0.5-2.2% | 46% |
| CA: picker alone | 0.1% | 7% | 0.1% | 1% |

Pricing each lead at its hand type's average search value
(`behavior.py`): theta6 −0.040 per lead (JD) and −0.148 (called ace),
or −0.5 score per 1,000 opportunities in each mode. 30M −0.457 and
−0.489 per lead, or −4.6 and −3.9 per 1,000 opportunities.

## 5. Value by hand type

Non-alone hands unless stated. Search columns are corpus means
(n nodes) and the split by iteration halves. Replay is greedy, over all
of theta6's nodes in the hand type, or a random 3,000 for the large
ones.

| hand type | search | search it1-3 / it4-6 | replay (theta6) | theta6 rate | 30M rate |
|---|---|---|---|---|---|
| JD: 2+ aces, 1 diamond trump, no void | +0.111 ± 0.014 (152) | +0.115 / +0.105 | −0.046 ± 0.025 (3,491) | 35.4% | 0.0% |
| JD: 2+ aces, 1 diamond trump, void | −0.134 ± 0.020 (48) | −0.123 / −0.140 | −0.147 ± 0.044 (1,183) | 26.0% | 0.0% |
| JD: 2+ aces, 1 trump (Q or J) | −0.110 ± 0.016 (211) | −0.069 / −0.154 | −0.171 ± 0.020 (5,502) | 3.3% | 0.0% |
| JD: 2+ aces, 2 trump | −0.176 ± 0.013 (384) | −0.132 / −0.220 | −0.110 ± 0.015 (10,111) | 3.5% | 0.1% |
| JD: 2+ aces, 3 trump | +0.130 ± 0.027 (62) | +0.059 / +0.222 | +0.000 ± 0.038 (1,560) | 48.3% | 2.3% |
| JD: 4+ trump | −0.051 ± 0.015 (145) | −0.043 / −0.056 | −0.088 ± 0.029 (3,201) | 9.3% | 28.3% |
| JD: <2 aces, 1-3 trump | −0.342 ± 0.003 (8,720) | −0.309 / −0.375 | −0.288 ± 0.028 (3,029) | 0.0% | 0.3% |
| JD: picker alone | −0.674 ± 0.008 (1,701) | −0.626 / −0.729 | −0.410 ± 0.021 (3,296) | 0.4% | 9.1% |
| CA: 4+ trump, called void, fail ace | +0.092 ± 0.024 (54) | +0.116 / +0.066 | −0.121 ± 0.053 (1,239) | 42.1% | 8.1% |
| CA: 4+ trump, called void, no fail ace | −0.132 ± 0.015 (247) | −0.125 / −0.138 | −0.229 ± 0.023 (6,280) | 3.2% | 6.4% |
| CA: 3 trump, called void, 2 fail aces | −0.203 ± 0.055 (25) | −0.373 / −0.089 | −0.294 ± 0.069 (701) | 12.7% | 2.7% |
| CA: 3 trump, called void, <2 aces | −0.553 ± 0.010 (1,074) | −0.535 / −0.568 | −0.486 ± 0.011 (28,198) | 0.0% | 1.8% |
| CA: holds called suit, or <3 trump | −0.654 ± 0.004 (7,760) | −0.634 / −0.674 | −0.520 ± 0.030 (3,018) | 0.0% | 0.6% |
| CA: picker alone | −0.721 ± 0.007 (2,480) | −0.693 / −0.756 | −0.495 ± 0.023 (3,040) | 0.1% | 0.1% |

The hand types were found with depth-5 regression trees mined on
iterations 1-3 and confirmed on 4-6 (`tree.py`), then simplified by
hand (`cells.py`, `final_table.py`).

### 5.1 Each agent's own leads (replay, own continuation)

| agent, mode | n | trump − fail |
|---|---|---|
| theta6, JD | 3,306 | −0.084 ± 0.027 |
| theta6, called ace | 871 | −0.084 ± 0.062 |
| 30M, JD | 453 | −0.192 ± 0.064 |
| 30M, called ace | 379 | −0.396 ± 0.092 |

The 30M's JD leads against a lone picker alone cost −0.38 ± 0.07
(n 238).

### 5.2 Theta6's choice within the hand types

Theta6's choice inside a hand type follows sub-rules that match the
search's ranking (`final_table.py`, sub-cell table in the session
log):

| sub-cell | search | replay (sampled) | theta6 rate |
|---|---|---|---|
| JD 1 diamond trump, no void: 3 fail aces | +0.248 (16) | −0.009 ± 0.063 | 91% |
| … 2 aces, picker immediately after me | +0.125 (55) | −0.070 ± 0.036 | 51% |
| … 2 aces, picker 2-4 seats after me | +0.074 (81) | −0.051 ± 0.025 | 16% |
| JD 3 trump, 2+ aces, no queen, picker 1-2 after | +0.208 (30) | +0.057 ± 0.042 | 63% |
| … no queen, picker 3-4 after | +0.068 (27) | +0.025 ± 0.047 | 36% |
| … holds a queen | −0.003 (5) | −0.052 ± 0.098 | 7% |
| CA 4+ trump, void, ace: top trump J or diamond | +0.114 (47) | −0.113 ± 0.050 | 41% |
| … holds a queen | −0.081 (5) | −0.147 ± 0.140 | 11% |

Theta6 learned the search's preferences faithfully. The replays do not
separate these sub-cells.

One within-cell split does show up in the replays: in the called-ace
hand type, the nodes where theta6 led trump replay at +0.02 ± 0.06,
and the nodes where it led fail at −0.21 ± 0.06 (sampled; greedy
−0.00 / −0.21). Theta6's choice uses only its own information, so this
is no hindsight effect. The features behind it were not isolated.

## 6. The estimator disagreement

In the four hand types search rates positive, the replays come in
about 0.12 to 0.22 lower. In the strongly negative hand types the
replays come in less negative. The disagreement is systematic, not
noise.

- **Not continuation temperature.** Sampled continuation at T = 0.5
  matches greedy in every hand type tested (`replay_sampled.py`):
  JD lone diamond −0.054 vs −0.046, JD 3 trump +0.036 vs 0.000,
  called-ace with a fail ace −0.112 vs −0.121, called-ace no ace −0.213 vs
  −0.209.
- **Not search budget.** The 2026-10-16 addendum re-searched JD
  singleton nodes at 4,096 iterations; the verdict moved +0.03.
- **Not selection.** The search's positive hand types hold up on
  held-out iterations (§5).
- **Remaining candidates.** (a) Strategy fusion: a determinized search
  plays each world with knowledge of the hidden cards, which flatters
  plans that need that knowledge, such as drawing the picker's trump
  and then cashing aces. (b) Continuation strength: search plays the
  rest of the hand better than the raw policy, and the trump lead may
  only pay off with that better play. Under (a) the trump leads are
  slightly wrong; under (b) they are right for a searching player and
  slightly wrong for the raw policy.

**Adjudication experiment (not run).** Replay the true deal with the
leader's later decisions made by search at the deploy budget, other
seats raw policy. If the positive hand types turn positive, (b) holds;
if they stay near −0.05, (a) does. Rough cost: about 1,000 nodes per
hand type, two branches, about ten searched decisions each.

The practical stakes are small either way. Theta6 makes these leads on
about 0.7% of opportunities, and the disagreement is about 0.15 per
lead, or roughly 0.001 score per opportunity.

## 7. Rules

Firm (both estimators agree, large n): as a defender on the opening
lead, **lead fail** when:

1. the picker went alone (−0.4 to −0.7);
2. you hold fewer than two fail aces and three or fewer trump (−0.3);
3. in called ace, you hold a card of the called suit (−0.5 to −0.65;
   lead the called suit, per C2);
4. in called ace, you hold three or fewer trump (−0.2 to −0.55);
5. in JD, your only trump is a queen or jack (−0.11 to −0.17);
6. in JD, you hold two trump, even with two aces (−0.11 to −0.18);
7. in JD, your only trump is a diamond and you have a fail void: it is
   your ruff (−0.13 to −0.15).

Tie band (search positive, replay at or slightly below zero): a trump
lead is acceptable, and search prefers it, when:

8. JD: two or more fail aces, a single diamond trump, no fail void.
   Strongest with three aces or with the picker directly after you.
9. JD: two or more fail aces and three trump, no queen. Strongest with
   the picker one or two seats after you.
10. Called ace: four or more trump, no card of the called suit, at
    least one fail ace, no queen.

Near-tie but leaning fail: JD with four or more trump (search −0.05,
replay −0.09); called ace with four or more trump, called void and no
fail ace (−0.13 / −0.23).

## 8. Answer to the motivating question

Is the 30M's trump-lead rate noise and theta6's a pattern? Yes.

- Theta6's leads concentrate in rules 8-10 and follow the search's
  sub-ranking within them. They cost between −0.08 per lead (replay)
  and roughly +0.1 (search) — at worst a tie-band cost.
- The 30M's leads are spread at low rates across nearly every hand
  type, including the worst (lone picker, holding the called suit).
  Both estimators price them at −0.2 to −0.5 per lead.

Whether theta6's pattern is strictly better than leading fail in rules
8-10, rather than a tie, is the open question in §6.

## 9. C2 — called-suit lead

### 9.1 Trick-0 rate

Node: the program's called-suit definition (`health_probe.py`,
`is_called_suit_lead`) restricted to trick 0. A true defender leads in
called-ace mode, the called suit has not been led, and the hand holds a
fail of the called suit and some alternative. Adherent means the lead is
a called-suit fail. Same seeded deals as §3, 100,000 called-ace games
per agent (`cs_probe.py`). Artifacts:
`runs/convention_optimality_202610/t0_called_suit/`.

| agent | trick-0 called-suit lead | non-adherent leads |
|---|---|---|
| theta6 | 99.97% [99.94, 99.99] (27,700/27,708) | 8 other fail, 0 trump |
| 30M | 95.23% [94.98, 95.47] (28,044/29,448) | 1,229 other fail, 175 trump |

Under-calls: theta6 100.00% (2,447/2,447), 30M 93.50% (2,345/2,508).
On the 22,238 nodes both agents reached with the same hand, theta6
alone led the called suit at 930 and the 30M alone at 5 (McNemar
z 30.3). The July study's 30M reads (94.0% scripted field, 94.4%
self-play) agree.

### 9.2 Trick-0 value

- **Search** (`cs_corpus.py`, iterations 1-6, non-alone, 6,685 nodes):
  the preferred called-suit card minus the preferred alternative is
  +0.417 ± 0.004. By called-suit length: one card +0.327, two +0.537,
  three or more +0.882. Under-call +0.274. Search's top card is a
  called-suit card at 86.9% of nodes, and only 5.2% of nodes have a
  negative difference.
- **Replay of the 30M's deviations** (`replay_cs.py`, 30M continuation,
  all 1,404): leading the called suit instead gains +0.144 ± 0.040 per
  deviation. Where the 30M led trump, +0.497 ± 0.128 (175). Where it led
  another fail, +0.094 ± 0.042 (1,229).

At trick 0, both estimators agree the convention holds without
exception at this policy level, consistent with addendum A2 of the July
study. Theta6 has fully internalized it. The 30M's 4.8% of deviations
are real mistakes worth about −0.007 score per opportunity.

### 9.3 Later tricks

PENDING. Addendum A2 found the one known exception: from trick 3 on, a
leader with no trump and a single called-suit card should not lead it
(−0.21, −0.42 with two cards left). The program's 93.6% all-tricks rate
for theta6 mixes that cell with trick 0.

## 10. What search adds: theta0 vs theta6

theta0 is the league-phase endpoint of `202609_recall_rc`, the
checkpoint the program handed to policy iteration
(`runs/202609_recall_rc/league/checkpoints/checkpoint_8000000.pt`;
`program.log` 2026-10-06 06:56 "HANDOFF after gen 8"). theta6 is the
same architecture after six search-teacher iterations. The pair isolates
what search added at these nodes. Every probe and replay of §3-§5 and
§9 was rerun on theta0 with the same seeded deals. Artifacts:
`runs/convention_optimality_202610/theta0/`.

### 10.1 Trick-0 trump-lead rates

100,000 seeded games per mode, same deals as §3. McNemar is paired on
nodes all three agents share (same game, seat and hand).

| mode | theta0 | theta6 | 30M | paired theta0 vs theta6 |
|---|---|---|---|---|
| JD | 5.366% [5.165, 5.574] (2,504/46,668) | 1.216% (573/47,141) | 1.008% (453/44,936) | theta0-only 2,085, theta6-only 132, z +41.5 |
| called ace | 0.743% [0.672, 0.823] (372/50,038) | 0.325% (159/48,903) | 0.790% (379/47,979) | 317 vs 90, z +11.3 |
| overall | 2.974% [2.869, 3.083] (2,876/96,706) | 0.762% (732/96,044) | 0.895% (832/92,915) | 2,402 vs 222, z +42.6 |

The program's own gen-8 read was 3.40% overall (4,000 unseeded games),
consistent with this one. In called-ace mode theta0 and the 30M are
statistically equal (paired z −1.6); in JD theta0 leads trump four to
five times as often as either.

### 10.2 Where theta0's leads fall

Most of theta0's JD trump leads follow one rule the search phase pruned:
**with a single trump that is a jack or a diamond, lead it whenever
holding a fail ace.** By trump class and fail aces (non-alone JD hands,
one trump):

| lone trump | fail aces | theta0 | theta6 | 30M |
|---|---|---|---|---|
| Q | any | 0-4% | 0% | 0% |
| J | 1 | 30% | 0% | 0% |
| J | 2+ | 35% | 8% | 0% |
| A/10/K of diamonds | 1 | 9% | 0% | 0% |
| A/10/K of diamonds | 2+ | 94-97% | 10-27% | 0% |
| 7/8/9 of diamonds | 1 | 51-54% | 0.3% | 0% |
| 7/8/9 of diamonds | 2+ | 81-85% | 44-47% | 0% |
| any | 0 | 0-8% | 0% | 0% |

1,786 of theta0's 2,504 JD trump leads (71%) are in the one-trump,
one-plus-ace family. Search kept only the two-ace diamond corner
(§7, rule 8) and cut the rest.

### 10.3 Value by hand type

Search columns are the cell average of the prior-preferred trump minus
the prior-preferred fail over all corpus nodes in the hand type
(game score); "it1" uses only iteration 1, whose prior was theta0
itself. Replay is the paired true-deal replay of theta0's own trump
leads, theta0 continuation (`replay.py`).

| hand type (non-alone unless noted) | theta0 rate | theta6 rate | 30M rate | search, it1-6 (n) | search, it1 | replay, theta0's own leads (n) |
|---|---|---|---|---|---|---|
| JD: 1 trump (J or diamond), 2+ fail aces | 70.0% | 25.0% | 0.0% | −0.013 ± 0.013 (304) | −0.006 ± 0.032 | −0.139 ± 0.056 (784) |
| JD: 1 trump (J or diamond), 1 fail ace | 30.3% | 0.1% | 0.0% | −0.181 ± 0.009 (711) | −0.096 ± 0.024 | −0.130 ± 0.051 (1,002) |
| JD: 1 trump (Q), 1+ fail aces | 3.8% | 0.0% | 0.0% | −0.205 ± 0.011 (469) | −0.134 ± 0.027 | −0.347 ± 0.192 (75) |
| JD: 1 trump, no fail ace | 0.7% | 0.0% | 0.0% | −0.306 ± 0.009 (848) | −0.252 ± 0.019 | +0.04 ± 0.28 (25) |
| JD: 2-3 trump, 2+ fail aces | 8.7% | 9.6% | 0.3% | −0.134 ± 0.013 (446) | −0.051 ± 0.029 | **+0.261 ± 0.128** (142) |
| JD: 2-3 trump, <2 fail aces | 1.3% | 0.0% | 0.5% | −0.369 ± 0.003 (6,799) | −0.311 ± 0.008 | +0.020 ± 0.088 (350) |
| JD: 4+ trump | 0.0% | 11.8% | 28.3% | −0.051 ± 0.015 (145) | −0.027 ± 0.041 | — (0) |
| CA: 4+ trump, called void, fail ace | 0.9% | **45.0%** | 8.1% | **+0.092 ± 0.024** (54) | +0.151 ± 0.065 | — (1) |
| CA: 4+ trump, called void, no fail ace | 0.7% | 2.7% | 6.4% | −0.132 ± 0.015 (247) | −0.215 ± 0.037 | +0.4 ± 0.4 (5) |
| CA: <4 trump, called void | 2.0% | 0.2% | 1.1% | −0.613 ± 0.008 (2,174) | −0.566 ± 0.020 | −0.416 ± 0.119 (185) |
| CA: holds called suit | 0.7% | 0.0% | 0.6% | −0.649 ± 0.005 (6,685) | −0.623 ± 0.011 | −0.206 ± 0.117 (180) |
| JD: picker went alone | 1.5% | 0.4% | 9.1% | −0.674 ± 0.008 (1,701) | −0.573 ± 0.016 | −0.03 ± 0.16 (126) |
| CA: picker went alone | 0.0% | 0.1% | 0.1% | −0.721 ± 0.007 (2,480) | −0.654 ± 0.013 | — (1) |

All of theta0's own trump leads, replayed: JD −0.089 ± 0.032 (2,504),
called ace −0.298 ± 0.082 (372), overall −0.116 ± 0.030.

The search phase did two things at this node:

1. **Pruned.** It removed theta0's one-trump dump outside the two-ace
   diamond corner, every called-ace trump lead while holding the called
   suit or short of trump, and most lone-picker leads. Both estimators
   price those leads negative: replay −0.13 to −0.42, search −0.18 to
   −0.65.
2. **Installed.** It created leads theta0 almost never made. The
   clearest is called ace with 4+ trump, a called-suit void and a fail
   ace: 0.9% → 45%, the only hand type search prices positive at
   2 SE (+0.09 overall, +0.15 on theta0's own iteration). JD with 4+
   trump went 0% → 12% at a search value near zero (−0.05). §6's caveat
   applies: theta6's own replays put the installed cell near zero, not
   positive.

One hand type disagrees with search in the other direction: theta0's
142 trump leads in JD with 2-3 trump and 2+ fail aces replay at
+0.26 ± 0.13 (2.0 SE) under theta0's continuation, while search's cell
average is −0.13. theta0 picks a subset of those hands (8.7%), so the
cell average is not the price of its choices; theta6 leads at the same
rate there (9.6%), concentrated in the three-trump subcell search
rates +0.13 (§5).

### 10.4 Cost per opportunity (replay, own continuation)

| agent | JD | called ace |
|---|---|---|
| theta0 | −4.8 per 1,000 | −2.2 per 1,000 |
| 30M | −1.9 per 1,000 | −3.1 per 1,000 |
| theta6 | −1.0 per 1,000 | −0.3 per 1,000 |

The search phase recovered about 3.8 (JD) and 1.9 (called ace) score
per 1,000 trick-0 defender leads — 0.004 and 0.002 per opportunity.
This is a small part of the +0.034 per-game play-only gain of theta6
over theta0 (Training_Program_Redesign decision log, 10-27). Each of
theta0's trump leads cost little (−0.09 in JD); the loss came from how
often it led trump.

### 10.5 Trick-0 called-suit lead

Same 100,000 called-ace games as §9.1 (`cs_probe.py`).

| agent | trick-0 called-suit lead | non-adherent | under-calls |
|---|---|---|---|
| theta0 | 98.83% [98.69, 98.95] (26,505/26,820) | 135 other fail, 180 trump | 98.62% (2,362/2,395) |
| theta6 | 99.97% (27,700/27,708) | 8 other fail, 0 trump | 100% |
| 30M | 95.23% (28,044/29,448) | 1,229 other fail, 175 trump | 93.50% |

Paired on 24,624 shared nodes, theta0 never followed the convention
where theta6 did not; theta6 alone followed it at 262 nodes (McNemar
z −16.2). theta0 vs the 30M: 844 vs 171 (z +21.1).

Replay of theta0's 315 deviations (`replay_cs.py`, theta0
continuation): leading the called suit instead gains
+0.194 ± 0.087 per deviation — +0.228 ± 0.119 where theta0 led trump
(180), +0.148 ± 0.125 where it led another fail (135). Cost about
−0.0023 per opportunity, a third of the 30M's.

The league phase had already installed the trick-0 convention almost
fully (98.8%). Search removed the remaining 1.2%, all of it in the
direction both estimators price as an error.

### 10.6 Search's view of each prior

The corpora give a direct prior-vs-search read per iteration
(`corpus_iter.py`). The search node sample is boosted toward lead
nodes, so rates differ from the probe. "Top-Q trump" is inflated by
the winner's curse and is descriptive only.

| iteration (prior) | mode | prior argmax trump | search top-Q trump | trump−fail where the prior led trump (n) |
|---|---|---|---|---|
| 1 (theta0) | JD | 5.41% | 8.78% | +0.002 ± 0.021 (101) |
| 1 (theta0) | CA | 1.21% | 2.27% | −0.384 ± 0.059 (24) |
| 2 (theta1) | JD | 3.54% | 9.78% | +0.013 ± 0.024 (67) |
| 3 (theta2) | JD | 3.07% | 8.91% | +0.113 ± 0.018 (60) |
| 4 (theta3) | JD | 1.66% | 7.22% | +0.044 ± 0.029 (32) |
| 5 (theta4) | JD | 1.38% | 8.22% | +0.134 ± 0.041 (26) |
| 6 (theta5) | JD | 1.16% | 6.73% | +0.266 ± 0.063 (22) |

Called-ace prior trump leads fell to 0-6 per iteration from iteration 2
on (too few to price). In JD, each iteration kept fewer trump leads and
the survivors were priced higher by search: break-even at theta0,
+0.27 at theta5. The search phase did not suppress the trump lead
uniformly; it selected the hands where search rates it best.

### 10.7 Answer

Search contributed three things at these nodes:

1. It cut theta0's trick-0 defender trump leads by three quarters
   (2.97% → 0.76%), mostly by removing one broad rule (dump a lone
   jack or diamond trump when holding a fail ace) and keeping its
   two-ace diamond corner.
2. It installed a lead theta0 did not make (called ace, 4+ trump,
   called void, fail ace), the one hand type search prices positive.
   Its value is unresolved by replay (§6).
3. It finished the trick-0 called-suit convention (98.8% → 99.97%).

In score, this is worth about 0.003 per trick-0 defender lead
opportunity (trump lead, both modes averaged) plus 0.002 per trick-0
called-suit opportunity. The convention behaviour that distinguishes theta6 from
the 30M was mostly installed by search, not by the league phase: theta0
is no better than the 30M in called-ace mode and worse in JD.

## Reproduction

From the repo root, with `R=runs/convention_optimality_202610/t0_trump_lead`:

```
uv run python $R/t0_probe_v2.py --ckpt models/recall-rc-theta6.pt --mode 0 --n 100000 --out $R/rich_recall-rc-theta6_m0.json
uv run python $R/extract_corpus.py $R/corpus_t0.pkl
uv run python $R/corpus_cells.py $R          # writes corpus_feats.pkl
uv run python $R/behavior.py $R
uv run python $R/replay.py --ckpt models/recall-rc-theta6.pt --nodes $R/nodesB_theta6.json --out $R/replayB_theta6_0.jsonl --shard 0 --nshards 8
uv run python $R/final_table.py $R
```

`t0_probe.py` / `t0_probe_v2.py` take `--start` for game ranges
(the extension used 100,000-600,000). Node lists: `nodesA_*` (each
agent's trump leads plus the JD two-ace hand type), `nodesB_theta6`
(all theta6 nodes in the JD two-ace and called-ace 3+-trump
void families, plus all extension trump leads), `nodesC_theta6`
(JD 4+ trump and random controls), `nodesS_theta6` (sampled-replay set).
