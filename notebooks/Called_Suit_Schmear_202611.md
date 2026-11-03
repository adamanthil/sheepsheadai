# Called-Suit Schmear into the Hidden Partner — November 2026

**Status (2026-11-02): incidence on the release, the operator's
"speculative schmear" hypothesis, and the search-phase trend DONE.
Search read at the wrong-schmear nodes and a "last seat is the partner"
probe PENDING (§7).**

The operator played the deployed release (θ₆, `final/release.pt` =
`models/recall-rc-theta6.pt`, deploy sampling play T 0.5 / band 1.0) and
saw a defender, out of trump and void in the called suit, put an
offsuit ace on a called-suit trick when only the hidden partner was left
to play. A non-point fail was in hand. This notebook measures how often
that happens, tests why, and tracks it across the search phase of run
`202609_recall_rc`. It is the sibling of the called-suit over-trump
(`Arm_D_Review_202610.md` §9): same trick, same follow rule, opposite
direction of error.

## TL;DR

- **Incidence on θ₆:** the spot comes up in 0.6 of every 100 called-ace
  deals. The argmax plays the A/10 **15.8% ± 2.4** of the time (deployed
  17.7%), 1.7 points gifted per decision. That is about one gifted A/10
  per 1,000 called-ace deals.
- **It is not a lack of the general rule:** when the picking team's win
  is visible (defender last to play, picker or revealed partner
  winning), θ₆ throws an A/10 only **0.8%** (called-ace) / **1.2%** (JD).
- **Operator hypothesis, half supported.** The rate does not depend on
  whether a defender is still to act (0 left 15.8%, 1 left 17.3%, flat
  within tricks), so the agent does not check whether anyone remains who
  could trump. But it is not schmearing blindly: with a defender's trump
  already winning it schmears **78.6%**, and the rate does not rise with
  more defenders left, as a gamble on a teammate trumping would.
- **Reading:** the agent does not deduce that the one remaining unseen
  seat is the hidden partner, who must play the called card and cannot
  trump.
- **Search phase:** the wrong-schmear rate fell 63% → 17% (θ₀ → θ₆),
  almost all of it by θ₃, and has been flat since. It fell together
  with the ambiguous cell (a defender still to act), which ended at the
  same rate: the search taught "schmear less on untrumped called-suit
  tricks", not the seat deduction. The over-trump, by contrast, was
  still falling at θ₆ (§9.19: 40.9% → 22.0%).

## 1. The spot and the rule

Called-ace mode, the called suit led, the called card not yet played (so
the partner is hidden). The rules for the seats that follow
(`game.py` ~1255-1280; `reference_called_suit_follow_rules`):

- the picker must follow the called suit (they hold a called-suit card,
  or the under card on an under-call);
- the hidden partner must play the called card.

Neither can trump. So for a defender who is void in the suit and holds no
trump:

| trick state when the defender acts | who wins | right play |
|---|---|---|
| no trump in, no defender left to act | the hidden partner, certainly (the called ace, or a called 10 on the first called-suit lead) | the cheapest card |
| no trump in, a defender still to act | the partner unless that defender trumps | depends on that defender's hand |
| a defender's trump winning | the defenders, certainly (only defenders can trump) | schmear |

The first row is the operator's spot. It needs no inference about hidden
cards beyond what the rules force: the called card is out and only one
non-picker seat is left, so that seat holds it.

## 2. Instrument

`python -m sheepshead.research.scan_called_suit_schmears` (`f8173bf`).
Self-play with the checkpoint in all five seats, argmax play (the
`/analyze` path, so every `seed` / `stepIndex` reproduces on the Analyze
page with Called Ace and Deterministic on). Each decision also records
the deployed probability of an A/10 play under the manifest's deploy
sampling (`DeploySampling.distribution`, the same code path as the app).
The deployed figure is exact at each recorded node; the nodes are the
ones argmax play reaches.

Population for the called-suit cells: a defender (not picker, not
holding the called card) void in the called suit, **holding no trump**,
with an A/10 fail and a cheaper fail in hand. Cell rate = argmax plays
the A/10. The control is the visible-loss pure-fail discard: the defender
is the last seat, the picker or revealed partner is winning, no legal
card wins, all legal cards are fail, holding an A/10 and a 0-point fail.

Runs: `runs/202609_recall_rc/schmear_probe/run.sh`: θ₀ (league 8M) to
θ₅ on called-ace seeds 0–19,999, θ₆ on 0–39,999, θ₆ JD on
100,000–119,999. Tables: `summary_theta6.txt`, `summary_theta6_jd.txt`,
`trend.txt`. About 4 min per 20,000 deals at 9 workers.

## 3. Incidence on θ₆ (40,000 called-ace deals)

Cells = argmax A/10 % ± SE (deployed %) n.

| cell | rate |
|---|---|
| **WRONG: no trump in, 0 defenders left** | **15.8 ± 2.4 (17.7) n 240** |
| unclear: no trump in, ≥ 1 defender left | 14.3 ± 1.0 (14.9) n 1209 |
| RIGHT: defender trump winning, 0 left | 78.6 ± 3.9 (78.1) n 112 |
| control: visible loss, pure-fail discard | 0.8 ± 0.1 (0.9) n 7272 |
| control, JD mode (20,000 deals) | 1.2 ± 0.2 (1.3) n 4945 |

- WRONG-cell spots occur 0.6 times per 100 called-ace deals. Argmax
  A/10 gifts: 0.095 per 100 deals. Mean points gifted per WRONG decision
  1.67.
- Holding a 0-point fail as well (197 of the 240): 17.8% (deployed
  17.8%).
- Cards: AC 13, AH 13, AS 6, 10S 4, 10C 1, 10H 1. Offsuit aces dominate.
- The probabilities are near-ties, not confident errors, e.g. seed 17175
  step 17 (trick 2, hand KC AS 9S 8S): AS 0.57, KC 0.19, 8S 0.13; seed
  2201 step 12: AC 0.35, 8S 0.28, 7C 0.18. More in
  `summary_theta6.txt`.

## 4. Operator hypothesis: a speculative schmear

Hypothesis: without trump, the agent has learned to risk a high-point
schmear on the chance that a later defender trumps, without checking that
a defender is left to do it.

Predictions: (a) the WRONG rate is no lower than the unclear rate; (b) a
true gamble rises with the number of defenders left to trump.

No trump in the trick, by defenders still to act (θ₆):

| | 0 left | 1 left | 2 left |
|---|---|---|---|
| all tricks | 15.8 ± 2.4 (n 240) | 17.3 ± 1.4 (n 694) | 10.3 ± 1.3 (n 515) |
| trick 1 | 14.8 ± 6.8 (27) | 25.8 ± 4.5 (93) | — |
| trick 2 | 30.2 ± 6.3 (53) | 22.9 ± 3.3 (166) | — |
| trick 3 | 20.0 ± 5.2 (60) | 16.6 ± 2.8 (175) | 14.4 ± 2.9 (146) |
| trick 4 | 4.2 ± 2.1 (95) | 8.9 ± 1.8 (247) | 7.9 ± 1.4 (366) |

- **(a) holds.** "No defender left" does not lower the rate, in total
  or within any trick. Deployed rates in trick 4 are 11.8 / 12.2 / 8.3%.
- **(b) fails.** The rate is flat or lower with more defenders left; the
  2-left cell is mostly tricks 3–4, and within those tricks it is flat.
- **The agent does read the trick:** with a defender's trump already
  winning it schmears 78.6%, against 15.8% with no trump in.

So the agent does not test "is anyone left who could trump". But the
behaviour is a roughly fixed 15–25% A/10 propensity on untrumped
called-suit tricks, insensitive to who is still to act, not a gamble
sized to the chance of a teammate's trump.

## 5. Across the search phase

Same 20,000 called-ace deals per checkpoint (θ₆: seeds < 20,000 of its
scan). θ₀ = `league/checkpoints/checkpoint_8000000.pt`, θₖ =
`pi/iterk/bidding/final.pt`.

| | WRONG (target 0) | unclear | RIGHT (schmear correct) | visible-loss control |
|---|---|---|---|---|
| θ₀ | 63.2 ± 4.0 (60.5) n 144 | 75.9 ± 1.7 | 88.9 ± 4.0 n 63 | 20.3 |
| θ₁ | 38.1 ± 4.3 (39.1) n 126 | 46.8 ± 2.0 | 82.5 ± 5.0 n 57 | 10.1 |
| θ₂ | 23.0 ± 3.6 (24.1) n 135 | 32.7 ± 1.9 | 75.0 ± 5.6 n 60 | 3.9 |
| θ₃ | 18.2 ± 3.4 (22.2) n 132 | 21.8 ± 1.7 | 76.9 ± 5.8 n 52 | 2.4 |
| θ₄ | 17.2 ± 3.1 (20.4) n 145 | 20.4 ± 1.6 | 71.4 ± 6.0 n 56 | 1.6 |
| θ₅ | 15.2 ± 3.4 (18.0) n 112 | 15.4 ± 1.5 | 67.3 ± 6.5 n 52 | 1.2 |
| θ₆ | 17.1 ± 3.6 (19.0) n 111 | 15.4 ± 1.5 | 78.6 ± 5.5 n 56 | 0.6 |

- **WRONG fell 63 → 17%, and has been flat since θ₃** (18.2 → 17.1,
  within noise).
- **The unclear cell fell with it and converged on it** (θ₀ 76 vs 63, a
  12-point gap in the right direction; θ₅–θ₆ 15.4 vs 15–17). The search
  lowered the schmear rate on untrumped called-suit tricks across the
  board. It did not teach the seat deduction, and θ₀'s weak
  discrimination is gone.
- **The RIGHT cell did not improve;** if anything it drifted down (θ₀
  88.9 ± 4.0, θ₅ 67.3 ± 6.5, θ₆ 78.6 ± 5.5). That fits a blanket
  "schmear less" push that also touched the cell where schmearing is
  right.
- **The visible-loss control was learned almost completely** (20.3 →
  0.6%), steadily through θ₆.
- **Contrast with the over-trump** (§9.19, frozen-state argmax): 40.9%
  → 22.0% from θ₀ to d6, still falling at the last iterate. Both cells
  improved and neither reached the optimum, but this one stopped at θ₃.
  The populations differ, so only the shapes compare.

## 6. Reading

The defender has every fact it needs on the table: the called card has
not appeared, the picker has played or must follow, and one non-picker
seat is left. That seat must hold the called card and must play it. The
agent plays as if that seat were unknown. The same blind spot explains
the flat dependence on defenders left (§4) and why the search's gains
were a general shift (§5).

Related but distinct: `Architecture_Ablation_202610.md` §4.4 probes
whether a seat knows that **it** is the secret partner (perceiver-recall
probe AUC 0.96 called / 1.00 JD, with the secret-partner aux head
carrying it). This spot needs the agent to place the partner in
**another** seat by elimination, a deduction no aux head is trained on.

## 7. Next steps

1. **Search read at the WRONG nodes (cheap, decides the lever).** Run the
   policy-iteration committee search at θ₆'s 240 WRONG nodes and read Q
   for the A/10 against the cheapest card. If search clearly prefers the
   cheap card, the signal exists and distillation is not absorbing it
   (the over-trump pattern, §9.4 / §9.19). If search is near-indifferent,
   the label itself is weak here, plausibly because few corpus nodes hit
   this spot.
2. **"Which seat is the partner" probe.** A linear probe on the encoder
   output for "the next seat to act holds the called card" at
   called-suit decisions with the partner hidden, by defenders left. If
   it reads poorly where the answer is forced (0 defenders left), the
   representation is missing; if it reads well, the policy head is not
   using it.
3. **Candidate validation-run design item** alongside the over-trump cell
   (`Training_Program_Redesign_202609.md` decision log 2026-11-02): the
   stop rule should not count an iteration flat while this cell has a
   known nonzero floor.

## 8. Correction made during the analysis

A first in-session pass, on a scratch scanner, put **every** trumped
called-suit trick in the RIGHT cell, including hands where the defender
still held trump. That reported 54% (n 689, θ₆) and suggested the agent
also held back as if the last seat could over-trump. 577 of those 689
hands held trump: a different decision, the over-trump cell's territory.
With the population held to no trump in hand, like the other cells, the
RIGHT rate is 78.6% (n 112). The "holds back in reverse" reading is
withdrawn. The WRONG and unclear cells were never affected; the scratch
scanner reproduces the module exactly there (600-seed check).

## 9. Reproducing

```
zsh runs/202609_recall_rc/schmear_probe/run.sh          # ~40 min, 9 workers
uv run python -m sheepshead.research.scan_called_suit_schmears \
    --summary runs/202609_recall_rc/schmear_probe/scan_theta6.jsonl
uv run python -m sheepshead.research.scan_called_suit_schmears --max-seed 20000 \
    --trend theta_0=runs/202609_recall_rc/schmear_probe/scan_theta0.jsonl ... \
            theta_6=runs/202609_recall_rc/schmear_probe/scan_theta6.jsonl
```
