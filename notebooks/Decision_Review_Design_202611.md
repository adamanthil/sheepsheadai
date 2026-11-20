# Decision Review — Design (November 2026)

**Status (2026-11-20): DESIGN SETTLED, NOT BUILT.** The three screens
are prototyped with real production data in `design/decision-review/`.
The review engine (`app/server/services/hand_review.py`) and an offline
report script (`app/scripts/review_hands.py`) exist and were used to
produce every number here. No migration, worker, API or web code exists
yet. Sections 6 and 7 are proposals; everything else was decided during
the design iteration.

## TL;DR — what this is

A signed-in player can review every decision they made in a past hand
against what the deployed model would have done in the same seat with
the same cards, and see what each difference cost or gained in expected
score. It is a learning aid first and a model-inspection tool second.

It lives in the account area as three screens:

* **Hand history** gains a column of marks, one per decision, and an
  expandable strip per hand.
* **Hand page** walks one hand decision by decision, with a chart of the
  model's running expected score.
* **Review tab** totals every reviewed hand: agreement by kind of
  decision, the largest differences, and picking.

Every hand is reviewed once, automatically, when it ends. Nothing a
player does triggers inference.

## 1. What a decision costs

* **D1 — Cost is a difference of two estimates.** Cost = the model's
  expected score after the player's choice, minus its expected score
  after the model's own first choice (argmax). A decision where the
  two agree, or where only one action was legal, costs nothing and is
  never evaluated.

* **D2 — Each branch is played to the seat's next decision on the
  actual deal.** From the decision, both branches are played forward
  with the model playing every other seat (deterministic), until the
  reviewed seat must act again. The critic's estimate is read there
  from that seat's view. If the hand ends first, the final score is
  used.
  * Not played to the end of the hand: that result depends mostly on
    how the remaining cards fall, not on the decision.
  * Not resampled over hidden cards: that approximates ISMCTS and is
    inaccurate at the iteration counts a small server can afford.
  * The other seats are model-played in both branches even when they
    were humans, because what a human would have played against a card
    that was never played is unknowable. A consequence: the "after your
    choice" estimate can differ from the next real point of the hand.

* **D3 — PICK is averaged over every possible blind.** A pick is read
  as the mean over all 325 blinds (26 unseen cards, choose 2), each at
  the decision after the pick. PASS stays on the actual deal. This
  removes the luck of the blind from the cost of picking: over the
  production hands the all-blinds read tracked the decision-time value
  with sd 0.42 against sd 1.46 for the actual blind.

* **D4 — The bury is one decision.** The two buried cards are chosen
  together and stored unordered, so they are reviewed as one composite
  action. The policy shown is over pairs (both orders summed); the
  model's choice is its sequential argmax pair. Reviewing the two cards
  separately produced about 28% false disagreements from order alone.

* **D5 — Expected score, not win probability.** Everything is shown in
  game points (critic value × 12). The win-probability head is not
  comparable across pick and pass: its sign matched the policy on 39%
  of pick disagreements against 88% for expected score.

* **D6 — Half a point is the threshold.** A difference under half a
  point of score (game points, not card points) is "about even". The
  threshold is applied to the cost as displayed, rounded to one
  decimal, so a shown −0.5 is never labelled even.

* **D7 — A doublers redeal scores zero.** When all five pass at a
  doublers table the hand is redealt at doubled stakes; the passing
  branch reads as 0. The redealt hand is reviewed on its own.

* **D8 — Hands the player did not fully play are skipped.** Old hands
  where the AI made the call or bury for a human seat are not reviewed.
  Only the player's own decisions are ever shown.

## 2. Language

* Soft and descriptive. No "blunder" or "mistake": the model is
  sometimes wrong, and a decision that beats its choice is shown as
  such, in green.
* The four states and their marks:

  | Mark | Meaning |
  |---|---|
  | ● | same choice as the model |
  | ○ | different, about even |
  | ▼ (red) | cost half a point or more |
  | ▲ (green) | beat the model's choice by half a point or more |

* "About even", not "close call" (which reads as a near miss).
* Tuned to Sheepshead, minimal model internals. The policy appears only
  as "how the model plays it" bars. Anything deeper is a link out to
  Analyze AI.

## 3. The screens

Prototypes: `design/decision-review/*.html`, opened straight from disk.
Screenshots are in `design/decision-review/screenshots/`. The data is
real: 160 hands of one production player, reviewed with
`sheepsheadai-1.1-202611`.

### 3.1 Hand history (`hand-history.html`)

* One new column, **Decisions vs. model · net score**: a mark per open
  decision in order, then the hand's net expected score against the
  model. The column has exactly one meaning; an earlier version that
  labelled the single worst or best decision was confusing.
* A legend above the table explains the marks.
* A row expands to a strip: the costliest decision as a headline, a
  sparkline of expected score from the player's seat, a cell per
  decision (action, cost, and what the model does), and a link to the
  hand page.
* No "costliest decisions" filter in the first version.
* A hand still waiting for review shows "Review pending" and does not
  expand. A hand that cannot be reviewed shows a dash. (Neither state
  is drawn in the prototype yet.)

### 3.2 Hand page (`hand-page.html`)

* A one-line headline naming the costliest decision, then role, result
  and a count of decisions by state.
* **The chart**, expected score before each of the player's decisions:
  * **From your seat** — heavy black line with a dot per decision, red
    where the decision cost. The limited-information critic.
  * **With every card known** — solid light grey, 2px. The oracle
    critic on the same states. It reacts at once to a play whose cost
    the seat cannot yet see.
  * **Had you made the model's choice** — a dotted grey line from the
    decision to a hollow ring one step later, drawn only where the
    decision cost or gained half a point. A short vertical bar joins
    the ring to the real line, red or green, labelled with the cost.
    The bar's length is the cost, so the chart shows the measurement.
  * Line style carries meaning: solid lines are the hand as played,
    the dotted line is hypothetical.
  * **Ring anchor.** The ring sits at the real line's next point
    shifted by the cost, not at the model branch's raw estimate. The
    two differ when the other seats did not play as the model would
    (D2); anchoring to the real line keeps the bar equal to the cost
    shown.
* A list of decisions with costs; selecting one fills the detail panel:
  the hand at that moment, the model's policy as bars with the player's
  choice tagged, both expected scores, the cost, and a sentence on how
  it was measured.
* No "what would have happened" branch panel. It was prototyped and
  dropped as not worth the complexity.

### 3.3 Review tab (`review-tab.html`)

Pure queries over stored reviews.

* **You and the model**: share of open decisions with the same choice,
  how many cost half a point or more, how many beat the model's choice,
  and the net score against the model (total and per hand).
* **Same choice, by kind of decision**: pick or pass, partner call,
  bury, card play, each with its rate, count and net score.
* **Largest differences**: a list switchable between "cost you most"
  and "beat the model's choice", filterable by kind.
* **Picking**: pick agreement, and a card per hand where the player
  picked and the model passes, or the reverse, with the hand shown.

There is no coverage meter and no "review more" control, because
review is automatic.

## 4. What the production data showed

388 hands at the time of the dump (soft launch); 160 scored hands for
the most active player.

* Players make the model's choice on about 67% of open decisions, and
  the median cost of a difference is about zero.
* For that player: pick or pass 94% (net +0.8), partner call 49%
  (+0.3), bury 60% (−4.0), card play 65% (−31.3). Card play is where
  the points go.
* 49 decisions cost half a point or more, 28 beat the model's choice,
  162 were about even. Net −34.9, or −0.22 a hand.
* **The net is dominated by a few decisions.** One lead (a picker
  playing alone led the 8♦ instead of the J♦ at trick 5: −8 against +4
  when played out) is −11.9, a third of the total. The total is a plain
  sum; a per-hand median or a cap per decision are the alternatives if
  it reads too much like a skill rating.
* **Replay fidelity.** Reviewing AI seats with their own model
  reproduces the bury 100%, the call 99–100%, the pick 99.6%, and card
  play about 90% (those seats sample; the review takes the argmax).
* **The seat-view line lags.** On the showcase hand the trick 4 play
  moved the seat estimate from +3.0 to +1.7 and the oracle from +1.9 to
  −1.8. Played to the end it was a six-point swing; the next-decision
  cost is −2.0. This is the reason for showing the oracle line. Using
  the oracle for the cost itself was raised and not pursued.

## 5. Cost of running it

On an M1 Max CPU: 1.9–2.4 ms per evaluation, about 0.20 s per reviewed
hand (roughly 100 evaluations, most of them the 325-blind pick), 0.09 s
for a hand with no pick to average. The production box has not been
timed, and contention with live inference in the same process has not
been measured.

## 6. Runtime (proposed)

* **One code path.** `review_hand` in `hand_review.py` is the only
  implementation; the worker and the offline script both call it. The
  live table's agent state is not reused: review needs both branches of
  each decision, which live play never computes.
* **A queue in Postgres.** A row per human seat is inserted when a hand
  ends; a migration inserts rows for existing hands.
* **One background worker**, newest hand first, one hand at a time. It
  takes the server's inference gate for each single evaluation, so a
  live move waits behind at most one review evaluation. A hand that
  fails to replay is marked failed after a couple of attempts.
* **Players cannot trigger review.** This rules out one account forcing
  hundreds of analyses onto the server.

## 7. Storage (sketch, to be revised)

Two tables: one row per reviewed seat (queue state, model, summary
counts for the history column), and one row per decision (kind, trick,
action, the model's action, policy, forced flag, the estimate at the
decision, and both branch estimates). The sketch predates several
decisions above and needs, at least: the composite bury action, a
score-first summary (marks and net), the hand and table snapshot the
detail panel shows, the three pick readouts (all blinds, actual blind,
pass), and the oracle estimate per decision.

Build order: (1) migration; (2) worker with tests against recorded
hands; (3) API — review fields on history rows, a per-hand endpoint,
review-tab aggregates; (4) history column and strip; (5) hand page;
(6) review tab.

## 8. Open questions

* Should costs be multiplied by the stake on doubled hands?
* Should AI-made calls and buries be flagged individually from now on,
  so those hands are partly reviewable instead of skipped?
* Is the plain sum the right net score (section 4)?
* Timing on the production box, and contention under live load.

## 9. Files

* `app/server/services/hand_review.py` — replay and review, no
  database access.
* `app/scripts/review_hands.py` — read-only report over a database:
  `uv run python app/scripts/review_hands.py`.
* `design/decision-review/` — the three prototypes,
  `prototype-runtime.js` (a small standalone renderer so they open
  from disk), and screenshots.
