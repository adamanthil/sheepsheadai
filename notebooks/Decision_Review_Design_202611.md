# Decision Review — Design (November 2026)

**Status (2026-11-24): BUILT** on branch `decision-review` (see "As
built" at the end). What follows is the design as it was settled.

**Status before the build (2026-11-24): DESIGN SETTLED, NOT BUILT.** The three screens
are prototyped with real production data in `design/decision-review/`.
The review engine (`app/server/services/hand_review.py`) and an offline
report script (`app/scripts/review_hands.py`) exist and were used to
produce every number here. No migration, worker, API or web code exists
yet. Section 7 is a sketch; everything else was decided during the
design iteration (revised 11-24: history layout, wording, queue rules).

## TL;DR — what this is

A signed-in player can review every decision they made in a past hand
against what the deployed model would have done in the same seat with
the same cards, and see what each difference cost or gained in expected
score. It is a learning aid first and a model-inspection tool second.

It lives in the account area as three screens:

* **Hand history** gains an "AI comparison" view: per hand, what the
  player's choices lost and gained against the AI model, and the net.
* **Hand page** walks one hand decision by decision, with a chart of the
  AI model's running expected score.
* **AI review tab** totals every reviewed hand: agreement by kind of
  decision, the largest differences, and picking.

Hands played by registered accounts are reviewed once, automatically,
in the background. Nothing a player does triggers inference.

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

* **D6 (superseded by section 11: three bands at 0.25 and 1.0, which is
  what was built).** As first settled: **half a point is the
  threshold.** A difference under half a point of score (game points,
  not card points) is "about even". The threshold is applied to the cost
  as displayed, rounded to one decimal, so a shown −0.5 is never
  labelled even.

* **D7 — A doublers redeal scores zero.** When all five pass at a
  doublers table the hand is redealt at doubled stakes; the passing
  branch reads as 0. The redealt hand is reviewed on its own.

* **D8 — Hands the player did not fully play are skipped.** Old hands
  where the AI made the call or bury for a human seat are not reviewed.
  Only the player's own decisions are ever shown.

## 2. Language

* **"AI", not "model".** In everything a player reads. "AI model" is
  kept only in the two sentences that introduce the feature: the line
  beside the history toggle and the lede of the AI review tab.
* **Beta.** A small "Beta" tag sits at each entry point: the history
  toggle, the account tab, and the hand page's kicker.
* Soft and descriptive. No "blunder" or "mistake": the AI model is
  sometimes wrong, and a choice that beats its choice is shown as such,
  in green.
* One vocabulary on every screen: **agreed**, **about even**,
  **cost** / **cost you**, **gained**. Not "open decisions", not
  "close call" (which reads as a near miss), and not "lost" for
  expected score, which is kept for the hand's result.
* A signed number is never headed "Cost", since it can be a gain: the
  hand page's column is "vs. AI" and its figure is "Difference".
* Tuned to Sheepshead, minimal internals. The policy appears only as
  "how the AI plays it" bars. Anything deeper is a link out to
  Analyze AI.

## 3. The screens

Prototypes: `design/decision-review/*.html`, opened straight from disk.
Screenshots are in `design/decision-review/screenshots/`. The data is
real: 160 hands of one production player, reviewed with
`sheepsheadai-1.1-202611`.

### 3.1 Hand history (`hand-history.html`)

* **A toggle at the top: Results | AI comparison (beta).** Results is
  listed first; AI comparison is the default, for discoverability. Results is the table
  as it is today, unchanged (table, stakes, points, opponents, notes).
* **AI comparison columns:** Date, Role, Score, a Cost/Gained bar, Net.
  Table name and points are dropped here to give the bar room.
* **The bar.** From a centre line, a red bar grows left by the total
  expected score the player's choices cost in that hand, and a green
  bar grows right by the total they gained. The column heads "Cost" and
  "Gained" sit either side of the centre and are the legend. Every
  difference counts, including the about-even ones, so the two bars
  always add up to Net.
* **Scale.** Three points fills a side, the same for every row. A
  longer bar is cut at full width and notched.
* **One line of explanation** beside the toggle; no separate legend.
* A row expands to a strip: the costliest decision as a headline, a
  sparkline of expected score from the player's seat, a cell per
  decision (action, difference, and what the AI does), and a link to
  the hand page.
* A hand still in the queue shows "Waiting for review" in place of the
  bar and does not expand. A hand that cannot be reviewed shows a dash.
* Dates never wrap, in either view, as on the live page.
* **Phone.** The role moves under the date, the bar narrows, and
  nothing scrolls sideways. The expanded strip keeps the headline, the
  link and the sparkline with its dots, and drops the per-decision
  cells, which are too much for the width. The Results view scrolls
  sideways, as the live table does.
* No "costliest decisions" filter in the first version.
* An earlier version showed a mark per decision (● ○ ▼ ▲). It was too
  detailed to read at a glance and was replaced by the bar.

### 3.2 Hand page (`hand-page.html`)

* A one-line headline naming the costliest decision, then role, result,
  the hand's net against the AI with its cost and gained totals (the
  same three figures as the history row), and a count of decisions by
  state.
* **The chart**, expected score before each of the player's decisions:
  * **From your seat** — heavy black line with a dot per decision, red
    where the decision cost. The limited-information critic.
  * **With every card known** — solid light grey, 2px. The oracle
    critic on the same states. It reacts at once to a play whose cost
    the seat cannot yet see.
  * **Had you made the AI's choice** — a dotted grey line from
    the decision to a hollow ring one step later, drawn only where the
    decision cost or gained half a point. A short vertical bar joins
    the ring to the real line, red or green, labelled with the
    difference. The bar's length is the difference, so the chart shows
    the measurement.
  * Line style carries meaning: solid lines are the hand as played,
    the dotted line is hypothetical.
  * **Ring anchor.** The ring sits at the real line's next point
    shifted by the difference, not at the AI branch's raw estimate.
    The two differ when the other seats did not play as the AI model
    would (D2); anchoring to the real line keeps the bar equal to the
    figure shown.
* A list of decisions; selecting one fills the detail panel: the hand
  at that moment, the AI model's policy as bars with the player's
  choice tagged, both expected scores, the difference, and a sentence
  on how it was measured.
* **High trump**, in the detail under the hand: the highest trump not
  yet played at that decision, whether it is "in your hand" or "still
  out", and how many trump the player has not seen. Shown at every
  decision; "no trump left" once they are all played.
  * It is bookkeeping, not inference: the trump order less the cards
    played so far, counting the player's own bury as gone only when
    they are the picker.
  * Not stored. It is derived from the recorded plays when the hand
    page is loaded, so showing more later (the full list of trump
    still out) needs no migration or re-review.
* **At the table**, last and secondary, one line: the human players by
  name with seat and role, then a count of the AI players ("Andrew
  (you) · seat 3 · Picker, and 4 AI players"). AI players are not
  named. This is where the table context dropped from the AI
  comparison view is recovered.
* The page opens with the headline decision selected.
* **The headline depends on the hand.** Over 400 reviewed seat-hands
  in production, only 23% have a decision that cost half a point, so
  the headline needs four forms:

  | Hand | Share | Headline |
  |---|---|---|
  | has a costly decision | 23% | that decision: "You played the J♣. The AI plays the K♣." |
  | no cost, but a gain of half a point | 10% | the best decision, framed as a gain |
  | only about-even differences | 43% | a plain statement, e.g. "Three different choices, all about even." |
  | agreed throughout | 24% | "You and the AI agreed on every decision." |

  Where there is a costly decision, one decision is a fair summary: it
  is the only costly one in 76% of those hands, and in the median hand
  it is all of the hand's cost. 88% of headline decisions are card
  plays. The history strip's headline follows the same four forms.
* **Phone.** The detail opens in place, directly under the tapped
  decision, so the hand and the comparison are visible without
  scrolling. Tapping the open decision closes it; tapping another
  moves the detail there; Previous and Next step through. Each row
  carries a small caret. The side panel is not shown. The hand's cards
  shrink to stay on one row.
* The cost labels on the chart can crowd the lines (the −0.7 at the
  bury does); to be tuned against more hands during the build.
* No "what would have happened" branch panel. It was prototyped and
  dropped as not worth the complexity.

### 3.3 AI review tab (`review-tab.html`)

Pure queries over stored reviews. The account tab is "AI review", with
the beta tag.

* **You and the AI**: share of decisions where they agreed,
  how many cost half a point or more, how many beat its choice, and the
  net against the AI model (total and per hand).
* **Agreed, by kind of decision**: pick or pass, partner call,
  bury, card play, each with its rate, count and net.
* **Largest differences**: a list switchable between "cost you most"
  and "beat the AI's choice", filterable by kind. Each row is a link,
  marked with an arrow, that opens the hand page on that decision.
* **Picking**: pick agreement, and a card per hand where the player
  picked and the AI model passes, or the reverse, with the hand shown.

There is no coverage meter and no "review more" control, because
review is automatic.

### 3.4 Admin (`admin.html`)

A sketch, with placeholder numbers. An "Admin" account tab, shown only
to administrators, with one section for now:

* **AI review queue**: running or paused (and why), hands waiting, age
  of the oldest waiting hand, hands reviewed in the last 24 hours,
  average time per hand; then totals, failures, the slowest recent
  hand, the last hand reviewed, the AI version in use, and how many
  stored reviews came from an older version.

Later additions, not designed: hands played in the last day, and how
players did against the AI.

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

## 6. Runtime

* **One code path.** `review_hand` in `hand_review.py` is the only
  implementation; the worker and the offline script both call it. The
  live table's agent state is not reused: review needs both branches of
  each decision, which live play never computes.
* **Registered accounts only.** A seat is queued only if its player has
  an account. At the time of writing that is 882 of 1482 hands.
  * At hand end: a queue row for each seat whose player has an account.
  * At registration, before the email is verified: a queue row for
    every hand that player has already played. An account attaches to
    the existing player row, so the guest history comes with it.
  * The migration queues the existing hands of existing accounts.
* **A queue in Postgres**, one background worker, newest hand first,
  one hand at a time.
* **Live play comes first.** The worker runs only while at most one
  table is being played, and pauses when there are more. It also takes
  the server's inference gate for each single evaluation, so a live
  move waits behind at most one review evaluation.
  * The pause is checked between hands at least; a hand takes a
    fraction of a second, so checking between evaluations is not
    needed unless measurement says otherwise.
  * If the site is ever busy around the clock the queue will not
    drain; the admin page (3.4) is where that shows.
* **"In play"** means a table with a hand in progress and a human
  connected.
* **Administrators.** `account.is_admin BOOLEAN NOT NULL DEFAULT
  false`, added just before or with this work. The migration sets
  nobody; it is set by hand. It gates the admin tab and its endpoint.
  The seat row records when its review started and how long it took,
  which is what the admin figures are computed from.
* **A hand that fails to replay** is marked failed after a couple of
  attempts and shown as not reviewable.
* **Players cannot trigger review.** This rules out one account forcing
  hundreds of analyses onto the server.
* **Model changes do not re-review.** Each review records the model
  that produced it. A new deployed model reviews new hands only; old
  reviews stay as they are. Re-reviewing is an operator action
  (re-queue everything, or everything from one model), run through the
  same queue and so under the same limits.

## 7. Storage (sketch, to be revised)

Two tables: one row per reviewed seat (queue state, summary figures for
the history view), and one row per decision (kind, trick, action, the
AI model's action, policy, forced flag, the estimate at the decision,
and both branch estimates).

* **Both tables carry `ai_model_id`** as a foreign key to `ai_model`,
  so every stored figure names the version that produced it. The
  decision rows reference the seat row on (seat, model) together, so
  the two can never disagree. One review per seat is current; a
  re-review replaces it.
* The seat row needs: lost and gained totals (the history bar), the
  count of decisions by state, and the costliest decision.
* The decision row needs, beyond the first sketch: the composite bury
  action, the hand and table snapshot the detail panel shows, the three
  pick readouts (all blinds, actual blind, pass), and the oracle
  estimate.

Build order: (1) migration; (2) worker with tests against recorded
hands; (3) API — review fields on history rows, a per-hand endpoint,
review-tab aggregates; (0) `is_admin`; (4) history toggle, bar column and strip; (5) hand page;
(6) review tab.

## 8. Open questions

* Should differences be multiplied by the stake on doubled hands?
* Should AI-made calls and buries be flagged individually from now on,
  so those hands are partly reviewable instead of skipped? (Hands the
  history already badges "AI-assisted" are reviewed for the plays the
  player made; substituted plays are skipped.)
* Is the plain sum the right net (section 4)?
* Timing on the production box.

## 9. Files

* `app/server/services/hand_review.py` — replay and review, no
  database access.
* `app/scripts/review_hands.py` — read-only report over a database:
  `uv run python app/scripts/review_hands.py`.
* `design/decision-review/` — the three prototypes,
  `prototype-runtime.js` (a small standalone renderer so they open
  from disk), the admin sketch, and desktop and phone screenshots.

## 10. As built

Where the build differs from, or settles, the design above.

* **Storage.** `hand_review` is the queue and the per-seat summary;
  `decision_review` holds the decisions (migration 000010). Card codes
  are stored as text and the policy as JSON. `decision_review.difference`
  is a generated column, so every query agrees on the rounding.
  `account.is_admin` is migration 000009.
* **Nothing more is stored than section 7 lists**, less the hand and
  table snapshot, which is two columns (`hand`, `table_cards`), and the
  three pick readouts, which reduce to the two expected scores already
  kept (`chosen_score`, `model_score`).
* **The high trump is derived when the hand page loads**
  (`services/high_trump.py`), from the recorded tricks and each stored
  decision's hand.
* **Differences are not multiplied by the stake** on doubled hands
  (first open question): they are in the AI's expected score, which
  knows nothing of the doublers multiplier.
* **AI-made calls and buries are still not flagged individually**; a
  hand with one is marked `skipped`.
* **The worker** (`runtime/review_worker.py`) checks for more than one
  table in play before each hand, and takes the inference gate for each
  forward pass. A record the engine cannot follow fails at once; any
  other failure is retried once. The reason is stored and shown on the
  admin page. `AI_REVIEW_ENABLED=false` switches it off.
* **History refreshes itself.** While a hand on show is waiting, the
  page looks again every 20 seconds.
* **A decision's option list is capped**: the choices the AI gives a
  real chance, the player's own, and a count of the rest. A bury has 28
  legal pairs.
* **Links name a decision.** `/account/history/<game>?d=<index>` opens
  the hand page on that decision; the AI review tab's lists use it.
* **Measured on this machine while it was heavily loaded** (load
  average over 200): 1.1 s a hand on average and 7.4 s for the slowest,
  against 0.2 s when idle. The production box is still untimed.
* **Operator actions** (make an administrator, re-review, switch the
  worker off) are in `docs/deploy.md`.

## 11. Where the bands sit (measured 2026-11-24)

D6 set one threshold, half a point, by judgement. It was then tested
against what hands really did, and replaced by three bands.

**The test.** On a production snapshot of 1,629 hands, every decision
where a player and the AI differed (2,837 of them) was played out to the
end of the hand both ways, on the actual deal, with the AI playing every
seat from that point. That gives a real final score for each choice, to
set beside the review's predicted difference.

Card play (2,454 decisions):

| Predicted difference | n | Final score changed | As predicted | Other way | Mean predicted | Mean real |
|---|---|---|---|---|---|---|
| under 0.10 | 831 | 6% | 3% | 3% | 0.04 | −0.01 |
| 0.10 to 0.25 | 519 | 13% | 8% | 5% | 0.17 | 0.01 |
| 0.25 to 0.50 | 410 | 25% | 18% | 7% | 0.36 | 0.26 |
| 0.50 to 0.75 | 218 | 26% | 21% | 5% | 0.62 | 0.58 |
| 0.75 to 1.00 | 139 | 37% | 27% | 11% | 0.86 | 0.54 |
| 1.00 to 1.50 | 153 | 42% | 38% | 5% | 1.24 | 1.06 |
| 1.50 to 2.50 | 127 | 39% | 35% | 4% | 1.95 | 1.04 |
| 2.50 and up | 57 | 53% | 47% | 5% | 3.89 | 2.35 |

("Mean real" counts the real difference as positive when it went the
way the review predicted.)

* **Under a quarter point really is even**: the score moves about one
  time in ten and as often against the prediction as with it.
* **From a quarter point to a point the difference is real but small**:
  the score moves a quarter to a third of the time, three or four to
  one the predicted way, and the mean real difference matches the
  predicted one.
* **There is a step at one point**: the score moves about four times in
  ten and wrong-way results halve. When it moves it goes the predicted
  way 90% of the time, against 84% at half a point.
* "More likely than not to change the final score" is never reached on
  one deal short of 2.5 points, because scores move in whole points.
* Above about 1.5 the review overstates the size (real about half to
  two-thirds of predicted); the direction stays reliable.
* Pick, call and bury (383 decisions) are much noisier and too few above
  one point to set bands of their own; they use the same ones.

**The bands** (`EVEN_BELOW`, `FLAG_FROM` in `services/hand_review.py`):

| Difference | Standing | Shown as |
|---|---|---|
| under 0.25 | even | "about even", no colour |
| 0.25 to under 1.0 | slight | the number and the AI's choice, no colour; "a small difference" |
| 1.0 and up | costly / better | red or green; eligible for the headline; drawn on the chart |

On that snapshot the share of hands whose headline names a decision
falls from about a third (23% costly, 11% gain only) to about a sixth
(12% and 6%). A hand with nothing over a point says so: "Three different
choices, none worth a full point." The cost and gained totals and the
net still count every difference.

The collection and analysis scripts were one-off and are not in the
repository.

