"""Stage B: single-observer ISMCTS soft-teacher engine.

A search-derived, per-state, outcome-grounded teacher that produces a soft target
``pi'(a) ∝ N(a)^(1/tau_target)`` over the training agent's information set at any
decision point (pick / partner / bury / play). It is the principled replacement
for the hand-crafted exploration/shaping controllers (see
``notebooks/ISMCTS_Teacher_Refactor_Plan.md``).

Design (locked decisions, §3 of the plan)
-----------------------------------------
* **Algorithm:** SO-ISMCTS, statistics-only nodes keyed by the *observer's*
  action sequence. The recurrent memory is re-derived every iteration by encoding
  along the descended path in that iteration's determinized world (the Stage-A
  forced-replay mechanism, extended forward through the in-tree descent and the
  rollout).
* **Determinization (scheme B):** each iteration samples one hidden-card world
  with ``Game.sample_determinization`` (honours per-seat counts, play-revealed
  voids, forced plays, called-ace placement). Worlds are *self-normalized
  importance-weighted by the bidding likelihood only* (pick / pass / call /
  alone). Plays enter as hard void constraints inside the determinizer, never as
  soft weights. The observer's own bidding actions cancel in self-normalization
  (its hand is fixed), so the weight is dominated by P(picker would pick this
  determinized hand) — exactly the inference that corrects the trick-0 bias.
* **Selection:** PUCT with the network prior ``P(a)``. Availability-count PUCT at
  non-lead (follow-suit) play nodes — where the legal set varies across worlds —
  and plain PUCT everywhere else (the observer's own decision set is fixed).
* **Leaf evaluation:** truncated rollout of ``d_rollout`` further observer *play*
  plies, then a ``value_trunk`` V-bootstrap; a world that ends first contributes
  the observer's terminal score discounted on the same observer-action clock as
  PPO. Values are in the critic's units (game score / 12, i.e. ~[-1, 1]), so the
  AlphaZero ``c_puct = 1.25`` is calibrated without extra Q-normalization.
* **Seat policies (population grounding):** ``search(..., seat_policies={seat:
  PPOAgent})`` models each NON-observer seat — in the forced-replay pool build
  (including the scheme-B bidding belief weights), the in-tree advance phase,
  and rollouts — with the given controller, normally the agent actually sitting
  there in the live training game. The observer's own decisions, rollout plies,
  priors and critic bootstrap always use ``self.agent`` (self-modeling your own
  future is correct, and Q/V stay in the training agent's units). ``None``
  (default) reproduces pure self-play exactly. Rationale: a self-modeled
  rollout field cannot punish information-revealing play, so the teacher
  certifies leaks instead of correcting them (see the teacher trump-lead audit
  / ``notebooks/Population_Grounded_Teacher_Plan.md``).
* **Heads via one engine:** pick / partner / bury are *shallow* roots (``max_depth
  = 1``) and degenerate to bidding-weighted determinized rollout evaluation of
  each option; play is the deep tree (``max_depth = 6`` observer decisions).
* **Output:** ``pi'(a) ∝ N(a)^(1/tau_target)`` over the (weighted) root visit
  counts, plus the root ESS. Below ``ess_floor`` the caller skips the target
  (``ok = False``); the transition still trains via plain PG.

The engine is *side-effect free* on the agent's per-seat recurrent memory: it
snapshots and restores ``agent._player_memories`` around each search so the
acting policy is undisturbed. Search is **training-time only**; the shipped
network never searches.
"""

from __future__ import annotations

import copy
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import NamedTuple, TypedDict

import numpy as np
import torch

from sheepshead import (
    ACTION_IDS,
    ACTIONS,
    Game,
)
from sheepshead.agent import ppo
from sheepshead.training.training_utils import RETURN_SCALE

DEV = ppo.device

# Per-head iteration budgets and tree depths (plan §3).
_DEFAULT_ITERS = {"pick": 48, "partner": 64, "bury": 96, "play": 96}
_DEFAULT_DEPTH = {"pick": 1, "partner": 1, "bury": 1, "play": 6}

# Rollout depths above this never reach a critic bootstrap (the observer plays
# at most 6 plies in a hand), so oracle leaf capture would be pure overhead and
# is disarmed. Any "terminal-depth" sentinel (e.g. d_rollout=99) exceeds it.
ORACLE_LEAF_MAX_BOOTSTRAP_DEPTH = 12

# Iteration cap on a forced replay's outer loop (both executors); exceeding it
# means the replay is not converging on the public record.
_REPLAY_GUARD_LIMIT = 6000

# ``teacher.fail`` counter keys for replay/determinization failures.
FAIL_GUARD = "guard"
FAIL_PUB_DESYNC = "pub_desync"
FAIL_BAD_PUBLIC = "bad_public"
FAIL_BAD_PRIVATE = "bad_private"
FAIL_HIST_MISMATCH = "hist_mismatch"
FAIL_NO_ACTED = "no_acted"
FAIL_DETERMINIZE = "determinize"
FAIL_BATCHED_FALLBACK = "batched_fallback"


def _is_play_action(action_id: int) -> bool:
    return ACTIONS[action_id - 1].startswith("PLAY ")


def _is_weighted_bidding_action(action_id: int) -> bool:
    """Scheme B: only public BIDDING actions (pick / pass / call / alone /
    jd-partner) carry belief weight during the forced replay; plays are forced
    but never weighted, private bury/under never weighted."""
    return not _is_play_action(action_id)


def _is_private_action(action_id: int) -> bool:
    name = ACTIONS[action_id - 1]
    return name.startswith("BURY ") or name.startswith("UNDER ")


def _valid_has_play(valid) -> bool:
    return any(_is_play_action(action_id) for action_id in valid)


def _pool_norm_weights(pool) -> np.ndarray:
    """Unnormalized world importance weights ``exp(log_w - max(log_w))``
    (float64), the shared first step of the pool probability and ESS math."""
    log_weights = np.array([log_weight for _, _, log_weight in pool], dtype=np.float64)
    return np.exp(log_weights - log_weights.max())


def _draw_from_cumulative(rng, valid, prob_of) -> int:
    """One ``rng.random()`` draw walked down the cumulative distribution
    ``prob_of`` over ``valid`` (in order); numeric slack falls to the last
    action. Exactly one RNG consumption — part of the pinned draw order."""
    draw = rng.random()
    cum = 0.0
    for action_id in valid:
        cum += prob_of(action_id)
        if draw <= cum:
            return action_id
    return valid[-1]


def _minmax_unit(values: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]; a degenerate span maps everything to 0.5.
    Root-local Q normalization for the RM update and the gumbel readout (the
    PUCT selection loop keeps its own tree-global normalization)."""
    lo, hi = float(values.min()), float(values.max())
    if hi > lo:
        return (values - lo) / (hi - lo)
    return np.full_like(values, 0.5)


def _private_root_ready(real_game, world, valid) -> bool:
    if not any(_is_private_action(action_id) for action_id in valid):
        return True
    return (
        list(world.bury) == list(real_game.bury)
        and world.under_card == real_game.under_card
    )


def _at_replay_root(real_game, world, public_actions, seat, observer, valid) -> bool:
    """The forced replay has reached the live root: the public record is
    exhausted, it is the observer's turn, and (for later private roots) the
    world's bury/under progress matches the live game's."""
    return (
        not public_actions
        and seat == observer
        and _private_root_ready(real_game, world, valid)
    )


@dataclass(frozen=True)
class _PrivateDecision:
    """Director event: ``seat`` faces a forced bury/under decision. Each world
    resolves its OWN card from its determinization; never belief-weighted."""

    seat: int


@dataclass(frozen=True)
class _PublicAction:
    """Director event: ``seat`` must take the recorded public ``action_id``
    (identical across worlds). ``weighted`` marks scheme-B bidding actions
    whose policy log-prob enters the world's belief weight."""

    seat: int
    action_id: int
    weighted: bool


def _replay_events(real_game, ref_world, forced_public, observer):
    """The replay DIRECTOR: yield the forced-replay decision structure derived
    from ``ref_world``, one event per action, until the live root is reached
    (generator return). Both world-build executors run this control flow —
    ``_build_world`` drives it per world, ``_build_worlds_lockstep`` drives one
    instance off world 0 and applies each event to every world.

    Contract: the director only READS ``ref_world``; the consumer must apply
    each yielded event to the reference world (and any lockstepped worlds)
    before advancing the generator — the next event is derived from the
    post-action reference state. Owns a fresh copy of the public record;
    structural failures (guard overflow, public-record desync, no seat acted)
    raise ``_ReplayInconsistency`` with the matching ``FAIL_*`` key. Per-world
    failures (a recorded/forced action illegal in one world) are the
    EXECUTORS' to detect, because worlds may diverge from the reference in
    private cards — but never in public flow."""
    public_actions = deque(forced_public)
    guard = 0
    while True:
        guard += 1
        if guard > _REPLAY_GUARD_LIMIT:
            raise _ReplayInconsistency(FAIL_GUARD, "forced replay guard exceeded")
        acted = False
        for seat in range(1, 6):
            player = ref_world.players[seat - 1]
            valid = player.get_valid_action_ids()
            while valid:
                # Root reached: public record exhausted and it is the
                # observer's turn. If this is a later private decision, the
                # private events already forced bury/under progress to match
                # the live game.
                if _at_replay_root(
                    real_game, ref_world, public_actions, seat, observer, valid
                ):
                    return
                if any(_is_private_action(action_id) for action_id in valid):
                    yield _PrivateDecision(seat)
                else:
                    if not public_actions or public_actions[0][0] != seat:
                        raise _ReplayInconsistency(
                            FAIL_PUB_DESYNC, "forced replay: public action desync"
                        )
                    _, action_id = public_actions.popleft()
                    yield _PublicAction(
                        seat, action_id, _is_weighted_bidding_action(action_id)
                    )
                acted = True
                valid = player.get_valid_action_ids()
        if not acted:
            raise _ReplayInconsistency(FAIL_NO_ACTED, "forced replay: no seat acted")


class SearchResult(TypedDict):
    """The contract of ``ISMCTSTeacher.search``. Every key is always present
    (pinned by ``test_search_output_contract``); consumers index it directly."""

    pi: np.ndarray  # float32[action_size]; visit-count target, tau-sharpened.
    #   All-zero when no simulation completed (then ok=False).
    ess: float  # Effective sample size of the belief-weighted world pool.
    ok: bool  # ess >= config.ess_floor AND root statistics exist.
    head: str  # "pick" | "partner" | "bury" | "play".
    n_iter: int  # Worlds successfully built (NOT the iteration count).
    valid: list[int]  # Sorted legal action ids at the root.
    root_n: dict[int, float]  # Per-action weighted visit count.
    root_q: dict[int, float]  # Per-action mean value (critic units).
    root_prior: dict[int, float] | None  # Mean UNMIXED network prior;
    #   None until the root has been expanded at least once.
    pi_gumbel: np.ndarray | None  # Completed-Q readout; None when unavailable.
    pi_rm: np.ndarray | None  # RM+ average strategy; None unless
    #   root_selection == "rm" produced statistics.


class _ReplayInconsistency(Exception):
    """A determinized world could not be forced-replayed against the public record
    (a recorded action is illegal in that world, or the lockstep desynced). Rare:
    ``sample_determinization`` is consistent by construction, but void inference is
    not exhaustive, so an occasional redeal makes a recorded play illegal. The
    batched lockstep cannot skip one world mid-flight, so it raises this and the
    caller falls back to the per-world sequential build, which drops bad worlds.

    ``key`` is the ``teacher.fail`` counter this failure maps to (a ``FAIL_*``
    constant)."""

    def __init__(self, key: str = "replay", message: str = ""):
        super().__init__(message or key)
        self.key = key


@dataclass
class ISMCTSConfig:
    c_puct: float = 1.25
    d_rollout: int = 2
    tau_target: float = 1.0
    ess_floor: float = 4.0
    det_max_tries: int = 2000
    # Deterministic uniform mix into the ROOT prior only:
    # P_root(a) = (1 - f) * P(a) + f / n_legal. Guarantees the search explores
    # *every* root option even when the (possibly collapsed / leaked) policy
    # assigns one near-zero prior — otherwise PUCT starves the low-prior action
    # and the visit-count target just reproduces the policy it is meant to
    # correct. Deterministic (not Dirichlet) so the distillation target is
    # stable across calls. Applied at the root because that is what the soft
    # target is read from; deeper nodes keep the raw network prior.
    root_explore_frac: float = 0.25
    # First-play urgency: the value assigned to a not-yet-tried action, in the
    # min-max-NORMALIZED Q space ([0, 1]). 1.0 = optimistic, so every legal
    # action is tried before any is revisited — essential when the policy is
    # collapsed onto one action (a near-zero-prior alternative would otherwise
    # never be explored, and the visit-count target would just echo the leaked
    # policy it is meant to correct).
    fpu: float = 1.0
    iters: dict = field(default_factory=lambda: dict(_DEFAULT_ITERS))
    max_depth: dict = field(default_factory=lambda: dict(_DEFAULT_DEPTH))
    # Leaf-parallel batching: run ``batch_size`` simulations concurrently so the
    # transformer encoder runs on a batch of states per round instead of batch-1
    # per ply (the dominant search cost; see throughput profiling). ``virtual_loss``
    # is the pessimistic value (in critic units, ~[-1, 1]) charged to an in-flight
    # selected edge so concurrent sims in a chunk diversify instead of all taking
    # the PUCT-best path.
    batch_size: int = 32
    virtual_loss: float = 1.0
    # --- Root readout / selection experiments (Search_Readout_Comparison_202607) ---
    # ``root_selection``: "puct" (default; legacy path, bit-identical) or "rm"
    # (regret-matching root: after one forced visit per action, the root action
    # is SAMPLED from the RM+ policy mixed with ``rm_gamma`` uniform, and the
    # emitted ``pi_rm`` is the linearly-weighted average RM strategy). Interior
    # nodes keep PUCT either way. RM's exploration never enters its target:
    # forced/gamma visits update Q but ``pi_rm`` accumulates only sigma(regret+).
    # Bandit-grade property only (opponent models are fixed): converges toward
    # best response, not equilibrium.
    root_selection: str = "puct"
    rm_gamma: float = 0.10
    # Gumbel-style completed-Q readout, emitted as ``pi_gumbel`` on every search
    # (readout-only; never affects selection or ``pi``):
    #   pi_gumbel(a) ∝ exp(log P_raw(a) + (c_visit + max_b N(b)) * c_scale * qhat(a))
    # with qhat min-max normalized over root actions, unvisited actions completed
    # with the visit-weighted mean Q, and P_raw the UNMIXED network prior
    # (root_explore_frac never contaminates it). Adapted from Danihelka et al.
    # 2022 to the belief-averaged determinized root. Visit counts enter only
    # through the sigma scale (more visits -> sharper), so the forced-exploration
    # floor cannot leak into the target mass.
    gumbel_c_visit: float = 50.0
    gumbel_c_scale: float = 0.1
    # Truncated-leaf value source. "oracle" (default; operator decision
    # 2026-08-10) evaluates rollout leaves with the agent's privileged
    # OracleValueNetwork on the OBSERVER's full-information event stream —
    # legitimate inside a determinized world (all hands are known) and the
    # oracle's native input regime, so strictly better calibrated than the
    # limited critic wherever an oracle head exists. The stream is captured
    # to mirror training exactly (pre-action states + post-trick frames,
    # fresh zero memory per sequence) and evaluated lazily at the leaf.
    # Falls back to "limited" silently when the agent carries no oracle
    # critic (e.g. selfplay checkpoints); set "limited" to force the legacy
    # observation-only bootstrap. Terminal rollouts never consult either.
    leaf_evaluator: str = "oracle"


class _Node:
    """Statistics-only ISMCTS node, keyed (implicitly, by tree position) on the
    observer's action sequence. All counts are *weighted* by the per-iteration
    determinization importance weight.

    Per-action dicts (standard MCTS notation, keyed by action id):
      * ``N``     — visit count.
      * ``W``     — total backed-up value (mean Q = W/N).
      * ``P``     — network prior (root: explore_frac-mixed).
      * ``avail`` — availability count: rounds this action was LEGAL in the
        sim's determinized world (availability-count PUCT at follow nodes).
      * ``vloss`` — in-flight virtual-loss visits (leaf-parallel batching).
    """

    __slots__ = ("children", "N", "W", "P", "avail", "visited", "vloss")

    def __init__(self):
        self.children: dict[int, _Node] = {}
        self.N: dict[int, float] = {}
        self.W: dict[int, float] = {}
        self.P: dict[int, float] = {}
        self.avail: dict[int, float] = {}
        self.visited: bool = False
        # In-flight virtual-loss visit counts per action (leaf-parallel batching).
        self.vloss: dict[int, int] = {}


class _RootRM:
    """Regret-matching (RM+) root state: cumulative positive regrets plus the
    linearly-weighted average strategy (CFR+-style averaging). Updated once per
    completed simulation from the root's current mean-Q table (all actions
    visited); the average strategy is the ``pi_rm`` readout."""

    __slots__ = ("regret", "strat_sum", "t")

    def __init__(self, valid):
        self.regret = {a: 0.0 for a in valid}
        self.strat_sum = {a: 0.0 for a in valid}
        self.t = 0

    def sigma(self):
        total = sum(self.regret.values())
        if total <= 0.0:
            n = len(self.regret)
            return {a: 1.0 / n for a in self.regret}
        return {a: r / total for a, r in self.regret.items()}

    def update(self, q_norm):
        sig = self.sigma()
        self.t += 1
        v_bar = sum(sig[a] * q_norm[a] for a in sig)
        for a in sig:
            self.regret[a] = max(self.regret[a] + q_norm[a] - v_bar, 0.0)
            self.strat_sum[a] += self.t * sig[a]

    def average(self):
        total = sum(self.strat_sum.values())
        if total <= 0.0:
            return self.sigma()
        return {a: s / total for a, s in self.strat_sum.items()}


class _Sim:
    """One in-flight simulation in a leaf-parallel batch. Carries its determinized
    world, its per-seat recurrent memory (5, 256), its tree path (for backprop),
    and a small state machine: ``tree`` (observer decision at ``node``) -> ``advance``
    (opponents play to the observer's next turn) -> ``tree`` (child) ... -> ``rollout``
    (all seats sampled to the depth cap) -> ``done``."""

    __slots__ = (
        "world",
        "mem",
        "phase",
        "node",
        "depth",
        "path",
        "obs_plays",
        "pending_action",
        "seat",
        "valid",
        "oracle_seq",
    )

    def __init__(self, world, mem, root):
        self.world = world
        self.mem = mem  # (5, 256) tensor; row s-1 is seat s's memory
        self.phase = "tree"
        self.node = root
        self.depth = 0
        self.path: list = []  # (node, action) edges to backprop
        self.obs_plays = 0
        self.pending_action = None
        self.seat = None  # acting seat for the pending encode this round
        self.valid = None
        # Observer's full-information event stream for oracle leaf evaluation
        # (prefix from the world build + captures along this sim's own path;
        # dicts are shared immutably with the prefix, never mutated).
        self.oracle_seq: list = []


class _OracleCapture:
    """Collects the observer's full-information event streams for oracle leaf
    evaluation, mirroring the oracle critic's training protocol (pre-action
    observer states + post-trick frames, in game order, fresh zero memory per
    sequence).

    One instance per search. A disabled instance is a null object — every
    method early-returns — so call sites stay unconditional. During the world
    build each world accumulates a PREFIX stream (stored here keyed by world
    identity); each sim then extends a shallow copy of its pool world's prefix
    along its own descent/rollout path. Prefix dicts are shared
    immutably across sims, never mutated."""

    __slots__ = ("enabled", "_prefix_by_world")

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._prefix_by_world: dict[int, list] = {}

    def record_decision(self, stream: list, player) -> None:
        """Append the observer's pre-action oracle state (call only for the
        observer's own decisions)."""
        if self.enabled:
            stream.append(player.get_oracle_state_dict())

    def record_trick(self, stream: list, observer_player) -> None:
        """Append the observer's post-trick oracle frame."""
        if self.enabled:
            stream.append(observer_player.get_last_trick_oracle_state_dict())

    def store_prefix(self, world, prefix: list) -> None:
        """Register a root-ready world's prefix stream (keyed by identity;
        pool worlds stay alive for the duration of the search)."""
        if self.enabled:
            self._prefix_by_world[id(world)] = prefix

    def seq_for_world(self, world) -> list:
        """A per-sim shallow copy of ``world``'s prefix stream ([] when
        disabled or unknown — matching a fresh sim's empty stream)."""
        if not self.enabled:
            return []
        return list(self._prefix_by_world.get(id(world), []))

    def clear(self) -> None:
        """Drop the prefix store (end of search); the enabled flag persists
        for post-search inspection."""
        self._prefix_by_world = {}


class _EncodeRequest(NamedTuple):
    """One pending network evaluation for a sim this round.

    ``kind`` is the request type (``sim.seat``/``sim.valid`` are already set):
      * ``"tree"``   — observer decision at a tree node: actor probs feed
        prior expansion + PUCT/RM selection.
      * ``"world"``  — non-observer advance or rollout action: actor probs
        are sampled from directly.
      * ``"critic"`` — rollout depth cap reached: bootstrap the leaf value
        (limited critic or oracle) at this state.
    """

    sim: _Sim
    kind: str


class ISMCTSTeacher:
    def __init__(self, agent, config: ISMCTSConfig | None = None):
        self.agent = agent
        self.action_size = agent.action_size
        self.config = config or ISMCTSConfig()
        # Per-search transient state (reset in ``search``).
        self._rng = None
        self._oracle = _OracleCapture(False)
        self._qmin = math.inf
        self._qmax = -math.inf
        self._max_depth = 1
        # Effective rollout depth for the current search: the per-call override
        # (the trainer's trick-indexed schedule — roll deep early-game where the
        # critic is blind, shallow + bootstrap once the value head is calibrated
        # mid-game) or config.d_rollout. Resolved once per search.
        self._eff_d_rollout: int = self.config.d_rollout
        # Per-search non-observer seat controllers (population grounding); None
        # or a missing seat -> self.agent (pure self-play).
        self._seat_policies: dict | None = None
        # Root-readout transient state (reset in ``_search_inner``): the root
        # node, the RM state (root_selection == "rm" only), and the running sum
        # of RAW root priors for the pi_gumbel readout.
        self._root: _Node | None = None
        self._root_rm: _RootRM | None = None
        self._root_praw: dict = {}
        self._root_praw_writes = 0
        self.fail = defaultdict(int)

    def _controller(self, seat: int):
        """The PPOAgent modeling ``seat`` for this search (observer and unmapped
        seats -> ``self.agent``)."""
        if self._seat_policies is None:
            return self.agent
        return self._seat_policies.get(seat, self.agent)

    @property
    def _oracle_capture(self) -> bool:
        """Whether the current/most-recent search evaluated leaves with the
        oracle critic (see the arming condition in ``search``)."""
        return self._oracle.enabled

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def search(
        self,
        real_game,
        observer: int,
        forced_public,
        rng,
        d_rollout: int | None = None,
        seat_policies: dict | None = None,
    ) -> SearchResult:
        """Run the SO-ISMCTS teacher for ``observer``'s current decision in
        ``real_game``.

        Parameters
        ----------
        real_game : Game
            The live game, positioned at the observer's decision (the action has
            *not* been applied).
        observer : int
            Seat (1-5) whose information set is searched.
        forced_public : list[tuple[int, int]]
            Chronological ``(seat, action_id)`` of every PUBLIC action taken
            before this decision (passes / pick / call / alone / jd-partner /
            plays). Private bury/under actions are *not* included; they are forced
            from the sampled determinization during replay.
        rng : random.Random
            RNG for determinization sampling.
        d_rollout : int | None
            Optional per-call override of ``config.d_rollout`` (number of further
            observer *play* plies rolled before the critic bootstraps). Used by the
            training loop's trick-indexed depth schedule; ``None`` falls back to the
            config value.
        seat_policies : dict | None
            Optional ``{seat: PPOAgent}`` modeling NON-observer seats in the
            replay/advance/rollout (population grounding) — normally the agents
            actually controlling those seats in the live game. The observer
            entry, if present, is ignored (the observer is always
            ``self.agent``). ``None`` = pure self-play (legacy behavior).

        Returns
        -------
        dict with keys: ``pi`` (float32[action_size] soft target), ``ess``,
        ``ok`` (ESS >= floor and statistics present), ``head``, ``n_iter``
        (worlds successfully built), ``valid`` (root legal action ids),
        ``root_n`` / ``root_q`` (per-action weighted visit count / mean value).
        """
        self._rng = rng
        self._eff_d_rollout = (
            d_rollout if d_rollout is not None else self.config.d_rollout
        )
        # Oracle leaf evaluation is armed only when the config asks for it,
        # the agent actually carries an oracle critic, and the effective
        # rollout depth can produce critic leaves at all.
        self._oracle = _OracleCapture(
            self.config.leaf_evaluator == "oracle"
            and getattr(self.agent, "oracle_critic", None) is not None
            and self._eff_d_rollout <= ORACLE_LEAF_MAX_BOOTSTRAP_DEPTH
        )
        self._seat_policies = (
            {
                seat: agent
                for seat, agent in seat_policies.items()
                if seat != observer and agent is not None
            }
            if seat_policies
            else None
        )
        # The sequential paths route per-seat memory through each controller's
        # _player_memories — including the LIVE game's opponent agents — so
        # snapshot/restore every distinct agent involved, not just self.agent.
        involved = {id(self.agent): self.agent}
        if self._seat_policies:
            for agent in self._seat_policies.values():
                involved[id(agent)] = agent
        saved_memories = {
            agent_id: agent.snapshot_player_memories()
            for agent_id, agent in involved.items()
        }
        try:
            return self._search_inner(real_game, observer, forced_public)
        finally:
            for agent_id, agent in involved.items():
                agent.restore_player_memories(saved_memories[agent_id])
            self._eff_d_rollout = self.config.d_rollout
            self._seat_policies = None
            self._oracle.clear()

    # ------------------------------------------------------------------
    # Search driver
    # ------------------------------------------------------------------
    def _search_inner(self, real_game, observer, forced_public) -> SearchResult:
        observer_player = real_game.players[observer - 1]
        valid_real = sorted(observer_player.get_valid_action_ids())
        head = self._infer_head(valid_real)
        config = self.config
        m_iters = config.iters[head]
        self._max_depth = config.max_depth[head]

        # Reset transient search state.
        root = _Node()
        self._qmin = math.inf
        self._qmax = -math.inf
        self._root = root
        self._root_rm = _RootRM(valid_real) if config.root_selection == "rm" else None
        self._root_praw = {a: 0.0 for a in valid_real}
        self._root_praw_writes = 0
        self.fail = defaultdict(int)

        # Build a belief-weighted pool of determinized worlds, then run the tree
        # search by SAMPLING worlds from the pool ~ exp(log_weight) (scheme-B bidding
        # belief). The tree accumulates UNIT-weight visits, so it gets the full
        # m_iters of exploration (FPU + PUCT behave correctly and the visit-count
        # target is built from m_iters samples), while the bidding inference
        # enters only through how often each world is presented. This decouples
        # belief from visit allocation: weighting the tree counts directly would
        # collapse the effective visit budget to the (low, mid-game) ESS and the
        # target would just echo the policy. ESS is reported for the abort gate.
        pool = self._build_pool(real_game, observer, forced_public, m_iters)
        ess = self._pool_ess(pool)
        if pool:
            probs = self._pool_probs(pool)
            indices = self._rng.choices(range(len(pool)), weights=probs, k=m_iters)
            self._run_batched(root, pool, indices, observer)

        return self._finalize(root, valid_real, head, len(pool), ess)

    def _build_pool(self, real_game, observer, forced_public, n_worlds):
        """Sample up to ``n_worlds`` determinized worlds and rebuild all of them
        to the root by a single LOCKSTEP forced replay. Returns a list of
        ``(game_at_root, memory_snapshot, log_weight)``; ``log_weight`` is the
        scheme-B bidding log-likelihood.

        Every world replays the *identical* public action sequence
        (``forced_public``) and the same per-decision structure (the private
        bury/under count is fixed by the public record), differing only in the
        hidden cards. So instead of replaying each world separately (n_worlds
        batch-1 encoder calls per decision — the dominant search cost; see
        profiling), we step all worlds together and batch the encoder/actor over
        the n_worlds worlds at each decision point. Both builds are executors of
        the shared ``_replay_events`` director; their equivalence is pinned by
        the pool tests and the replay goldens."""
        config = self.config
        deals = []
        for _ in range(n_worlds):
            try:
                deals.append(
                    real_game.sample_determinization(
                        observer, self._rng, max_tries=config.det_max_tries
                    )
                )
            except RuntimeError:
                self.fail[FAIL_DETERMINIZE] += 1
        if not deals:
            return []
        return self._build_worlds_batched(real_game, deals, forced_public, observer)

    def _fresh_world(self, real_game, deal):
        """Fresh Game with the determinized hands + blind installed (pre-replay)."""
        world = Game(partner_selection_mode=real_game.partner_mode_flag)
        for seat in range(1, 6):
            hand = deal["initial_hands"][seat][:]
            world.players[seat - 1].hand = hand
            world.players[seat - 1].initial_hand = hand[:]
        world.blind = deal["blind"][:]
        return world

    def _encode_seat_batched(self, games, seat, seat_memories):
        """Batch-encode ``seat``'s current state across all ``games`` with that
        seat's controller and advance its (n, 256) recurrent memory. Returns
        (states, encoder_out)."""
        ctrl = self._controller(seat)
        states = [game.players[seat - 1].get_state_dict() for game in games]
        encoded = ctrl.encoder.encode_batch(
            states, memory_in=seat_memories[seat], device=DEV
        )
        seat_memories[seat] = encoded["memory_out"].detach()
        return states, encoded

    def _masked_actor_probs(self, ctrl, encoded, states, valid_list):
        """Post-mixture action probabilities (n, A) under ``ctrl`` for
        already-encoded states — the shared mask/hand_ids/actor plumbing.
        Mirrors ``get_action_probs_with_logits`` over n states at once;
        ``encoded`` must come from ``ctrl``'s own encoder."""
        masks = torch.stack(
            [ctrl.get_action_mask(valid, self.action_size) for valid in valid_list]
        ).to(DEV)
        hand_ids = torch.as_tensor(
            np.stack([state["hand_ids"] for state in states]),
            dtype=torch.long,
            device=DEV,
        )
        with torch.no_grad():
            probs, _ = ctrl.actor.forward_with_logits(
                encoded, masks, hand_ids, ctrl.encoder.card
            )
        return probs

    def _actor_probs_batched(self, encoded, states, valid_list, seat):
        """``_masked_actor_probs`` under ``seat``'s controller (replay path)."""
        return self._masked_actor_probs(
            self._controller(seat), encoded, states, valid_list
        )

    def _observe_trick_lockstep(self, games, seat_memories):
        """End-of-trick observe for every seat (with its controller), batched over
        worlds into the local ``seat_memories`` tensors (the plays are forced
        identically, so trick completion is synchronized across worlds)."""
        if not games[0].was_trick_just_completed:
            return
        for seat in range(1, 6):
            ctrl = self._controller(seat)
            states = [
                game.players[seat - 1].get_last_trick_state_dict() for game in games
            ]
            encoded = ctrl.encoder.encode_batch(
                states, memory_in=seat_memories[seat], device=DEV
            )
            seat_memories[seat] = encoded["memory_out"].detach()

    def _build_worlds_batched(self, real_game, deals, forced_public, observer):
        """Build the world pool, batched. Fast path is the lockstep replay; if any
        world is inconsistent with the forced replay (rare — see
        ``_ReplayInconsistency``), fall back to the per-world sequential build,
        which drops bad worlds instead of aborting."""
        try:
            return self._build_worlds_lockstep(
                real_game, deals, forced_public, observer
            )
        except _ReplayInconsistency:
            self.fail[FAIL_BATCHED_FALLBACK] += 1
            return self._build_pool_sequential(
                real_game, deals, forced_public, observer
            )

    def _build_pool_sequential(self, real_game, deals, forced_public, observer):
        """Per-world sequential build (the robust fallback executor): replay
        each deal with ``_build_world`` and skip the ones that fail
        (returns None)."""
        pool = []
        for deal in deals:
            world, log_weight = self._build_world(
                real_game, deal, forced_public, observer
            )
            if world is None:
                continue
            # Each seat's memory lives in its controller's dict after the replay.
            memory_snapshot = {}
            for seat in range(1, 6):
                memory = self._controller(seat)._player_memories.get(seat)
                if memory is not None:
                    memory_snapshot[seat] = memory.detach().clone()
            pool.append((world, memory_snapshot, log_weight))
        return pool

    def _build_worlds_lockstep(self, real_game, deals, forced_public, observer):
        """Lockstep batched EXECUTOR of the shared ``_replay_events`` director:
        one director instance driven off world 0 (all worlds share the
        public/private decision structure), each event applied to every world
        with batched encodes into local ``seat_memories`` tensors.

        Fast path semantics: the lockstep cannot drop a single inconsistent
        world mid-flight, so any pre-root failure raises
        ``_ReplayInconsistency`` (counter incremented here) for the caller's
        per-world fallback; root-stage history mismatches are instead dropped
        per world in ``_collect_lockstep_pool``. The per-world executor is
        ``_build_world``; their equivalence is pinned by
        test_batched_pool_matches_sequential / test_replay_equivalence_panel
        and the replay goldens."""
        n_worlds = len(deals)
        games = [self._fresh_world(real_game, deal) for deal in deals]
        oracle_prefixes: list[list] = [[] for _ in games]
        det_buries = [deque(deal["bury"]) for deal in deals]
        det_unders = [deal["under_card"] for deal in deals]
        mem_width = self.agent.state_size
        seat_memories = {
            seat: torch.zeros((n_worlds, mem_width), device=DEV) for seat in range(1, 6)
        }
        log_weights = torch.zeros(n_worlds, device=DEV)
        try:
            for event in _replay_events(real_game, games[0], forced_public, observer):
                log_weights = self._apply_event_lockstep(
                    games,
                    event,
                    det_buries,
                    det_unders,
                    observer,
                    seat_memories,
                    log_weights,
                    oracle_prefixes,
                )
        except _ReplayInconsistency as exc:
            self.fail[exc.key] += 1
            raise
        return self._collect_lockstep_pool(
            games, real_game, seat_memories, log_weights, oracle_prefixes
        )

    def _apply_event_lockstep(
        self,
        games,
        event,
        det_buries,
        det_unders,
        observer,
        seat_memories,
        log_weights,
        oracle_prefixes,
    ):
        """Apply one director event to every lockstepped world (batched encode,
        per-world act) and run the synchronized end-of-trick observe. Returns
        the updated ``log_weights`` tensor. Any per-world failure raises
        ``_ReplayInconsistency`` out of the whole build (all-or-nothing)."""
        seat = event.seat
        if isinstance(event, _PrivateDecision):
            # Forced bury/under: encode (advance memory), then act each
            # world with its own determinized card. Not weighted.
            self._encode_seat_batched(games, seat, seat_memories)
            for i, game in enumerate(games):
                world_valid = game.players[seat - 1].get_valid_action_ids()
                action_id = self._forced_private(
                    world_valid, det_buries[i], det_unders[i]
                )
                if action_id is None or action_id not in world_valid:
                    raise _ReplayInconsistency(
                        FAIL_BAD_PRIVATE, "batched replay: bad forced private action"
                    )
                if seat == observer:
                    self._oracle.record_decision(
                        oracle_prefixes[i], game.players[seat - 1]
                    )
                game.players[seat - 1].act(action_id)
        else:
            action_id = event.action_id
            states, encoded = self._encode_seat_batched(games, seat, seat_memories)
            # Weight bidding actions only (scheme B); plays are forced
            # but never weighted. The actor head runs only here.
            if event.weighted:
                valid_lists = [
                    game.players[seat - 1].get_valid_action_ids() for game in games
                ]
                probs = self._actor_probs_batched(encoded, states, valid_lists, seat)
                action_probs = probs[:, action_id - 1].clamp_min(1e-8)
                log_weights = log_weights + torch.log(action_probs)
            for i, game in enumerate(games):
                world_valid = game.players[seat - 1].get_valid_action_ids()
                if action_id not in world_valid:
                    raise _ReplayInconsistency(
                        FAIL_BAD_PUBLIC, "batched replay: bad forced public action"
                    )
                if seat == observer:
                    self._oracle.record_decision(
                        oracle_prefixes[i], game.players[seat - 1]
                    )
                game.players[seat - 1].act(action_id)
        self._observe_trick_lockstep(games, seat_memories)
        if self._oracle.enabled:
            for i, game in enumerate(games):
                if game.was_trick_just_completed:
                    self._oracle.record_trick(
                        oracle_prefixes[i], game.players[observer - 1]
                    )
        return log_weights

    def _collect_lockstep_pool(
        self, games, real_game, seat_memories, log_weights, oracle_prefixes
    ):
        """Root-stage per-world filter + snapshot: a history-mismatched world
        is DROPPED (counter, never an exception — mirroring the sequential
        executor's drop); survivors get a dense 5-seat memory snapshot from
        the local tensors and their oracle prefix registered."""
        pool = []
        for i, game in enumerate(games):
            if game.history != real_game.history:
                self.fail[FAIL_HIST_MISMATCH] += 1
                continue
            memory_snapshot = {
                seat: seat_memories[seat][i].detach().clone() for seat in range(1, 6)
            }
            self._oracle.store_prefix(game, oracle_prefixes[i])
            pool.append((game, memory_snapshot, float(log_weights[i].item())))
        return pool

    @staticmethod
    def _pool_probs(pool):
        weights = _pool_norm_weights(pool)
        return (weights / weights.sum()).tolist()

    @staticmethod
    def _pool_ess(pool) -> float:
        if not pool:
            return 0.0
        weights = _pool_norm_weights(pool)
        total = weights.sum()
        if total <= 0:
            return 0.0
        return float(total * total / np.square(weights).sum())

    def _finalize(self, root, valid_real, head, n_worlds_built, ess) -> SearchResult:
        pi = np.zeros(self.action_size, dtype=np.float32)
        counts = np.array(
            [root.N.get(action_id, 0.0) for action_id in valid_real], dtype=np.float64
        )
        root_n = {
            action_id: float(root.N.get(action_id, 0.0)) for action_id in valid_real
        }
        root_q = {
            action_id: (
                float(root.W[action_id] / root.N[action_id])
                if root.N.get(action_id, 0.0) > 0
                else 0.0
            )
            for action_id in valid_real
        }
        root_prior = (
            {
                action_id: self._root_praw.get(action_id, 0.0) / self._root_praw_writes
                for action_id in valid_real
            }
            if self._root_praw_writes > 0
            else None
        )
        ok = False
        pi_gumbel = None
        pi_rm = None
        if counts.sum() > 0.0:
            powered = np.power(counts, 1.0 / self.config.tau_target)
            powered /= powered.sum()
            for action_id, prob in zip(valid_real, powered):
                pi[action_id - 1] = prob
            ok = ess >= self.config.ess_floor
            if self._root_rm is not None:
                pi_rm = np.zeros(self.action_size, dtype=np.float32)
                for action_id, prob in self._root_rm.average().items():
                    pi_rm[action_id - 1] = prob
            pi_gumbel = self._gumbel_readout(valid_real, counts, root_q, root_prior)
        return SearchResult(
            pi=pi,
            ess=ess,
            ok=ok,
            head=head,
            n_iter=n_worlds_built,
            valid=valid_real,
            root_n=root_n,
            root_q=root_q,
            root_prior=root_prior,
            pi_gumbel=pi_gumbel,
            pi_rm=pi_rm,
        )

    def _gumbel_readout(self, valid_real, counts, root_q, root_prior):
        """Completed-Q root readout (``pi_gumbel``): softmax(log P_raw +
        (c_visit + max N) * c_scale * qhat) over the legal set. Unvisited
        actions are completed with the visit-weighted mean Q; qhat is min-max
        normalized over the root actions (this codebase's Q convention; the
        mctx transform differs slightly). Forced-exploration visits never enter
        the target mass — only the sharpness scale."""
        if root_prior is None:
            return None
        praw = np.array(
            [root_prior[action_id] for action_id in valid_real], dtype=np.float64
        )
        q = np.array([root_q[action_id] for action_id in valid_real], dtype=np.float64)
        visited = counts > 0.0
        if not visited.any():
            return None
        v_mix = float((counts[visited] * q[visited]).sum() / counts[visited].sum())
        q_completed = np.where(visited, q, v_mix)
        qhat = _minmax_unit(q_completed)
        scale = (
            self.config.gumbel_c_visit + float(counts.max())
        ) * self.config.gumbel_c_scale
        logits = np.log(np.clip(praw, 1e-12, None)) + scale * qhat
        z = np.exp(logits - logits.max())
        z /= z.sum()
        pi = np.zeros(self.action_size, dtype=np.float32)
        for action_id, prob in zip(valid_real, z):
            pi[action_id - 1] = prob
        return pi

    @staticmethod
    def _infer_head(valid) -> str:
        names = [ACTIONS[action_id - 1] for action_id in valid]
        if any(name in ("PICK", "PASS") for name in names):
            return "pick"
        if any(
            name == "ALONE" or name == "JD PARTNER" or name.startswith("CALL ")
            for name in names
        ):
            return "partner"
        if any(name.startswith("BURY ") or name.startswith("UNDER ") for name in names):
            return "bury"
        return "play"

    # ------------------------------------------------------------------
    # Leaf-parallel batched search (Tier 2): run batch_size simulations
    # concurrently and batch every encoder/actor/critic call across them, with
    # virtual loss so concurrent sims in a chunk diversify. Profiling: the tree
    # descent + opponent advance + per-trick observes are ~84% of search
    # encodes, so this is where the throughput is.
    # ------------------------------------------------------------------
    @staticmethod
    def _next_actor(world):
        """First seat (in seat order 1-5) with a legal action, or None if the
        world is terminal."""
        for player in world.players:
            if player.get_valid_action_ids():
                return player.position
        return None

    def _run_batched(self, root, pool, indices, observer):
        batch_size = self.config.batch_size
        start = 0
        while start < len(indices):
            chunk = indices[start : start + batch_size]
            start += len(chunk)
            sims = []
            for pool_idx in chunk:
                world = copy.deepcopy(pool[pool_idx][0])
                memory_snapshot = pool[pool_idx][1]
                mem = torch.zeros((5, self.agent.state_size), device=DEV)
                for seat in range(1, 6):
                    if seat in memory_snapshot:
                        mem[seat - 1] = memory_snapshot[seat]
                sim = _Sim(world, mem, root)
                sim.oracle_seq = self._oracle.seq_for_world(pool[pool_idx][0])
                sims.append(sim)
            self._run_chunk(sims, observer)

    def _run_chunk(self, sims, observer):
        """Drive a chunk of sims to completion, one network round per pass:
        collect encode requests -> batched network forward -> (oracle leaves)
        -> apply outputs -> end-of-trick observes."""
        guard = 0
        while any(sim.phase != "done" for sim in sims):
            guard += 1
            if guard > 100000:
                raise RuntimeError("batched chunk guard exceeded")
            requests = self._collect_requests(sims, observer)
            if not requests:
                continue
            probs_np, values_np = self._run_network_round(requests)
            self._evaluate_oracle_leaves(requests, values_np, observer)
            completers = self._apply_round(requests, probs_np, values_np, observer)
            for sim in completers:
                self._oracle.record_trick(
                    sim.oracle_seq, sim.world.players[observer - 1]
                )
            self._observe_completers_batched(completers)

    def _collect_requests(self, sims, observer) -> list[_EncodeRequest]:
        """Resolve no-network state-machine transitions; return this round's
        encode requests (sims that finished during preparation drop out)."""
        requests = []
        for sim in sims:
            if sim.phase == "done":
                continue
            kind = self._prepare(sim, observer)
            if kind is not None:
                requests.append(_EncodeRequest(sim, kind))
        return requests

    def _run_network_round(self, requests):
        """Encode every request and run the actor (and, without oracle leaves,
        the limited critic) — grouped by the acting seat's controller
        (population grounding). The encode COUNT matches the ungrouped path
        exactly; only the batch grouping fragments when seat_policies are
        present (see notebooks/Population_Grounded_Teacher_Plan.md). The
        critic runs only on groups that contain a bootstrap request. Returns
        ``(probs_np, values_np)`` indexed like ``requests``."""
        states = [
            req.sim.world.players[req.sim.seat - 1].get_state_dict() for req in requests
        ]
        groups: dict[int, tuple] = {}
        for req_idx, req in enumerate(requests):
            ctrl = self._controller(req.sim.seat)
            groups.setdefault(id(ctrl), (ctrl, []))[1].append(req_idx)

        probs_np = np.zeros((len(requests), self.action_size), dtype=np.float32)
        values_np = np.zeros(len(requests), dtype=np.float32)
        for ctrl, req_idxs in groups.values():
            group_states = [states[req_idx] for req_idx in req_idxs]
            memory_in = torch.stack(
                [
                    requests[req_idx].sim.mem[requests[req_idx].sim.seat - 1]
                    for req_idx in req_idxs
                ]
            )
            encoded = ctrl.encoder.encode_batch(
                group_states, memory_in=memory_in, device=DEV
            )
            memory_out = encoded["memory_out"].detach()
            for row, req_idx in enumerate(req_idxs):
                sim = requests[req_idx].sim
                sim.mem[sim.seat - 1] = memory_out[row]
            probs = self._masked_actor_probs(
                ctrl,
                encoded,
                group_states,
                [requests[req_idx].sim.valid for req_idx in req_idxs],
            )
            if not self._oracle_capture and any(
                requests[req_idx].kind == "critic" for req_idx in req_idxs
            ):
                with torch.no_grad():
                    values_np[req_idxs] = (
                        ctrl.critic(encoded).detach().view(-1).cpu().numpy()
                    )
            probs_np[req_idxs] = probs.detach().cpu().numpy()
        return probs_np, values_np

    def _evaluate_oracle_leaves(self, requests, values_np, observer):
        """Oracle leaf evaluation: one ragged batch over the critic requests'
        full-information event streams (prefix + path + the leaf decision
        state), fresh zero memory per sequence — the training protocol of
        forward_sequences. Same value units as the limited critic (both
        trained on the same lambda-returns). Overwrites ``values_np`` rows.
        No-op unless oracle capture is armed."""
        if not self._oracle.enabled:
            return
        critic_rows = [
            req_idx for req_idx, req in enumerate(requests) if req.kind == "critic"
        ]
        if not critic_rows:
            return
        seqs = []
        for req_idx in critic_rows:
            sim = requests[req_idx].sim
            seqs.append(
                sim.oracle_seq
                + [sim.world.players[observer - 1].get_oracle_state_dict()]
            )
        with torch.no_grad():
            oracle_vals = self.agent.oracle_critic.forward_sequences(seqs, device=DEV)
        for row, req_idx in enumerate(critic_rows):
            values_np[req_idx] = float(oracle_vals[row, len(seqs[row]) - 1].item())

    def _apply_round(self, requests, probs_np, values_np, observer):
        """Apply each request's network outputs to its sim; return the sims
        that completed a trick this round (their end-of-trick observe is
        batched afterwards)."""
        completers = []
        for req_idx, req in enumerate(requests):
            if req.kind == "critic":
                self._finish_value(
                    req.sim,
                    self._discount(float(values_np[req_idx]), req.sim.obs_plays),
                )
            else:
                self._apply_actor(
                    req.sim,
                    observer,
                    probs_np[req_idx],
                    req.kind == "tree",
                    completers,
                )
        return completers

    def _prepare(self, sim, observer):
        """Run no-network state-machine transitions until ``sim`` needs an encode
        (sets sim.seat/valid and returns the ``_EncodeRequest`` kind —
        "tree" / "world" / "critic") or is done (returns None)."""
        while True:
            world = sim.world
            if world.is_done():
                self._finish_terminal(sim, observer)
                return None
            if sim.phase == "tree":
                valid = sorted(world.players[observer - 1].get_valid_action_ids())
                if not valid:
                    sim.phase = "rollout"  # defensive; should not happen at a tree node
                    continue
                sim.seat, sim.valid = observer, valid
                return "tree"
            if sim.phase == "advance":
                next_seat = self._next_actor(world)
                if next_seat is None:
                    self._finish_terminal(sim, observer)
                    return None
                if next_seat == observer:
                    # Done advancing -> descend into the selected action's child.
                    parent, action_id = sim.node, sim.pending_action
                    child = parent.children.get(action_id)
                    if child is None:
                        child = _Node()
                        parent.children[action_id] = child
                    sim.node, sim.depth, sim.pending_action = child, sim.depth + 1, None
                    sim.phase = "tree"
                    continue
                sim.seat = next_seat
                sim.valid = sorted(world.players[next_seat - 1].get_valid_action_ids())
                return "world"
            # rollout
            next_seat = self._next_actor(world)
            if next_seat is None:
                self._finish_terminal(sim, observer)
                return None
            sim.seat = next_seat
            sim.valid = sorted(world.players[next_seat - 1].get_valid_action_ids())
            if (
                next_seat == observer
                and _valid_has_play(sim.valid)
                and sim.obs_plays >= self._eff_d_rollout
            ):
                return "critic"
            return "world"

    def _apply_actor(self, sim, observer, probs, is_tree, completers):
        if is_tree:
            self._apply_tree_decision(sim, observer, probs, completers)
        else:
            self._apply_world_action(sim, observer, probs, completers)

    def _apply_tree_decision(self, sim, observer, probs, completers):
        """Observer decision at a tree node: write priors/availability, stop at
        a leaf (-> rollout), otherwise select (PUCT / root-RM), charge virtual
        loss, and act into the ``advance`` phase."""
        node, valid = sim.node, sim.valid
        is_root = sim.depth == 0
        explore_frac = self.config.root_explore_frac
        n_legal = len(valid)
        if is_root:
            self._root_praw_writes += 1
        for action_id in valid:
            prior = float(probs[action_id - 1])
            if is_root:
                # Accumulate the UNMIXED prior for the pi_gumbel readout
                # before the explore_frac mix touches it.
                self._root_praw[action_id] = self._root_praw.get(action_id, 0.0) + prior
            if is_root and explore_frac > 0.0:
                prior = (1.0 - explore_frac) * prior + explore_frac / n_legal
            node.P[action_id] = prior
            node.N.setdefault(action_id, 0.0)
            node.W.setdefault(action_id, 0.0)
            node.avail.setdefault(action_id, 0.0)
            node.avail[action_id] += 1.0
        leaf = (not node.visited) or (sim.depth >= self._max_depth)
        node.visited = True
        if leaf:
            # Observer is rolled out starting next round (re-encoded there;
            # the freshly-written priors are not consumed at a leaf).
            sim.phase = "rollout"
            return
        use_availability_puct = self._is_following(sim.world, observer)
        if is_root and self._root_rm is not None:
            action_id = self._select_root_rm(node, valid)
        else:
            action_id = self._select_vl(node, valid, use_availability_puct)
        node.vloss[action_id] = node.vloss.get(action_id, 0) + 1
        sim.path.append((node, action_id))
        sim.pending_action = action_id
        self._oracle.record_decision(sim.oracle_seq, sim.world.players[observer - 1])
        sim.world.players[observer - 1].act(action_id)
        sim.phase = "advance"
        self._after_world_step(sim, observer, completers)

    def _apply_world_action(self, sim, observer, probs, completers):
        """Non-observer advance / rollout action: sample from the masked policy
        and act; observer rollout plays advance the depth clock."""
        seat, valid = sim.seat, sim.valid
        action_id = self._sample_action(probs, valid)
        is_obs_play = seat == observer and _valid_has_play(valid)
        if seat == observer:
            self._oracle.record_decision(
                sim.oracle_seq, sim.world.players[observer - 1]
            )
        sim.world.players[seat - 1].act(action_id)
        if sim.phase == "rollout" and is_obs_play:
            sim.obs_plays += 1
        self._after_world_step(sim, observer, completers)

    def _after_world_step(self, sim, observer, completers):
        """Shared tail of every world mutation: queue the end-of-trick observe
        and finish the sim if its world just terminated."""
        if sim.world.was_trick_just_completed:
            completers.append(sim)
        if sim.world.is_done():
            self._finish_terminal(sim, observer)

    def _select_root_rm(self, node, valid) -> int:
        """RM+ root selection: force one visit per action first (a Q estimate
        must exist for every action), then SAMPLE from the gamma-mixed RM
        policy. Sampling is inherently diversifying, so root virtual loss is
        moot here (still charged; harmless). Interior nodes always use PUCT."""
        unvisited = [a for a in valid if node.N.get(a, 0.0) + node.vloss.get(a, 0) <= 0]
        if unvisited:
            return unvisited[self._rng.randrange(len(unvisited))]
        sigma = self._root_rm.sigma()
        gamma = self.config.rm_gamma
        n = len(valid)
        return _draw_from_cumulative(
            self._rng,
            valid,
            lambda action_id: (1.0 - gamma) * sigma.get(action_id, 0.0) + gamma / n,
        )

    def _rm_observe(self):
        """One RM+ update from the root's current mean-Q table, min-max
        normalized over the root actions. Called once per completed simulation;
        no-op until every root action has a visit."""
        root, rm = self._root, self._root_rm
        if any(root.N.get(a, 0.0) <= 0.0 for a in rm.regret):
            return
        q = {a: root.W[a] / root.N[a] for a in rm.regret}
        unit = _minmax_unit(np.array(list(q.values()), dtype=np.float64))
        rm.update(dict(zip(q.keys(), unit.tolist())))

    def _select_vl(self, node, valid, use_availability_puct) -> int:
        """PUCT selection with virtual loss: an in-flight selected edge is charged
        ``virtual_loss`` extra (pessimistic) visits so concurrent sims diversify.

        ``use_availability_puct`` swaps the exploration numerator from the
        node-total visit count to the per-action availability count — used at
        follow-suit play nodes, where the legal set varies across determinized
        worlds and total-count PUCT would starve rarely-available actions.
        Q is normalized on the TREE-GLOBAL [qmin, qmax] span (not root-local
        like the RM/gumbel readouts) so FPU=1.0 stays optimistic everywhere."""
        c_puct = self.config.c_puct
        virtual_loss = self.config.virtual_loss
        effective_counts = {
            action_id: node.N[action_id] + node.vloss.get(action_id, 0)
            for action_id in valid
        }
        sqrt_total = math.sqrt(sum(effective_counts.values()) + 1.0)
        qmin, qmax = self._qmin, self._qmax
        has_span = qmax > qmin
        span = (qmax - qmin) if has_span else 1.0
        best_action, best_score = valid[0], -math.inf
        for action_id in valid:
            n_effective = effective_counts[action_id]
            if n_effective > 0:
                w_effective = (
                    node.W[action_id] - node.vloss.get(action_id, 0) * virtual_loss
                )
                q_norm = (w_effective / n_effective - qmin) / span if has_span else 0.5
            else:
                q_norm = self.config.fpu
            if use_availability_puct:
                explore = (
                    c_puct
                    * node.P[action_id]
                    * math.sqrt(node.avail[action_id])
                    / (1.0 + n_effective)
                )
            else:
                explore = c_puct * node.P[action_id] * sqrt_total / (1.0 + n_effective)
            score = q_norm + explore
            if score > best_score:
                best_score, best_action = score, action_id
        return best_action

    def _sample_action(self, probs, valid) -> int:
        """Sample an action id from the masked policy over ``valid`` (search RNG)."""
        return _draw_from_cumulative(
            self._rng, valid, lambda action_id: float(probs[action_id - 1])
        )

    def _gamma(self) -> float:
        return float(getattr(self.agent, "gamma", 1.0))

    def _discount(self, value: float, observer_actions_elapsed: int) -> float:
        if observer_actions_elapsed <= 0:
            return float(value)
        return float((self._gamma() ** observer_actions_elapsed) * value)

    def _terminal_value(
        self, world, observer, observer_actions_elapsed: int = 0
    ) -> float:
        return self._discount(
            world.players[observer - 1].get_score() / RETURN_SCALE,
            observer_actions_elapsed,
        )

    def _finish_terminal(self, sim, observer):
        # During rollout, obs_plays includes the observer action that can carry the
        # terminal reward, so that final action is not additionally discounted.
        # Tree/advance terminal values are discounted across prior tree edges in
        # _finish_value.
        elapsed = max(sim.obs_plays - 1, 0) if sim.phase == "rollout" else 0
        self._finish_value(
            sim,
            self._terminal_value(sim.world, observer, elapsed),
        )

    def _finish_value(self, sim, value):
        backed = float(value)
        for node, action_id in reversed(sim.path):
            node.N[action_id] += 1.0
            node.W[action_id] += backed
            node.vloss[action_id] = node.vloss.get(action_id, 0) - 1
            q = node.W[action_id] / node.N[action_id]
            if q < self._qmin:
                self._qmin = q
            if q > self._qmax:
                self._qmax = q
            backed *= self._gamma()
        sim.phase = "done"
        if self._root_rm is not None and sim.path:
            self._rm_observe()

    def _observe_completers_batched(self, completers):
        """Batched end-of-trick observe (advance every seat's memory) across the
        sims that just completed a trick — the per-trick 5-seat observe is a large
        share of search encodes, so it is batched over the completer subset."""
        if not completers:
            return
        for seat in range(1, 6):
            ctrl = self._controller(seat)
            states = [
                sim.world.players[seat - 1].get_last_trick_state_dict()
                for sim in completers
            ]
            memory_in = torch.stack([sim.mem[seat - 1] for sim in completers])
            encoded = ctrl.encoder.encode_batch(states, memory_in=memory_in, device=DEV)
            memory_out = encoded["memory_out"].detach()
            for i, sim in enumerate(completers):
                sim.mem[seat - 1] = memory_out[i]

    # ------------------------------------------------------------------
    # World advancement helpers
    # ------------------------------------------------------------------
    def _observe_trick_sequential(self, world):
        """End-of-trick observe for every seat, routed through each seat
        controller's own ``_player_memories`` (the sequential replay's memory
        home; the lockstep replay uses local tensors instead)."""
        if world.was_trick_just_completed:
            for seat in world.players:
                self._controller(seat.position).observe(
                    seat.get_last_trick_state_dict(), player_id=seat.position
                )

    @staticmethod
    def _is_following(world, observer) -> bool:
        """Observer is following suit (legal set is determinization-dependent ->
        availability-count PUCT). Lead plays and all bidding/bury decisions have
        a fixed legal set -> plain PUCT."""
        return (
            world.play_started
            and not world.is_leaster
            and world.cards_played > 0
            and bool(world.current_suit)
        )

    # ------------------------------------------------------------------
    # Determinized-world reconstruction (forced replay)
    # ------------------------------------------------------------------
    def _build_world(self, real_game, deal, forced_public, observer):
        """Per-world sequential EXECUTOR of the shared ``_replay_events``
        director: replay the public record into a fresh game whose hidden hands
        are the sampled determinization, rebuilding every seat's recurrent
        memory (through each controller's own ``_player_memories``), and stop
        at the observer's current decision (root). Returns
        ``(world, log_weight)``, or ``(None, None)`` on any inconsistency —
        this executor DROPS a bad world (its own director instance is derived
        from that world) where the lockstep executor must raise.

        ``log_weight`` is the sum of policy log-probs of every forced PUBLIC
        *bidding* action (pick / pass / call / alone / jd-partner) under the
        rebuilt memory + determinized hands (scheme B). Plays are forced (to
        rebuild memory and reproduce the record) but never weighted. Private
        bury/under are forced from the determinization and never weighted.
        Equivalence with ``_build_worlds_lockstep`` is pinned by the pool
        tests and the replay goldens."""
        world = self._fresh_world(real_game, deal)
        self.agent.reset_recurrent_state()
        if self._seat_policies:
            for policy in self._seat_policies.values():
                policy.reset_recurrent_state()
        det_bury = deque(deal["bury"])
        det_under = deal["under_card"]
        log_weight = 0.0
        oracle_prefix: list = []
        try:
            for event in _replay_events(real_game, world, forced_public, observer):
                log_weight += self._apply_event_sequential(
                    world, event, det_bury, det_under, observer, oracle_prefix
                )
        except _ReplayInconsistency as exc:
            self.fail[exc.key] += 1
            return None, None
        if world.history != real_game.history:
            self.fail[FAIL_HIST_MISMATCH] += 1
            return None, None
        self._oracle.store_prefix(world, oracle_prefix)
        return world, log_weight

    def _apply_event_sequential(
        self, world, event, det_bury, det_under, observer, oracle_prefix
    ) -> float:
        """Apply one director event to a sequential world (batch-1 encode via
        the seat controller, which advances its ``_player_memories``) and run
        the end-of-trick observe. Returns the log-weight increment; raises
        ``_ReplayInconsistency`` if the forced action is illegal here."""
        player = world.players[event.seat - 1]
        valid = player.get_valid_action_ids()
        log_weight = 0.0
        if isinstance(event, _PrivateDecision):
            action_id = self._forced_private(valid, det_bury, det_under)
            if action_id is None or action_id not in valid:
                raise _ReplayInconsistency(
                    FAIL_BAD_PRIVATE, "replay: bad forced private action"
                )
            # Advance this seat's memory through the forced decision.
            self._controller(event.seat).get_action_probs_with_logits(
                player.get_state_dict(), valid, player_id=event.seat
            )
        else:
            action_id = event.action_id
            if action_id not in valid:
                raise _ReplayInconsistency(
                    FAIL_BAD_PUBLIC, "replay: bad forced public action"
                )
            probs, _ = self._controller(event.seat).get_action_probs_with_logits(
                player.get_state_dict(), valid, player_id=event.seat
            )
            if event.weighted:
                action_prob = float(probs[0][action_id - 1].item())
                log_weight = math.log(max(action_prob, 1e-8))
        if event.seat == observer:
            self._oracle.record_decision(oracle_prefix, player)
        player.act(action_id)
        self._observe_trick_sequential(world)
        if world.was_trick_just_completed:
            self._oracle.record_trick(oracle_prefix, world.players[observer - 1])
        return log_weight

    @staticmethod
    def _forced_private(valid, det_bury, det_under):
        is_under = any(
            ACTIONS[action_id - 1].startswith("UNDER ") for action_id in valid
        )
        if is_under:
            if det_under is None:
                return None
            return ACTION_IDS.get(f"UNDER {det_under}")
        if not det_bury:
            return None
        return ACTION_IDS.get(f"BURY {det_bury.popleft()}")
