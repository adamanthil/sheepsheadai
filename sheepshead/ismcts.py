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
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import torch

from sheepshead.agent import ppo
from sheepshead import (
    ACTION_IDS,
    ACTIONS,
)
from sheepshead.training.training_utils import RETURN_SCALE

DEV = ppo.device

# Per-head iteration budgets and tree depths (plan §3).
_DEFAULT_ITERS = {"pick": 48, "partner": 64, "bury": 96, "play": 96}
_DEFAULT_DEPTH = {"pick": 1, "partner": 1, "bury": 1, "play": 6}


def _is_play_action(action_id: int) -> bool:
    return ACTIONS[action_id - 1].startswith("PLAY ")


def _is_private_action(action_id: int) -> bool:
    name = ACTIONS[action_id - 1]
    return name.startswith("BURY ") or name.startswith("UNDER ")


def _valid_has_play(valid) -> bool:
    return any(_is_play_action(action_id) for action_id in valid)


def _private_root_ready(real_game, world, valid) -> bool:
    if not any(_is_private_action(action_id) for action_id in valid):
        return True
    return (
        list(world.bury) == list(real_game.bury)
        and world.under_card == real_game.under_card
    )


class _ReplayInconsistency(Exception):
    """A determinized world could not be forced-replayed against the public record
    (a recorded action is illegal in that world, or the lockstep desynced). Rare:
    ``sample_determinization`` is consistent by construction, but void inference is
    not exhaustive, so an occasional redeal makes a recorded play illegal. The
    batched lockstep cannot skip one world mid-flight, so it raises this and the
    caller falls back to the per-world sequential build, which drops bad worlds."""


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


class _Node:
    """Statistics-only ISMCTS node, keyed (implicitly, by tree position) on the
    observer's action sequence. All counts are *weighted* by the per-iteration
    determinization importance weight."""

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
        "value",
        "seat",
        "valid",
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
        self.value = None
        self.seat = None  # acting seat for the pending encode this round
        self.valid = None


class ISMCTSTeacher:
    def __init__(self, agent, config: ISMCTSConfig | None = None):
        self.agent = agent
        self.action_size = agent.action_size
        self.config = config or ISMCTSConfig()
        # Per-search transient state (reset in ``search``).
        self._rng = None
        self._nodes: list[_Node] = []
        self._qmin = math.inf
        self._qmax = -math.inf
        self._max_depth = 1
        # Per-call rollout-depth override (None -> use config.d_rollout). Lets the
        # caller apply a trick-indexed schedule: roll deep early-game where the
        # critic is blind (§1.2 partial-obs ceiling), shallow + bootstrap once the
        # value head is calibrated mid-game.
        self._d_rollout_override: int | None = None
        # Per-search non-observer seat controllers (population grounding); None
        # or a missing seat -> self.agent (pure self-play).
        self._seat_policies: dict | None = None
        self.fail = defaultdict(int)

    def _controller(self, seat: int):
        """The PPOAgent modeling ``seat`` for this search (observer and unmapped
        seats -> ``self.agent``)."""
        if self._seat_policies is None:
            return self.agent
        return self._seat_policies.get(seat, self.agent)

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
    ) -> dict:
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
        self._d_rollout_override = d_rollout
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
            agent_id: {
                player_id: memory.detach().clone()
                for player_id, memory in agent._player_memories.items()
            }
            for agent_id, agent in involved.items()
        }
        try:
            return self._search_inner(real_game, observer, forced_public)
        finally:
            for agent_id, agent in involved.items():
                agent._player_memories = {
                    player_id: memory.detach().clone()
                    for player_id, memory in saved_memories[agent_id].items()
                }
            self._d_rollout_override = None
            self._seat_policies = None

    # ------------------------------------------------------------------
    # Search driver
    # ------------------------------------------------------------------
    def _search_inner(self, real_game, observer, forced_public) -> dict:
        observer_player = real_game.players[observer - 1]
        valid_real = sorted(observer_player.get_valid_action_ids())
        head = self._infer_head(valid_real)
        config = self.config
        m_iters = config.iters[head]
        self._max_depth = config.max_depth[head]

        # Reset transient search state.
        root = _Node()
        self._nodes = [root]
        self._qmin = math.inf
        self._qmax = -math.inf
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
        the n_worlds worlds at each decision point. The per-world sequential path
        is kept as ``_build_world`` (the reference the batched build is
        validated against)."""
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
                self.fail["determinize"] += 1
        if not deals:
            return []
        return self._build_worlds_batched(real_game, deals, forced_public, observer)

    def _fresh_world(self, real_game, deal):
        """Fresh Game with the determinized hands + blind installed (pre-replay)."""
        from sheepshead import Game

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

    def _actor_probs_batched(self, encoded, states, valid_list, seat):
        """Post-mixture action probabilities (n, A) under ``seat``'s controller —
        mirrors ``get_action_probs_with_logits`` but over n worlds at once.
        ``encoded`` must come from the same controller's encoder
        (``_encode_seat_batched`` on the same seat)."""
        ctrl = self._controller(seat)
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

    def _after_action_batched(self, games, seat_memories):
        """End-of-trick observe for every seat (with its controller), batched over
        worlds (the plays are forced identically, so trick completion is
        synchronized across worlds)."""
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
            self.fail["batched_fallback"] += 1
            return self._build_pool_sequential(
                real_game, deals, forced_public, observer
            )

    def _build_pool_sequential(self, real_game, deals, forced_public, observer):
        """Per-world sequential build (the robust reference): replay each deal with
        ``_build_world`` and skip the ones that fail (returns None)."""
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
        """Lockstep batched analogue of ``_build_world`` over many worlds at once.

        Drives control flow off world 0 (all worlds share the public/private
        decision structure). Because the lockstep cannot drop a single inconsistent
        world mid-flight, any legality/desync failure raises ``_ReplayInconsistency``
        so the caller can fall back to the per-world build."""
        from collections import deque

        n_worlds = len(deals)
        games = [self._fresh_world(real_game, deal) for deal in deals]
        det_buries = [deque(deal["bury"]) for deal in deals]
        det_unders = [deal["under_card"] for deal in deals]
        public_actions = deque(forced_public)
        seat_memories = {
            seat: torch.zeros((n_worlds, 256), device=DEV) for seat in range(1, 6)
        }
        log_weights = torch.zeros(n_worlds, device=DEV)

        guard = 0
        while True:
            guard += 1
            if guard > 6000:
                self.fail["guard"] += 1
                raise _ReplayInconsistency("batched pool build guard exceeded")
            acted = False
            for pos in range(1, 6):
                ref_player = games[0].players[pos - 1]
                ref_valid = ref_player.get_valid_action_ids()
                while ref_valid:
                    # Root reached: public record exhausted and it is the
                    # observer's turn. If this is a later private decision, first
                    # force the already-taken private actions so the replay stops
                    # at the same bury/under step as the live game.
                    if (
                        not public_actions
                        and pos == observer
                        and _private_root_ready(real_game, games[0], ref_valid)
                    ):
                        pool = []
                        for i, game in enumerate(games):
                            if game.history != real_game.history:
                                self.fail["hist_mismatch"] += 1
                                continue
                            memory_snapshot = {
                                seat: seat_memories[seat][i].detach().clone()
                                for seat in range(1, 6)
                            }
                            pool.append(
                                (game, memory_snapshot, float(log_weights[i].item()))
                            )
                        return pool

                    if any(_is_private_action(action_id) for action_id in ref_valid):
                        # Forced bury/under: encode (advance memory), then act each
                        # world with its own determinized card. Not weighted.
                        self._encode_seat_batched(games, pos, seat_memories)
                        for i, game in enumerate(games):
                            world_valid = game.players[pos - 1].get_valid_action_ids()
                            action_id = self._forced_private(
                                world_valid, det_buries[i], det_unders[i]
                            )
                            if action_id is None or action_id not in world_valid:
                                self.fail["bad_private"] += 1
                                raise _ReplayInconsistency(
                                    "batched replay: bad forced private action"
                                )
                            game.players[pos - 1].act(action_id)
                    else:
                        if not public_actions or public_actions[0][0] != pos:
                            self.fail["pub_desync"] += 1
                            raise _ReplayInconsistency(
                                "batched replay: public action desync"
                            )
                        _, action_id = public_actions.popleft()
                        states, encoded = self._encode_seat_batched(
                            games, pos, seat_memories
                        )
                        # Weight bidding actions only (scheme B); plays are forced
                        # but never weighted. The actor head runs only here.
                        if not _is_play_action(action_id):
                            valid_lists = [
                                game.players[pos - 1].get_valid_action_ids()
                                for game in games
                            ]
                            probs = self._actor_probs_batched(
                                encoded, states, valid_lists, pos
                            )
                            action_probs = probs[:, action_id - 1].clamp_min(1e-8)
                            log_weights = log_weights + torch.log(action_probs)
                        for game in games:
                            world_valid = game.players[pos - 1].get_valid_action_ids()
                            if action_id not in world_valid:
                                self.fail["bad_public"] += 1
                                raise _ReplayInconsistency(
                                    "batched replay: bad forced public action"
                                )
                            game.players[pos - 1].act(action_id)

                    acted = True
                    self._after_action_batched(games, seat_memories)
                    ref_valid = ref_player.get_valid_action_ids()
            if not acted:
                self.fail["no_acted"] += 1
                raise _ReplayInconsistency("batched replay: no seat acted")

    @staticmethod
    def _pool_probs(pool):
        log_weights = np.array(
            [log_weight for _, _, log_weight in pool], dtype=np.float64
        )
        weights = np.exp(log_weights - log_weights.max())
        return (weights / weights.sum()).tolist()

    @staticmethod
    def _pool_ess(pool) -> float:
        if not pool:
            return 0.0
        log_weights = np.array(
            [log_weight for _, _, log_weight in pool], dtype=np.float64
        )
        weights = np.exp(log_weights - log_weights.max())
        total = weights.sum()
        if total <= 0:
            return 0.0
        return float(total * total / np.square(weights).sum())

    def _finalize(self, root, valid_real, head, n_used, ess) -> dict:
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
        if counts.sum() <= 0.0:
            return dict(
                pi=pi,
                ess=ess,
                ok=False,
                head=head,
                n_iter=n_used,
                valid=valid_real,
                root_n=root_n,
                root_q=root_q,
            )
        powered = np.power(counts, 1.0 / self.config.tau_target)
        powered /= powered.sum()
        for action_id, prob in zip(valid_real, powered):
            pi[action_id - 1] = prob
        ok = ess >= self.config.ess_floor
        return dict(
            pi=pi,
            ess=ess,
            ok=ok,
            head=head,
            n_iter=n_used,
            valid=valid_real,
            root_n=root_n,
            root_q=root_q,
        )

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
        """Seat (1-5) whose turn it is, or None if terminal (exactly one seat has
        legal actions at any non-terminal point)."""
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
                mem = torch.zeros((5, 256), device=DEV)
                for seat in range(1, 6):
                    if seat in memory_snapshot:
                        mem[seat - 1] = memory_snapshot[seat]
                sims.append(_Sim(world, mem, root))
            self._run_chunk(sims, observer)

    def _run_chunk(self, sims, observer):
        guard = 0
        while any(sim.phase != "done" for sim in sims):
            guard += 1
            if guard > 100000:
                raise RuntimeError("batched chunk guard exceeded")

            # 1. Resolve no-network transitions; collect this round's encode
            # requests.
            requests = []  # (sim, kind, is_tree)
            for sim in sims:
                if sim.phase == "done":
                    continue
                prepared = self._prepare(sim, observer)
                if prepared is not None:
                    requests.append((sim, prepared[0], prepared[1]))
            if not requests:
                continue

            # 2+3. Encode + actor/critic heads, grouped by the acting seat's
            # controller (population grounding). The encode COUNT matches the
            # ungrouped path exactly; only the batch grouping fragments when
            # seat_policies are present (see notebooks/Population_Grounded_Teacher_Plan.md).
            # The critic now runs only on groups that contain a bootstrap request
            # (it used to run on every row and be discarded).
            states = [
                sim.world.players[sim.seat - 1].get_state_dict()
                for sim, _, _ in requests
            ]
            groups: dict[int, tuple] = {}
            for req_idx, (sim, _, _) in enumerate(requests):
                ctrl = self._controller(sim.seat)
                groups.setdefault(id(ctrl), (ctrl, []))[1].append(req_idx)

            probs_np = np.zeros((len(requests), self.action_size), dtype=np.float32)
            values_np = np.zeros(len(requests), dtype=np.float32)
            for ctrl, req_idxs in groups.values():
                group_states = [states[req_idx] for req_idx in req_idxs]
                memory_in = torch.stack(
                    [
                        requests[req_idx][0].mem[requests[req_idx][0].seat - 1]
                        for req_idx in req_idxs
                    ]
                )
                encoded = ctrl.encoder.encode_batch(
                    group_states, memory_in=memory_in, device=DEV
                )
                memory_out = encoded["memory_out"].detach()
                for row, req_idx in enumerate(req_idxs):
                    req_sim = requests[req_idx][0]
                    req_sim.mem[req_sim.seat - 1] = memory_out[row]
                masks = torch.stack(
                    [
                        ctrl.get_action_mask(
                            requests[req_idx][0].valid, self.action_size
                        )
                        for req_idx in req_idxs
                    ]
                ).to(DEV)
                hand_ids = torch.as_tensor(
                    np.stack([states[req_idx]["hand_ids"] for req_idx in req_idxs]),
                    dtype=torch.long,
                    device=DEV,
                )
                with torch.no_grad():
                    group_probs, _ = ctrl.actor.forward_with_logits(
                        encoded, masks, hand_ids, ctrl.encoder.card
                    )
                    if any(requests[req_idx][1] == "critic" for req_idx in req_idxs):
                        values_np[req_idxs] = (
                            ctrl.critic(encoded).detach().view(-1).cpu().numpy()
                        )
                probs_np[req_idxs] = group_probs.detach().cpu().numpy()

            # 4. Apply each request; collect sims that completed a trick this round.
            completers = []
            for req_idx, (sim, kind, is_tree) in enumerate(requests):
                if kind == "critic":
                    self._finish_value(
                        sim, self._discount(float(values_np[req_idx]), sim.obs_plays)
                    )
                else:
                    self._apply_actor(
                        sim, observer, probs_np[req_idx], is_tree, completers
                    )

            # 5. End-of-trick observe for the completer subset, batched per seat.
            self._observe_completers_batched(completers)

    def _prepare(self, sim, observer):
        """Run no-network state-machine transitions until ``sim`` needs an encode
        (sets sim.seat/valid and returns (kind, is_tree)) or is done (returns None)."""
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
                return ("actor", True)
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
                        self._nodes.append(child)
                    sim.node, sim.depth, sim.pending_action = child, sim.depth + 1, None
                    sim.phase = "tree"
                    continue
                sim.seat = next_seat
                sim.valid = sorted(world.players[next_seat - 1].get_valid_action_ids())
                return ("actor", False)
            # rollout
            next_seat = self._next_actor(world)
            if next_seat is None:
                self._finish_terminal(sim, observer)
                return None
            sim.seat = next_seat
            sim.valid = sorted(world.players[next_seat - 1].get_valid_action_ids())
            rollout_depth = (
                self._d_rollout_override
                if self._d_rollout_override is not None
                else self.config.d_rollout
            )
            if (
                next_seat == observer
                and _valid_has_play(sim.valid)
                and sim.obs_plays >= rollout_depth
            ):
                return ("critic", False)
            return ("actor", False)

    def _apply_actor(self, sim, observer, probs, is_tree, completers):
        if is_tree:
            node, valid = sim.node, sim.valid
            is_root = sim.depth == 0
            explore_frac = self.config.root_explore_frac
            n_legal = len(valid)
            for action_id in valid:
                prior = float(probs[action_id - 1])
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
            following = self._is_following(sim.world, observer)
            action_id = self._select_vl(node, valid, following)
            node.vloss[action_id] = node.vloss.get(action_id, 0) + 1
            sim.path.append((node, action_id))
            sim.pending_action = action_id
            sim.world.players[observer - 1].act(action_id)
            sim.phase = "advance"
        else:
            seat, valid = sim.seat, sim.valid
            action_id = self._sample_action(probs, valid)
            is_obs_play = seat == observer and _valid_has_play(valid)
            sim.world.players[seat - 1].act(action_id)
            if sim.phase == "rollout" and is_obs_play:
                sim.obs_plays += 1
        if sim.world.was_trick_just_completed:
            completers.append(sim)
        if sim.world.is_done():
            self._finish_terminal(sim, observer)

    def _select_vl(self, node, valid, following) -> int:
        """PUCT selection with virtual loss: an in-flight selected edge is charged
        ``virtual_loss`` extra (pessimistic) visits so concurrent sims diversify."""
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
            if following:
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
        draw = self._rng.random()
        cum = 0.0
        for action_id in valid:
            cum += float(probs[action_id - 1])
            if draw <= cum:
                return action_id
        return valid[-1]

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
        sim.value = value

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
    def _after_action(self, world):
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
        """Replay the public record into a fresh game whose hidden hands are the
        sampled determinization, rebuilding every seat's recurrent memory, and
        stop at the observer's current decision (root). Returns
        ``(world, log_weight)`` or ``(None, None)`` on a replay/desync failure.

        ``log_weight`` is the sum of policy log-probs of every forced PUBLIC *bidding*
        action (pick / pass / call / alone / jd-partner) under the rebuilt
        memory + determinized hands (scheme B). Plays are forced (to rebuild
        memory and reproduce the record) but never weighted. Private bury/under
        are forced from the determinization and never weighted.
        """
        from collections import deque

        from sheepshead import Game

        world = Game(partner_selection_mode=real_game.partner_mode_flag)
        for seat in range(1, 6):
            hand = deal["initial_hands"][seat][:]
            world.players[seat - 1].hand = hand
            world.players[seat - 1].initial_hand = hand[:]
        world.blind = deal["blind"][:]

        self.agent.reset_recurrent_state()
        if self._seat_policies:
            for policy in self._seat_policies.values():
                policy.reset_recurrent_state()
        public_actions = deque(forced_public)
        det_bury = deque(deal["bury"])
        det_under = deal["under_card"]
        log_weight = 0.0
        guard = 0
        while True:
            guard += 1
            if guard > 6000:
                self.fail["guard"] += 1
                return None, None
            acted = False
            for player in world.players:
                valid = player.get_valid_action_ids()
                while valid:
                    # Root reached: all public actions forced and it is the
                    # observer's turn. For later private roots, keep replaying the
                    # determinized private actions until bury/under progress
                    # matches the live root; the simulate step encodes the root.
                    if (
                        not public_actions
                        and player.position == observer
                        and _private_root_ready(real_game, world, valid)
                    ):
                        if world.history != real_game.history:
                            self.fail["hist_mismatch"] += 1
                            return None, None
                        return world, log_weight

                    if any(_is_private_action(action_id) for action_id in valid):
                        action_id = self._forced_private(valid, det_bury, det_under)
                        if action_id is None or action_id not in valid:
                            self.fail["bad_private"] += 1
                            return None, None
                        # Advance this seat's memory through the forced decision.
                        self._controller(player.position).get_action_probs_with_logits(
                            player.get_state_dict(), valid, player_id=player.position
                        )
                        player.act(action_id)
                    else:
                        if (
                            not public_actions
                            or public_actions[0][0] != player.position
                        ):
                            self.fail["pub_desync"] += 1
                            return None, None
                        _, action_id = public_actions.popleft()
                        if action_id not in valid:
                            self.fail["bad_public"] += 1
                            return None, None
                        probs, _ = self._controller(
                            player.position
                        ).get_action_probs_with_logits(
                            player.get_state_dict(), valid, player_id=player.position
                        )
                        if not _is_play_action(action_id):
                            action_prob = float(probs[0][action_id - 1].item())
                            log_weight += math.log(max(action_prob, 1e-8))
                        player.act(action_id)
                    acted = True
                    valid = player.get_valid_action_ids()
                    self._after_action(world)
            if not acted:
                self.fail["no_acted"] += 1
                return None, None

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
