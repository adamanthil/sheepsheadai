"""Stage B: single-observer ISMCTS soft-teacher engine.

A search-derived, per-state, outcome-grounded teacher that produces a soft target
``pi'(a) ∝ N(a)^(1/tau_target)`` over the training agent's information set at any
decision point (pick / partner / bury / play). It is the principled replacement
for the hand-crafted exploration/shaping controllers (see
``ISMCTS_Teacher_Refactor_Plan.md``).

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
  plies (all seats sampled from ``pi_theta``) then a ``value_trunk`` V-bootstrap;
  a world that ends first contributes the observer's true terminal score. Values
  are in the critic's units (game score / 12, i.e. ~[-1, 1]), so the AlphaZero
  ``c_puct = 1.25`` is calibrated without extra Q-normalization.
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

import ppo
from sheepshead import (
    ACTIONS,
    ACTION_IDS,
)

DEV = ppo.device

# Per-head iteration budgets and tree depths (plan §3).
_DEFAULT_ITERS = {"pick": 48, "partner": 64, "bury": 96, "play": 96}
_DEFAULT_DEPTH = {"pick": 1, "partner": 1, "bury": 1, "play": 6}

# Critic / reward scale: PPO trains the value head on final_score / 12, so a
# terminal game score is divided by the same factor to share the critic's units.
_RETURN_SCALE = 12.0


def _is_play_action(action_id: int) -> bool:
    return ACTIONS[action_id - 1].startswith("PLAY ")


def _is_private_action(action_id: int) -> bool:
    name = ACTIONS[action_id - 1]
    return name.startswith("BURY ") or name.startswith("UNDER ")


def _valid_has_play(valid) -> bool:
    return any(_is_play_action(a) for a in valid)


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


class _Node:
    """Statistics-only ISMCTS node, keyed (implicitly, by tree position) on the
    observer's action sequence. All counts are *weighted* by the per-iteration
    determinization importance weight."""

    __slots__ = ("children", "N", "W", "P", "avail", "visited")

    def __init__(self):
        self.children: dict[int, _Node] = {}
        self.N: dict[int, float] = {}
        self.W: dict[int, float] = {}
        self.P: dict[int, float] = {}
        self.avail: dict[int, float] = {}
        self.visited: bool = False


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
        self.fail = defaultdict(int)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def search(
        self, real_game, observer: int, forced_public, rng, d_rollout: int | None = None
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

        Returns
        -------
        dict with keys: ``pi`` (float32[action_size] soft target), ``ess``,
        ``ok`` (ESS >= floor and statistics present), ``head``, ``n_iter``
        (worlds successfully built), ``valid`` (root legal action ids),
        ``root_n`` / ``root_q`` (per-action weighted visit count / mean value).
        """
        self._rng = rng
        self._d_rollout_override = d_rollout
        saved_mem = {pid: t.detach().clone() for pid, t in self.agent._player_memories.items()}
        try:
            return self._search_inner(real_game, observer, forced_public)
        finally:
            self.agent._player_memories = {
                pid: t.detach().clone() for pid, t in saved_mem.items()
            }
            self._d_rollout_override = None

    # ------------------------------------------------------------------
    # Search driver
    # ------------------------------------------------------------------
    def _search_inner(self, real_game, observer, forced_public) -> dict:
        observer_player = real_game.players[observer - 1]
        valid_real = sorted(observer_player.get_valid_action_ids())
        head = self._infer_head(valid_real)
        cfg = self.config
        m_iters = cfg.iters[head]
        self._max_depth = cfg.max_depth[head]

        # Reset transient search state.
        root = _Node()
        self._nodes = [root]
        self._qmin = math.inf
        self._qmax = -math.inf
        self.fail = defaultdict(int)

        # Build a belief-weighted pool of determinized worlds, then run the tree
        # search by SAMPLING worlds from the pool ~ exp(log_w) (scheme-B bidding
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
            for k in indices:
                world = copy.deepcopy(pool[k][0])
                self._restore_pool_memory(pool[k][1])
                self._simulate(root, world, observer, 0)

        return self._finalize(root, valid_real, head, len(pool), ess)

    def _build_pool(self, real_game, observer, forced_public, k):
        """Sample up to ``k`` determinized worlds, each rebuilt to the root by
        forced replay. Returns a list of ``(game_at_root, memory_snapshot,
        log_w)``; ``log_w`` is the scheme-B bidding log-likelihood."""
        cfg = self.config
        pool = []
        for _ in range(k):
            try:
                deal = real_game.sample_determinization(
                    observer, self._rng, max_tries=cfg.det_max_tries
                )
            except RuntimeError:
                self.fail["determinize"] += 1
                continue
            world, log_w = self._build_world(real_game, deal, forced_public, observer)
            if world is None:
                continue
            mem = {
                pid: t.detach().clone()
                for pid, t in self.agent._player_memories.items()
            }
            pool.append((world, mem, log_w))
        return pool

    def _restore_pool_memory(self, mem):
        self.agent._player_memories = {
            pid: t.detach().clone() for pid, t in mem.items()
        }

    @staticmethod
    def _pool_probs(pool):
        lw = np.array([w for _, _, w in pool], dtype=np.float64)
        w = np.exp(lw - lw.max())
        return (w / w.sum()).tolist()

    @staticmethod
    def _pool_ess(pool) -> float:
        if not pool:
            return 0.0
        lw = np.array([w for _, _, w in pool], dtype=np.float64)
        w = np.exp(lw - lw.max())
        s = w.sum()
        if s <= 0:
            return 0.0
        return float(s * s / np.square(w).sum())

    def _finalize(self, root, valid_real, head, n_used, ess) -> dict:
        pi = np.zeros(self.action_size, dtype=np.float32)
        counts = np.array([root.N.get(a, 0.0) for a in valid_real], dtype=np.float64)
        root_n = {a: float(root.N.get(a, 0.0)) for a in valid_real}
        root_q = {
            a: (float(root.W[a] / root.N[a]) if root.N.get(a, 0.0) > 0 else 0.0)
            for a in valid_real
        }
        if counts.sum() <= 0.0:
            return dict(pi=pi, ess=ess, ok=False, head=head, n_iter=n_used,
                        valid=valid_real, root_n=root_n, root_q=root_q)
        powered = np.power(counts, 1.0 / self.config.tau_target)
        powered /= powered.sum()
        for a, p in zip(valid_real, powered):
            pi[a - 1] = p
        ok = ess >= self.config.ess_floor
        return dict(pi=pi, ess=ess, ok=ok, head=head, n_iter=n_used,
                    valid=valid_real, root_n=root_n, root_q=root_q)

    @staticmethod
    def _infer_head(valid) -> str:
        names = [ACTIONS[a - 1] for a in valid]
        if any(n in ("PICK", "PASS") for n in names):
            return "pick"
        if any(n == "ALONE" or n == "JD PARTNER" or n.startswith("CALL ") for n in names):
            return "partner"
        if any(n.startswith("BURY ") or n.startswith("UNDER ") for n in names):
            return "bury"
        return "play"

    # ------------------------------------------------------------------
    # One simulation (recursive descent over observer decision nodes)
    # ------------------------------------------------------------------
    def _simulate(self, node, world, observer, depth) -> float:
        obs_player = world.players[observer - 1]
        valid = sorted(obs_player.get_valid_action_ids())
        if not valid:
            # Defensive: advancement should always stop at an observer decision.
            return self._critic_value(world, observer)

        # Encode the observer's state (advances its recurrent memory for this
        # world) and refresh priors. The observation depends only on the
        # observer's own hand + public record, but the in-tree opponent plays
        # differ per world, so deeper-node priors genuinely differ -> refresh.
        probs = self._observer_probs(world, observer)
        following = self._is_following(world, observer)
        is_root = depth == 0
        frac = self.config.root_explore_frac
        n_legal = len(valid)
        for a in valid:
            p = float(probs[a - 1])
            if is_root and frac > 0.0:
                p = (1.0 - frac) * p + frac / n_legal
            node.P[a] = p
            node.N.setdefault(a, 0.0)
            node.W.setdefault(a, 0.0)
            node.avail.setdefault(a, 0.0)
            node.avail[a] += 1.0

        # Leaf: first visit, or depth cap reached -> evaluate by rollout.
        if not node.visited or depth >= self._max_depth:
            node.visited = True
            return self._rollout(world, observer)
        node.visited = True

        a = self._select(node, valid, following)
        obs_player.act(a)
        self._after_action(world)
        if world.is_done():
            v = obs_player.get_score() / _RETURN_SCALE
        else:
            self._advance_opponents(world, observer)
            if world.is_done():
                v = obs_player.get_score() / _RETURN_SCALE
            else:
                child = node.children.get(a)
                if child is None:
                    child = _Node()
                    node.children[a] = child
                    self._nodes.append(child)
                v = self._simulate(child, world, observer, depth + 1)

        node.N[a] += 1.0
        node.W[a] += v
        q = node.W[a] / node.N[a]
        if q < self._qmin:
            self._qmin = q
        if q > self._qmax:
            self._qmax = q
        return v

    def _select(self, node, valid, following) -> int:
        c = self.config.c_puct
        total_n = sum(node.N[a] for a in valid)
        sqrt_total = math.sqrt(total_n + 1.0)
        # MuZero-style min-max normalization maps Q into [0, 1] so the AlphaZero
        # c_puct = 1.25 is calibrated regardless of the (small, score/12-scaled)
        # action-value gaps. Without it the prior-weighted exploration term
        # swamps the EV signal and visits track the policy.
        qmin, qmax = self._qmin, self._qmax
        has_span = qmax > qmin
        span = (qmax - qmin) if has_span else 1.0
        best_a, best_u = valid[0], -math.inf
        for a in valid:
            n = node.N[a]
            if n > 0:
                q_norm = (node.W[a] / n - qmin) / span if has_span else 0.5
            else:
                q_norm = self.config.fpu  # first-play urgency (optimistic)
            if following:
                explore = c * node.P[a] * math.sqrt(node.avail[a]) / (1.0 + n)
            else:
                explore = c * node.P[a] * sqrt_total / (1.0 + n)
            u = q_norm + explore
            if u > best_u:
                best_u, best_a = u, a
        return best_a

    # ------------------------------------------------------------------
    # Rollout / leaf evaluation
    # ------------------------------------------------------------------
    def _rollout(self, world, observer) -> float:
        """Truncated rollout: sample all seats from pi_theta for ``d_rollout``
        further observer *play* plies, then bootstrap with the critic. A world
        that terminates first returns the observer's true score."""
        d_rollout = (
            self._d_rollout_override
            if self._d_rollout_override is not None
            else self.config.d_rollout
        )
        obs_plays = 0
        while not world.is_done():
            for player in world.players:
                valid = player.get_valid_action_ids()
                while valid:
                    is_obs = player.position == observer
                    obs_play_here = is_obs and _valid_has_play(valid)
                    if obs_play_here and obs_plays >= d_rollout:
                        return self._critic_value(world, observer)
                    a, _, _ = self.agent.act(
                        player.get_state_dict(), valid, player.position
                    )
                    if obs_play_here:
                        obs_plays += 1
                    player.act(a)
                    valid = player.get_valid_action_ids()
                    self._after_action(world)
        return world.players[observer - 1].get_score() / _RETURN_SCALE

    def _critic_value(self, world, observer) -> float:
        player = world.players[observer - 1]
        state = player.get_state_dict()
        mem_in = self.agent.get_recurrent_memory(observer, device=DEV)
        enc = self.agent.encoder.encode_batch(
            [state], memory_in=mem_in.unsqueeze(0), device=DEV
        )
        self.agent.set_recurrent_memory(observer, enc["memory_out"][0])
        with torch.no_grad():
            v = self.agent.critic(enc)
        return float(v.item())

    # ------------------------------------------------------------------
    # World advancement helpers
    # ------------------------------------------------------------------
    def _advance_opponents(self, world, observer):
        """Sample non-observer seats from pi_theta until the observer is to act
        again or the game ends."""
        while not world.is_done():
            for player in world.players:
                valid = player.get_valid_action_ids()
                while valid:
                    if player.position == observer:
                        return
                    a, _, _ = self.agent.act(
                        player.get_state_dict(), valid, player.position
                    )
                    player.act(a)
                    valid = player.get_valid_action_ids()
                    self._after_action(world)
            # No seat acted and observer not pending -> guard against a stall.
            if not any(world.players[s].get_valid_action_ids() for s in range(5)):
                return

    def _after_action(self, world):
        if world.was_trick_just_completed:
            for seat in world.players:
                self.agent.observe(
                    seat.get_last_trick_state_dict(), player_id=seat.position
                )

    def _observer_probs(self, world, observer) -> np.ndarray:
        player = world.players[observer - 1]
        valid = player.get_valid_action_ids()
        probs_t, _ = self.agent.get_action_probs_with_logits(
            player.get_state_dict(), valid, player_id=observer
        )
        return probs_t[0].detach().cpu().numpy()

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
        ``(world, log_w)`` or ``(None, None)`` on a replay/desync failure.

        ``log_w`` is the sum of policy log-probs of every forced PUBLIC *bidding*
        action (pick / pass / call / alone / jd-partner) under the rebuilt
        memory + determinized hands (scheme B). Plays are forced (to rebuild
        memory and reproduce the record) but never weighted. Private bury/under
        are forced from the determinization and never weighted.
        """
        from collections import deque

        from sheepshead import Game

        g = Game(partner_selection_mode=real_game.partner_mode_flag)
        for s in range(1, 6):
            h = deal["initial_hands"][s][:]
            g.players[s - 1].hand = h
            g.players[s - 1].initial_hand = h[:]
        g.blind = deal["blind"][:]

        self.agent.reset_recurrent_state()
        pub = deque(forced_public)
        det_bury = deque(deal["bury"])
        det_under = deal["under_card"]
        log_w = 0.0
        guard = 0
        while True:
            guard += 1
            if guard > 6000:
                self.fail["guard"] += 1
                return None, None
            acted = False
            for player in g.players:
                valid = player.get_valid_action_ids()
                while valid:
                    # Root reached: all public actions forced and it is the
                    # observer's turn. Leave its memory at the pre-decision state
                    # (the simulate step encodes the root itself).
                    if not pub and player.position == observer:
                        if g.history != real_game.history:
                            self.fail["hist_mismatch"] += 1
                            return None, None
                        return g, log_w

                    if any(_is_private_action(a) for a in valid):
                        aid = self._forced_private(valid, det_bury, det_under)
                        if aid is None or aid not in valid:
                            self.fail["bad_private"] += 1
                            return None, None
                        # Advance this seat's memory through the forced decision.
                        self.agent.get_action_probs_with_logits(
                            player.get_state_dict(), valid, player_id=player.position
                        )
                        player.act(aid)
                    else:
                        if not pub or pub[0][0] != player.position:
                            self.fail["pub_desync"] += 1
                            return None, None
                        _, aid = pub.popleft()
                        if aid not in valid:
                            self.fail["bad_public"] += 1
                            return None, None
                        probs_t, _ = self.agent.get_action_probs_with_logits(
                            player.get_state_dict(), valid, player_id=player.position
                        )
                        if not _is_play_action(aid):
                            p_a = float(probs_t[0][aid - 1].item())
                            log_w += math.log(max(p_a, 1e-8))
                        player.act(aid)
                    acted = True
                    valid = player.get_valid_action_ids()
                    self._after_action(g)
            if not acted:
                self.fail["no_acted"] += 1
                return None, None

    @staticmethod
    def _forced_private(valid, det_bury, det_under):
        is_under = any(ACTIONS[a - 1].startswith("UNDER ") for a in valid)
        if is_under:
            if det_under is None:
                return None
            return ACTION_IDS.get(f"UNDER {det_under}")
        if not det_bury:
            return None
        return ACTION_IDS.get(f"BURY {det_bury.popleft()}")
