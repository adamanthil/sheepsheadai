from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


class RulesInput(BaseModel):
    """Full rules specification; unknown keys rejected."""

    model_config = ConfigDict(extra="forbid")
    partnerMode: Literal[0, 1] = 1
    # What happens when all five players pass: "leasters" plays the hand out
    # with no picker, "doublers" throws it in and redeals at double stakes.
    allPassMode: Literal["leasters", "doublers"] = "leasters"
    doubleOnTheBump: bool = True


class RulesUpdate(BaseModel):
    """Partial rules patch; only provided keys are merged."""

    model_config = ConfigDict(extra="forbid")
    partnerMode: Optional[Literal[0, 1]] = None
    allPassMode: Optional[Literal["leasters", "doublers"]] = None
    doubleOnTheBump: Optional[bool] = None


class CreateTableRequest(BaseModel):
    name: str
    rules: RulesInput = Field(default_factory=RulesInput)

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("name must not be empty")
        if len(v) > 48:
            raise ValueError("name must be at most 48 characters")
        return v


class JoinTableRequest(BaseModel):
    # Identity is derived from the Authorization header, never the body.
    display_name: str
    # The key from CreateTableResponse; the join presenting it becomes host.
    host_key: Optional[str] = None

    @field_validator("display_name")
    @classmethod
    def validate_display_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("display_name must not be empty")
        if len(v) > 32:
            raise ValueError("display_name must be at most 32 characters")
        return v


class UpdatePlayerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Optional[str] = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip()
        if not v:
            raise ValueError("name must not be empty (use null to clear)")
        if len(v) > 32:
            raise ValueError("name must be at most 32 characters")
        return v


USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{3,20}$")
# Names that would read as the site itself or as a bot. Compared
# case-insensitively; the DB enforces only the character set.
RESERVED_USERNAMES = frozenset(
    {
        "admin",
        "administrator",
        "ai",
        "anonymous",
        "bot",
        "guest",
        "host",
        "mod",
        "moderator",
        "null",
        "root",
        "sheepshead",
        "support",
        "system",
        "undefined",
    }
)


def validate_username(v: str) -> str:
    v = v.strip()
    if not USERNAME_PATTERN.match(v):
        raise ValueError("username must be 3-20 characters: letters, digits, _ or -")
    if v.lower() in RESERVED_USERNAMES:
        raise ValueError("username is reserved")
    return v


def validate_password(v: str) -> str:
    if not 8 <= len(v) <= 128:
        raise ValueError("password must be 8-128 characters")
    return v


def normalize_email(v: str) -> str:
    return v.strip().lower()


class RegisterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    username: str
    email: EmailStr
    password: str

    _username = field_validator("username")(validate_username)
    _password = field_validator("password")(validate_password)
    _email = field_validator("email", mode="after")(normalize_email)


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # Email address or username.
    login: str = Field(min_length=1, max_length=320)
    password: str = Field(min_length=1, max_length=128)


class EmailTokenRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    token: str = Field(min_length=1, max_length=128)


class ForgotPasswordRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    email: EmailStr

    _email = field_validator("email", mode="after")(normalize_email)


class ResetPasswordRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    token: str = Field(min_length=1, max_length=128)
    password: str

    _password = field_validator("password")(validate_password)


class ChangePasswordRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    current_password: str = Field(min_length=1, max_length=128)
    new_password: str

    _new_password = field_validator("new_password")(validate_password)


class AccountPublic(BaseModel):
    username: str
    email: str
    email_verified: bool
    # May open the admin area.
    is_admin: bool


class AccountMeResponse(BaseModel):
    player_id: str
    name: Optional[str]
    # None for a guest.
    account: Optional[AccountPublic]


class AccountSessionResponse(BaseModel):
    player_id: str
    name: Optional[str]
    # Present only when a new session was minted (always on login; on
    # register only for a caller with no identity yet). The client stores
    # it as its bearer token, replacing any guest token it held.
    session_token: Optional[str]
    account: AccountPublic


class UsernameAvailability(BaseModel):
    available: bool


class LeaderboardRow(BaseModel):
    rank: int
    username: str
    hands: int
    total: int
    # Score per hand; win and pick rates are fractions in [0, 1]. All
    # count an abandoned hand's win as 0 (persistence.stats).
    sph: float
    win_pct: float
    pick_pct: float
    is_you: bool


class LeaderboardResponse(BaseModel):
    # overall, one partner mode (called_ace / jd), or hands vs the AI alone.
    board: Literal["overall", "called_ace", "jd", "vs_ai"]
    sort: Literal["total", "sph", "win_pct"]
    # Every hand, or only hands closed in the current or the previous
    # month (US Central).
    period: Literal["all_time", "month", "last_month"]
    # The month's bounds, [start, end); both None for all_time. The
    # current month's board resets at period_end.
    period_start: Optional[datetime]
    period_end: Optional[datetime]
    # Hands of this board's kind needed to qualify for it (in the period).
    min_hands: int
    # The top of the board only; nothing below it is ever served.
    rows: List[LeaderboardRow]
    # The caller's own row when they qualify but rank below `rows`.
    you: Optional[LeaderboardRow]
    # Hands the signed-in caller still needs to qualify, when they aren't
    # on this board; None for anonymous callers and a finished month.
    qualifies_in: Optional[int]


class CompanySplit(BaseModel):
    hands: int
    # None until the first hand in this split.
    sph: Optional[float]
    # Half-width of a 95% interval on sph; None below
    # SPLIT_MARGIN_MIN_HANDS hands.
    sph_margin: Optional[float]
    win_pct: Optional[float]


class RoleSplit(CompanySplit):
    role: Literal["picker", "partner", "defender", "leaster"]


class PositionOffers(BaseModel):
    # Bidding order: 1 bids first, 5 last.
    position: int
    # Deals the pick reached the player from this position, and how many
    # of them they picked.
    offered_hands: int
    picks: int


class AccountStatsResponse(BaseModel):
    username: str
    hands: int
    total: int
    # None until the first finished hand.
    sph: Optional[float]
    win_pct: Optional[float]
    pick_pct: Optional[float]
    leaster_hands: int
    # Hands where the AI made more than abandon_threshold of the player's
    # decisions (excused moves aside): their losses count in full, their
    # wins as 0. forfeited_score is the winnings zeroed that way, and the
    # score figures above already exclude it.
    abandoned_hands: int
    completion_rate: Optional[float]
    forfeited_score: int
    # Hands where the AI made any of the player's decisions.
    ai_assisted_hands: int
    abandon_threshold: int
    # Rank on the default (total score) board, or None when not yet
    # eligible; then qualifies_in says how many more hands it takes.
    rank: Optional[int]
    qualifies_in: Optional[int]
    min_hands: int
    # Hands where the other four seats were all dealt to and played by the
    # AI, and the rest.
    vs_ai: CompanySplit
    with_people: CompanySplit
    split_margin_min_hands: int
    # Picks the player made themselves, and the trump and queens among
    # the six cards they were dealt on those hands (None without picks).
    picks: int
    trump_per_pick: Optional[float]
    queens_per_pick: Optional[float]
    # Deals where the pick reached the player (everyone up to the picker,
    # and every seat of a leaster or pass-out), and the share of them they
    # picked; pick_pct is picks over every deal instead.
    offered_hands: int
    pick_offered_pct: Optional[float]
    # The same, by bidding position, 1 to 5.
    pick_offers_by_position: List[PositionOffers]
    # Score/hand and win rate by the player's part in the hand, in the
    # order picker, partner, defender, leaster.
    roles: List[RoleSplit]


class AccountRunningScoreResponse(BaseModel):
    # Every finished hand, oldest first, as parallel lists: the counted
    # score (an abandoned win as 0), the player's role, the doublers
    # stake, and when it closed (Unix seconds).
    scores: List[int]
    roles: List[Literal["picker", "partner", "defender", "leaster"]]
    multipliers: List[int]
    closed: List[int]


class HandOpponents(BaseModel):
    humans: int
    ai: int


ReviewStatus = Literal["queued", "reviewed", "skipped", "failed"]
DecisionKind = Literal["pick", "call", "under", "bury", "play"]


class HandReviewBrief(BaseModel):
    status: ReviewStatus
    # Expected score the player's choices cost and gained against the AI's,
    # both >= 0 (gained - cost is the hand's net); None until reviewed.
    cost: Optional[float]
    gained: Optional[float]
    # One of the hand's decisions cost or gained a full point and its net
    # came out a full point the same way: the hand stands out in a list.
    notable: bool


class HighTrump(BaseModel):
    # The highest trump not yet played at the decision, as far as the seat
    # could know; None once every trump is gone.
    card: Optional[str]
    held: bool
    # Trump neither played nor in the seat's hand.
    unseen: int


class DecisionReviewOut(BaseModel):
    index: int
    kind: DecisionKind
    # 0-based trick for a card play.
    trick: Optional[int]
    # Card codes: the seat's hand, and the [seat, card] plays already made
    # to the trick.
    hand: List[str]
    table: List[Tuple[int, str]]
    # Engine action names; a bury is "BURY <card> <card>".
    action: str
    ai_action: str
    # [action, probability], the AI's most likely first.
    policy: List[Tuple[str, float]]
    # even: under a quarter point either way; slight: a real difference
    # short of a full point; costly / better: a point or more.
    standing: Literal["forced", "agreed", "even", "slight", "costly", "better"]
    # The AI's expected score at the decision: from the seat's view, and
    # with every card known (None for a model without that estimate).
    expected_score: float
    oracle_score: Optional[float]
    # Expected score after the player's choice and after the AI's, and the
    # first less the second as displayed; None where they agreed.
    chosen_score: Optional[float]
    ai_score: Optional[float]
    difference: Optional[float]
    high_trump: HighTrump
    # The table could tell who the partner was by this decision: the card
    # that names them had been played.
    partner_known: bool


class HandSeat(BaseModel):
    seat: int
    is_ai: bool
    is_you: bool
    # Display name; None for an AI seat.
    name: Optional[str]
    role: Literal["picker", "partner", "defender", "leaster"]


class HandReviewSummary(BaseModel):
    # Decisions where the player had a choice, by how they compare.
    decisions: int
    agreed: int
    even: int
    slight: int
    costly: int
    better: int
    cost: float
    gained: float
    # index of the decision that sums the hand up, if any reaches a full
    # point: the costliest, else the best gain.
    headline_index: Optional[int]
    # As HandReviewBrief.notable.
    notable: bool


class HandReviewResponse(BaseModel):
    hand: HandHistoryRow
    seat: int
    called_card: Optional[str]
    # The deal as the pick analyzer takes it: the six cards the seat was
    # dealt, the blind, and the partner mode (0 = jack of diamonds, 1 =
    # called ace).
    dealt_hand: List[str]
    blind: List[str]
    partner_mode: int
    seats: List[HandSeat]
    # None unless the review is done.
    summary: Optional[HandReviewSummary]
    decisions: List[DecisionReviewOut]
    # Label of the AI model that produced the review.
    ai_model: Optional[str]
    newer_game_id: Optional[UUID]
    older_game_id: Optional[UUID]


class ReviewKindRow(BaseModel):
    kind: Literal["pick", "call", "bury", "play"]
    decisions: int
    agreed: int
    # Decisions of this kind that cost, and that gained, a full point.
    costly: int
    better: int
    net: float


class ReviewDifference(BaseModel):
    game_id: UUID
    index: int
    kind: DecisionKind
    trick: Optional[int]
    # True when the play led the trick.
    led: bool
    action: str
    ai_action: str
    # How likely the AI was to make its own choice.
    ai_probability: float
    difference: float
    time_closed: datetime
    table_name: str


class ReviewPick(BaseModel):
    game_id: UUID
    time_closed: datetime
    table_name: str
    seat: int
    hand: List[str]
    picked: bool
    # How likely the AI was to pick.
    ai_pick_probability: float
    pick_score: float
    pass_score: float
    # What the hand scored for the player.
    score: int


class AccountReviewResponse(BaseModel):
    hands_reviewed: int
    hands_waiting: int
    decisions: int
    agreed: int
    even: int
    slight: int
    costly: int
    better: int
    net: float
    by_kind: List[ReviewKindRow]
    # Largest first, at most REVIEW_LIST_SIZE each.
    cost_most: List[ReviewDifference]
    gained_most: List[ReviewDifference]
    pick_decisions: int
    pick_agreed: int
    picked_ai_passes: List[ReviewPick]
    passed_ai_picks: List[ReviewPick]


class ReviewQueueFailure(BaseModel):
    game_id: UUID
    seat: int
    reason: Optional[str]


class ReviewQueueResponse(BaseModel):
    paused: bool
    tables_in_play: int
    waiting: int
    oldest_queued: Optional[datetime]
    reviewed_24h: int
    average_ms_24h: Optional[int]
    slowest_ms_24h: Optional[int]
    reviewed: int
    skipped: int
    failed: int
    total: int
    last_reviewed: Optional[datetime]
    ai_model: str
    # Reviews made by an AI model other than the one now deployed.
    other_model_reviews: int
    failures: List[ReviewQueueFailure]


class HandHistoryRow(BaseModel):
    game_id: UUID
    time_closed: datetime
    table_name: str
    role: Literal["picker", "partner", "defender", "leaster"]
    # The picker played alone (no partner) this hand.
    alone: bool
    multiplier: int
    # The hand's result, and what stats count it as: 0 for an abandoned win.
    score: int
    counted_score: int
    # Card points taken: the player's team's (with the bury, as the engine
    # scores it) or, in a leaster, the player's own.
    points_taken: int
    points_scope: Literal["team", "own"]
    ai_assisted: bool
    abandoned: bool
    # The other four seats of the hand, by who was dealt them.
    opponents: HandOpponents
    # Where the hand's AI review stands; None if it was never queued.
    review: Optional[HandReviewBrief]


class HandHistoryResponse(BaseModel):
    hands: List[HandHistoryRow]
    # Pass as ?before= for the next (older) page; None on the last page.
    next_cursor: Optional[str]


class UpdateTableRulesRequest(BaseModel):
    client_id: str
    rules: RulesUpdate


class StartGameRequest(BaseModel):
    client_id: str
    seed: Optional[int] = None


class ActionRequest(BaseModel):
    client_id: str
    action_id: int


class SeatRequest(BaseModel):
    client_id: str
    seat: int


class NextHandRequest(BaseModel):
    client_id: str


class KickRequest(BaseModel):
    client_id: str  # the host's own connection
    target_client_id: str


class AfterHandRequest(BaseModel):
    client_id: str
    # False takes the request back.
    on: bool = True


class CloseTableRequest(BaseModel):
    client_id: str


class LeaveRequest(BaseModel):
    client_id: str


class ScoreColumn(BaseModel):
    # A player's own column, or the occupant id of an AI's (runtime.scores).
    id: str
    name: str
    # The AI's own column, not a player's: tells an AI apart from a player
    # who shares its name.
    ai: bool = False


class ScoreSegment(BaseModel):
    # Carried-over columns in their old places, the rest in seat order at
    # the block's first deal (runtime.scores).
    columns: List[ScoreColumn]


class TablePublic(BaseModel):
    id: str
    name: str
    status: Literal["open", "playing", "finished"]
    rules: RulesInput
    # Stake the hand in progress is being played for: 1 normally, doubling
    # with each consecutive passed-out deal at a doublers table, up to a cap.
    scoreMultiplier: int = 1
    # Consecutive passed-out deals behind the hand in progress.
    passOutCount: int = 0
    seats: Dict[int, Optional[str]] | Dict[str, Optional[str]]
    runningBySeat: Dict[int, int] | Dict[str, int]
    seatOccupants: Dict[int, Optional[str]] | Dict[str, Optional[str]]
    seatIsAI: Dict[int, bool] | Dict[str, bool]
    # Verified account username of each seat's human, for the account badge.
    seatAccount: Dict[int, Optional[str]] | Dict[str, Optional[str]]
    # Whether another player may move into each seat now (Table.seat_takeable).
    seatTakeable: Dict[int, bool] | Dict[str, bool]
    seatLeavingAfterHand: Dict[int, bool] | Dict[str, bool]
    closingAfterHand: bool
    host: Optional[str]
    # Verified account username of the host, as seatAccount, or None.
    hostAccount: Optional[str] = None
    resultsHistory: List[Dict[str, Any]]
    # Blocks of hands played by one set of players (runtime.scores).
    scoreSegments: List[ScoreSegment]


class CreateTableResponse(TablePublic):
    # One-time key for the creator's own /join, so nobody who spots the new
    # table in the public list can join first and take host.
    host_key: str


class JoinTableResponse(BaseModel):
    client_id: str
    player_id: str
    # Present only when a fresh identity was minted on this join.
    session_token: Optional[str]
    is_host: bool
    table: TablePublic


class PlayerPublic(BaseModel):
    player_id: str
    name: Optional[str]


class RulesUpdateResponse(BaseModel):
    status: str
    rules: RulesInput


class OkResponse(BaseModel):
    ok: bool


class ActionsLookupResponse(BaseModel):
    action_lookup: Dict[int, str]


class HealthResponse(BaseModel):
    status: str
    db: bool
    model_label: str = Field(alias="model")
    # The deployment sampling (SamplingSpec.label()), so a re-temper is
    # verifiable from outside like a model swap.
    sampling: str

    model_config = ConfigDict(populate_by_name=True)


# Analyze AI Model schemas
class AnalyzeCardEmbeddingEntry(BaseModel):
    id: int  # card id (1..32 real cards in DECK order, 33 = UNDER)
    card: str  # card code, or "UNDER"
    vector: List[float]


class AnalyzeCardEmbeddings(BaseModel):
    """The model's learned card-embedding table and precomputed geometry.
    Row order is DECK order (all trump first, then fail), UNDER last."""

    dims: int
    cards: List[AnalyzeCardEmbeddingEntry]
    cosineSim: List[List[float]]  # (n, n), order matches `cards`
    pcaCoords: List[List[float]]  # (n, 2) projection onto top-2 PCs
    pcaExplainedVariance: List[float]  # variance ratio of the 2 PCs


class AnalyzeModelResponse(BaseModel):
    modelLabel: str
    arch: str
    criticMode: str
    hasAuxHeads: bool
    hasOracle: bool
    gamma: float
    # None for architectures without a card-embedding table (onehot-ff).
    cardEmbeddings: Optional[AnalyzeCardEmbeddings] = None


class AnalyzeSimulateMeta(BaseModel):
    """Echo of the simulate request plus the model attributes the UI needs
    to interpret the trace."""

    model_config = ConfigDict(populate_by_name=True)

    partnerMode: int
    deterministic: bool
    # How actions were chosen: "argmax", the deployed sampling's label
    # (e.g. "play=0.5 band=1"), or "raw policy (T=1)" before tuning.
    sampling: str
    seed: Optional[int] = None
    model_label: str = Field(alias="model")
    gamma: float
    criticMode: str
    hasOracle: bool


class AnalyzePickMeta(BaseModel):
    """Echo of the pick request plus the serving model's label."""

    model_config = ConfigDict(populate_by_name=True)

    partnerMode: int
    seat: int
    seed: Optional[int] = None
    deterministic: bool
    sampling: str
    model_label: str = Field(alias="model")


class AnalyzeSimulateRequest(BaseModel):
    # extra="forbid" so removed fields (e.g. the old client-supplied
    # modelPath, which let callers point torch.load at arbitrary files)
    # are rejected with a 422 instead of silently ignored.
    model_config = ConfigDict(extra="forbid")
    seed: Optional[int] = None
    partnerMode: int = 1  # 0 = JD, 1 = Called Ace
    deterministic: bool = False
    maxSteps: int = 200


class AnalyzeProbability(BaseModel):
    actionId: int
    action: str
    prob: float
    logit: float


class AnalyzePointEstimate(BaseModel):
    seat: int
    seatName: str
    points: float
    relativePosition: int


class AnalyzeTrumpSeenMaskEntry(BaseModel):
    card: str
    probabilitySeen: float
    actualSeen: bool


class AnalyzeActionDetail(BaseModel):
    stepIndex: int
    seat: int
    seatName: str
    phase: str  # "pick" | "partner" | "bury" | "play"
    actionId: int
    action: str
    valueEstimate: float
    # Privileged (full-information) critic value, only when the loaded
    # checkpoint was trained with critic_mode="oracle". Diagnostic: the gap
    # to valueEstimate shows how much hidden information changes the value.
    oracleValue: Optional[float] = None
    discountedReturn: Optional[float] = None
    stepReward: Optional[float] = None
    stepRewardBase: Optional[float] = None
    stepRewardHeadShaping: Optional[float] = None
    # Terminal-only per-step reward (no shaping / trick rewards / leaster bonus),
    # matching reward_mode="terminal" in the league trainer.
    stepRewardTerminal: Optional[float] = None
    winProb: Optional[float] = None  # [0,1]
    expectedFinalReturn: Optional[float] = None  # unscaled, undiscounted
    secretPartnerProb: Optional[float] = None  # [0,1]
    pointEstimates: Optional[List[AnalyzePointEstimate]] = None
    pointActuals: Optional[List[AnalyzePointEstimate]] = None
    trumpSeenMask: Optional[List[AnalyzeTrumpSeenMaskEntry]] = None
    unseenTrumpHigherThanHandProb: Optional[float] = None  # [0,1]
    unseenTrumpHigherThanHandActual: Optional[bool] = None
    # Recurrent-memory update magnitude at this decision: cosine distance
    # between the actor's GRU memory before and after their own encode.
    # Measured only at the actor's decisions (trick-completion observes
    # update memory too but are not sampled here). None on the seat's first
    # encode (memory is zeros).
    memoryCosineDistance: Optional[float] = None
    memoryNorm: Optional[float] = None
    validActionIds: List[int]
    probabilities: List[AnalyzeProbability]
    view: Dict[str, Any]


class AnalyzeSeatCalibration(BaseModel):
    seat: int
    seatName: str
    won: bool  # final score > 0
    decisionCount: int
    firstWinProb: float
    lastWinProb: float
    meanWinProb: float
    brierScore: float  # mean (winProb - won)^2 over this seat's decisions
    # Mean |predicted - actual| of predictions ABOUT this seat, against the
    # points known at each decision (the head's training target), not the
    # final totals.
    pointsMae: float


class AnalyzeCalibrationSummary(BaseModel):
    """Game-level rollup of aux-head prediction quality, computed from the
    trace once the game is done. Absent for no-aux architectures."""

    seats: List[AnalyzeSeatCalibration]
    overallBrier: float
    overallPointsMae: float
    # Fraction of (decision x trump card) predictions where
    # (probabilitySeen > 0.5) matches the seen/unseen ground truth.
    trumpMaskAccuracy: Optional[float] = None
    trumpMaskCount: int = 0


class AnalyzePickRequest(BaseModel):
    # extra="forbid" for the same reason as AnalyzeSimulateRequest.
    model_config = ConfigDict(extra="forbid")
    seed: Optional[int] = None
    partnerMode: int = 1  # 0 = JD, 1 = Called Ace
    seat: int = 1  # 1..5; seats before it are forced to pass
    # Cards guaranteed to be in the target seat's hand / the blind (up to
    # 6 / 2); the remainder is dealt randomly. None or [] = fully random.
    hand: Optional[List[str]] = None
    blind: Optional[List[str]] = None
    deterministic: bool = False  # off = the deployed (tuned) sampling


class AnalyzePickScenario(BaseModel):
    seat: int
    seatName: str
    hand: List[str]  # the 6 cards the target seat holds at the pick decision
    blind: List[str]


class AnalyzePickOutcome(BaseModel):
    """Where the pre-play phases ended up once play was about to start."""

    pickerSeat: Optional[int] = None  # None = everyone passed (leaster)
    pickerName: Optional[str] = None
    isLeaster: bool
    aloneCalled: bool
    calledCard: Optional[str] = None
    calledUnder: bool
    underCard: Optional[str] = None
    bury: List[str]


class AnalyzePickResponse(BaseModel):
    meta: AnalyzePickMeta
    scenario: AnalyzePickScenario
    # One entry per pre-play decision (pick/pass, call, under, bury), same
    # shape as the simulate trace; reward/return fields stay None since no
    # full game is played.
    decisions: List[AnalyzeActionDetail]
    outcome: AnalyzePickOutcome


class AnalyzeMemoryObserve(BaseModel):
    """Memory update from a trick-completion observation.

    After every completed trick, all five seats fold the finished trick
    into their recurrent memory via agent.observe — the same GRU update as
    a decision encode, but without an action. These carry the trick's
    outcome, so they are often the largest belief revisions."""

    afterStepIndex: int  # the observe happened right after this trace step
    trick: int  # 0-indexed completed trick
    seat: int
    seatName: str
    memoryCosineDistance: Optional[float] = None  # None if memory was zeros
    memoryNorm: float


class AnalyzeSimulateResponse(BaseModel):
    meta: AnalyzeSimulateMeta
    calibration: Optional[AnalyzeCalibrationSummary] = None
    trace: List[AnalyzeActionDetail]
    memoryObserves: List[AnalyzeMemoryObserve] = []
    final: Optional[Dict[str, Any]] = None
