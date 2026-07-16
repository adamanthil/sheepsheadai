from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RulesInput(BaseModel):
    """Full rules specification; unknown keys rejected."""

    model_config = ConfigDict(extra="forbid")
    partnerMode: Literal[0, 1] = 1
    doubleOnTheBump: bool = True


class RulesUpdate(BaseModel):
    """Partial rules patch; only provided keys are merged."""

    model_config = ConfigDict(extra="forbid")
    partnerMode: Optional[Literal[0, 1]] = None
    doubleOnTheBump: Optional[bool] = None


class CreateTableRequest(BaseModel):
    name: str
    fillWithAI: bool = True
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


class RedealRequest(BaseModel):
    # Host-triggered; require client_id for authorization
    client_id: Optional[str] = None


class CloseTableRequest(BaseModel):
    client_id: str


class TablePublic(BaseModel):
    id: str
    name: str
    status: Literal["open", "playing", "finished"]
    rules: RulesInput
    fillWithAI: bool
    seats: Dict[int, Optional[str]] | Dict[str, Optional[str]]
    runningBySeat: Dict[int, int] | Dict[str, int]
    seatOccupants: Dict[int, Optional[str]] | Dict[str, Optional[str]]
    seatIsAI: Dict[int, bool] | Dict[str, bool]
    host: Optional[str]
    resultsHistory: List[Dict[str, Any]]
    initialSeatOrder: List[str]
    initialNames: Dict[str, str]


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


class AnalyzeObservationTrickSlot(BaseModel):
    seat: int  # absolute seat (1..5)
    seatName: str
    relativePosition: int  # 1 = actor, 2 = left-hand opponent, ... 5
    card: Optional[str] = None  # None = empty slot; "UNDER" = face-down under
    isPicker: bool
    isPartnerKnown: bool


class AnalyzeObservation(BaseModel):
    """The observation the model actually received (Player.get_state_dict),
    decoded from card ids to card codes. Blind/bury are empty unless the
    actor is the picker — this mirrors exactly what the actor sees, unlike
    the omniscient `view`."""

    partnerMode: int  # 0 = JD, 1 = Called Ace
    isLeaster: bool
    playStarted: bool
    currentTrick: int  # 0-indexed trick number
    aloneCalled: bool
    calledUnder: bool
    calledCard: Optional[str] = None
    pickerRel: int  # picker's relative seat (1 = actor .. 5; 0 = none yet)
    partnerRel: int  # known partner's relative seat (0 = unknown/none)
    leaderRel: int  # trick leader's relative seat (0 before play starts)
    pickerPosition: int  # absolute picker seat (0 = none yet)
    hand: List[str]
    blind: List[str]  # empty unless actor is picker
    bury: List[str]  # empty unless actor is picker
    trick: List[AnalyzeObservationTrickSlot]  # relative seat order


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
    # Recurrent-memory drift at this decision: cosine distance between the
    # actor's GRU memory before and after their own encode. Measured only at
    # the actor's decisions (trick-completion observes update memory too but
    # are not sampled here). None on the seat's first encode (memory is zeros).
    memoryCosineDistance: Optional[float] = None
    memoryNorm: Optional[float] = None
    validActionIds: List[int]
    probabilities: List[AnalyzeProbability]
    view: Dict[str, Any]
    observation: AnalyzeObservation


class AnalyzeSeatCalibration(BaseModel):
    seat: int
    seatName: str
    won: bool  # final score > 0
    decisionCount: int
    firstWinProb: float
    lastWinProb: float
    meanWinProb: float
    brierScore: float  # mean (winProb - won)^2 over this seat's decisions
    pointsMae: float  # mean |predicted - final| of predictions ABOUT this seat


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


class AnalyzeGameSummary(BaseModel):
    hands: Dict[str, List[str]]  # player name -> cards
    blind: List[str]
    picker: Optional[str] = None
    partner: Optional[str] = None
    bury: List[str]
    pickerPoints: int
    defenderPoints: int
    scores: List[int]  # indexed by seat-1


class AnalyzePickRequest(BaseModel):
    # extra="forbid" for the same reason as AnalyzeSimulateRequest.
    model_config = ConfigDict(extra="forbid")
    seed: Optional[int] = None
    partnerMode: int = 1  # 0 = JD, 1 = Called Ace
    seat: int = 1  # 1..5; seats before it are forced to pass
    hand: Optional[List[str]] = None  # exactly 6 cards, or None for random
    blind: Optional[List[str]] = None  # exactly 2 cards, or None for random
    deterministic: bool = True


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
    meta: Dict[str, Any]
    scenario: AnalyzePickScenario
    # One entry per pre-play decision (pick/pass, call, under, bury), same
    # shape as the simulate trace; reward/return fields stay None since no
    # full game is played.
    decisions: List[AnalyzeActionDetail]
    outcome: AnalyzePickOutcome


class AnalyzeSimulateResponse(BaseModel):
    meta: Dict[str, Any]
    summary: Optional[AnalyzeGameSummary] = None
    calibration: Optional[AnalyzeCalibrationSummary] = None
    trace: List[AnalyzeActionDetail]
    final: Optional[Dict[str, Any]] = None
