from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from uuid import UUID

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
    display_name: str
    player_id: Optional[UUID] = None

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
    status: str
    rules: Dict[str, Any]
    fillWithAI: bool
    seats: Dict[int, Optional[str]] | Dict[str, Optional[str]]
    runningBySeat: Dict[int, int] | Dict[str, int]
    seatOccupants: Dict[int, Optional[str]] | Dict[str, Optional[str]]
    seatIsAI: Dict[int, bool] | Dict[str, bool]
    host: Optional[str]
    resultsHistory: List[Dict[str, Any]]
    initialSeatOrder: List[str]
    initialNames: Dict[str, str]


# Analyze AI Model schemas
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


class AnalyzeTrumpProbability(BaseModel):
    card: str
    probability: float


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
    validActionIds: List[int]
    probabilities: List[AnalyzeProbability]
    view: Dict[str, Any]
    state: Optional[List[float]] = None


class AnalyzeGameSummary(BaseModel):
    hands: Dict[str, List[str]]  # player name -> cards
    blind: List[str]
    picker: Optional[str] = None
    partner: Optional[str] = None
    bury: List[str]
    pickerPoints: int
    defenderPoints: int
    scores: List[int]  # indexed by seat-1


class AnalyzeSimulateResponse(BaseModel):
    meta: Dict[str, Any]
    actionLookup: Dict[int, str]
    players: List[str]
    summary: Optional[AnalyzeGameSummary] = None
    trace: List[AnalyzeActionDetail]
    final: Optional[Dict[str, Any]] = None


class ChatMessage(BaseModel):
    id: str
    table_id: str
    type: str  # "player" | "system"
    author: Optional[str] = None  # player display name for player messages
    body: str
    timestamp: float


class ChatSendRequest(BaseModel):
    client_id: str
    message: str
