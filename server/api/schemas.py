from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class CreateTableRequest(BaseModel):
    name: str
    fillWithAI: bool = True
    rules: Dict[str, Any] = {}


class JoinTableRequest(BaseModel):
    display_name: str
    seat: Optional[int] = None


class UpdateTableRulesRequest(BaseModel):
    client_id: str
    rules: Dict[str, Any]


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
    host: Optional[str]
    hostId: Optional[str]
    resultsHistory: List[Dict[str, Any]]
    initialSeatOrder: List[str]
    initialNames: Dict[str, str]


# Analyze AI Model schemas
class AnalyzeSimulateRequest(BaseModel):
    seed: Optional[int] = None
    partnerMode: int = 1  # 0 = JD, 1 = Called Ace
    deterministic: bool = False
    modelPath: Optional[str] = None
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
    winProb: Optional[float] = None  # [0,1]
    expectedFinalReturn: Optional[float] = None  # unscaled, undiscounted
    secretPartnerProb: Optional[float] = None  # [0,1]
    pointEstimates: Optional[List[AnalyzePointEstimate]] = None
    pointActuals: Optional[List[AnalyzePointEstimate]] = None
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


