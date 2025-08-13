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


