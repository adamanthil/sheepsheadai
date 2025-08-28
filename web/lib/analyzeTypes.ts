// TypeScript types that mirror the backend response schemas

export interface AnalyzeSimulateRequest {
  seed?: number;
  partnerMode?: number; // 0 = JD, 1 = Called Ace
  deterministic?: boolean;
  modelPath?: string;
  maxSteps?: number;
}

export interface AnalyzeProbability {
  actionId: number;
  action: string;
  prob: number;
}

export interface AnalyzeActionDetail {
  stepIndex: number;
  seat: number;
  seatName: string;
  phase: string; // "pick" | "partner" | "bury" | "play"
  actionId: number;
  action: string;
  valueEstimate: number;
  validActionIds: number[];
  probabilities: AnalyzeProbability[];
  view: Record<string, any>;
  state?: number[];
}

export interface AnalyzeGameSummary {
  hands: Record<string, string[]>; // player name -> cards
  blind: string[];
  picker?: string;
  partner?: string;
  bury: string[];
  pickerPoints: number;
  defenderPoints: number;
  scores: number[]; // indexed by seat-1
}

export interface AnalyzeSimulateResponse {
  meta: {
    partnerMode: number;
    deterministic: boolean;
    seed?: number;
    modelPath?: string;
  };
  actionLookup: Record<number, string>;
  players: string[];
  summary?: AnalyzeGameSummary;
  trace: AnalyzeActionDetail[];
  final?: Record<string, any>;
}
