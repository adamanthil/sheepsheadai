import React from "react";
import DesktopStage from "./stage/DesktopStage";
import MobileStage from "./stage/MobileStage";
import type { StageProps } from "./stage/types";

export type { CallOption, SeatView, StageProps } from "./stage/types";

export default function Stage(props: StageProps) {
  return props.isMobile ? (
    <MobileStage {...props} />
  ) : (
    <DesktopStage {...props} />
  );
}
