import { useState, useCallback, useRef } from 'react';

export type CalloutKind = 'PICK' | 'CALL' | 'LEASTER' | 'ALONE';

export interface Callout {
  kind: CalloutKind;
  message: string;
}

export interface UseCalloutReturn {
  callout: Callout | null;
  showCallout: (kind: CalloutKind, message: string, duration?: number) => void;
  clearCallout: () => void;
}

export function useCallout(): UseCalloutReturn {
  const [callout, setCallout] = useState<Callout | null>(null);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  const clearCallout = useCallback(() => {
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
    setCallout(null);
  }, []);

  const showCallout = useCallback((kind: CalloutKind, message: string, duration = 1800) => {
    // Clear any existing timer
    if (timerRef.current) {
      clearTimeout(timerRef.current);
    }

    setCallout({ kind, message });

    timerRef.current = setTimeout(() => {
      setCallout(null);
      timerRef.current = null;
    }, duration);
  }, []);

  return { callout, showCallout, clearCallout };
}

