import { useEffect, useState } from "react";
import type { TableSummary } from "../types";
import { apiFetch } from "../api";
import { STORAGE_KEYS } from "../storage";

const SHIRE_TOWNS = [
  "Bywater",
  "Hobbiton",
  "Overhill",
  "Frogmorton",
  "Michel Delving",
  "Tuckborough",
  "Bree",
  "Bucklebury",
  "Deephallow",
  "Waymoot",
  "Nobottle",
  "Whitfurrows",
] as const;

const HOBBIT_NAMES = [
  "Bilbo",
  "Frodo",
  "Samwise",
  "Merry",
  "Pippin",
  "Otho",
  "Smeagol",
  "Deagol",
  "Lobelia",
  "Belladonna",
] as const;

const getRandomItem = <T,>(items: readonly T[]) =>
  items[Math.floor(Math.random() * items.length)];

export interface UseIdentityReturn {
  name: string;
  handleTableNameChange: (value: string) => void;
  displayPlaceholder: string | null;
  displayNameInput: string;
  setDisplayNameInput: (value: string) => void;
  displayName: string;
  playerId: string | null;
  persistTypedName: (
    id: string | null,
    typedName: string,
  ) => Promise<void>;
  persistIdentityFromJoin: (
    joined: { player_id?: string },
    typedName: string,
  ) => Promise<string | null>;
}

export function useIdentity(tables: TableSummary[]): UseIdentityReturn {
  const [name, setName] = useState<string>("");
  const [displayPlaceholder, setDisplayPlaceholder] = useState<string | null>(
    null,
  );
  const [displayNameInput, setDisplayNameInput] = useState("");
  const displayName = displayNameInput.trim() || displayPlaceholder || "";
  const [hasCustomName, setHasCustomName] = useState(false);
  const [playerId, setPlayerId] = useState<string | null>(null);

  useEffect(() => {
    setName(getRandomItem(SHIRE_TOWNS));
  }, []);

  useEffect(() => {
    if (!displayPlaceholder) {
      setDisplayPlaceholder(getRandomItem(HOBBIT_NAMES));
    }
  }, [displayPlaceholder]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    let cancelled = false;
    const storedId = window.localStorage.getItem(STORAGE_KEYS.playerId);
    const storedName =
      window.localStorage.getItem(STORAGE_KEYS.displayName) ?? "";
    if (storedName) {
      setDisplayNameInput(storedName);
    }
    if (!storedId) {
      setPlayerId(null);
      return;
    }
    setPlayerId(storedId);
    (async () => {
      try {
        const res = await apiFetch(`/api/players/${storedId}`);
        if (cancelled) return;
        if (res.status === 404) {
          window.localStorage.removeItem(STORAGE_KEYS.playerId);
          setPlayerId(null);
          return;
        }
        if (!res.ok) return;
        const data = await res.json();
        if (data.name) {
          window.localStorage.setItem(STORAGE_KEYS.displayName, data.name);
          setDisplayNameInput((prev) => prev || data.name);
        }
      } catch {
        // Network failure — keep the stored id; the user can still play.
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (hasCustomName || tables.length === 0) return;
    const usedNames = new Set(tables.map((t) => t.name));
    if (!usedNames.has(name)) return;
    const available = SHIRE_TOWNS.filter((town) => !usedNames.has(town));
    if (!available.length) return;
    setName(getRandomItem(available));
  }, [tables, hasCustomName, name]);

  const handleTableNameChange = (value: string) => {
    if (!hasCustomName) {
      setHasCustomName(true);
    }
    setName(value);
  };

  const persistTypedName = async (id: string | null, typedName: string) => {
    if (typeof window === "undefined" || !id || !typedName) return;
    const previous = window.localStorage.getItem(STORAGE_KEYS.displayName);
    if (previous === typedName) return;
    try {
      await apiFetch(`/api/players/${id}`, {
        method: "PATCH",
        body: JSON.stringify({ name: typedName }),
      });
      window.localStorage.setItem(STORAGE_KEYS.displayName, typedName);
    } catch {
      // best-effort; live display name still works via display_name field
    }
  };

  const persistIdentityFromJoin = async (
    joined: { player_id?: string },
    typedName: string,
  ): Promise<string | null> => {
    if (typeof window === "undefined") return playerId;
    let resultId = playerId;
    if (joined.player_id && joined.player_id !== playerId) {
      window.localStorage.setItem(STORAGE_KEYS.playerId, joined.player_id);
      setPlayerId(joined.player_id);
      resultId = joined.player_id;
    }
    await persistTypedName(resultId, typedName);
    return resultId;
  };

  return {
    name,
    handleTableNameChange,
    displayPlaceholder,
    displayNameInput,
    setDisplayNameInput,
    displayName,
    playerId,
    persistTypedName,
    persistIdentityFromJoin,
  };
}
