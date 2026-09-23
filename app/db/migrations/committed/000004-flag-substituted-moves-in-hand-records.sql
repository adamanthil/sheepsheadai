--! Previous: sha1:936baee3b2a71e53a071636bb9ac3e38ed2b9c94
--! Hash: sha1:14492e0991b9f34ff05955917590b5f3ace76e3e
--! Message: Flag substituted moves in hand records

-- Moves made by someone other than the game_player row's owner: the AI
-- playing for a disconnected or timed-out human, or a human who took over
-- an AI seat mid-hand. Such hands are excluded when measuring the AI
-- against humans. Rows written before this migration are unflagged.
ALTER TABLE game_player
    ADD COLUMN IF NOT EXISTS is_substituted_pick BOOLEAN NOT NULL DEFAULT FALSE;

ALTER TABLE trick_card
    ADD COLUMN IF NOT EXISTS is_substituted BOOLEAN NOT NULL DEFAULT FALSE;
