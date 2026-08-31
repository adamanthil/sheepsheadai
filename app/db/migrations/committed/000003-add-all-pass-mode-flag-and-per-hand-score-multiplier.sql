--! Previous: sha1:0a47dfcadf90acb0d02d96e90421a35481c7f36f
--! Hash: sha1:936baee3b2a71e53a071636bb9ac3e38ed2b9c94
--! Message: Add all-pass mode flag and per-hand score multiplier

-- All-pass house rule (leasters vs doublers).
--
-- game_table.is_doublers is the table-level setting: FALSE = a passed-out
-- hand is played as a leaster (the historical behaviour), TRUE = the hand is
-- thrown in and redealt with the stake doubled.
--
-- game.score_multiplier is the stake the hand was played for: 1 normally,
-- 2 / 4 / ... for each consecutive passed-out redeal that preceded it. A
-- passed-out doublers deal is itself recorded as a game row carrying the
-- multiplier it was dealt at, so the whole chain is auditable.

ALTER TABLE game_table
    ADD COLUMN IF NOT EXISTS is_doublers BOOLEAN NOT NULL DEFAULT FALSE;

ALTER TABLE game
    ADD COLUMN IF NOT EXISTS score_multiplier SMALLINT NOT NULL DEFAULT 1;
