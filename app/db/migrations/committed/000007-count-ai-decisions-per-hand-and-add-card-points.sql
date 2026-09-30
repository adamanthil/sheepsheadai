--! Previous: sha1:ea47fa086b612307fa0f3713850ddf39a58d8aea
--! Hash: sha1:9df74a01f5c96dbd4d0e804b42b53783d8cf740c
--! Message: Count AI decisions per hand and add card points

-- Decisions the AI made on each human row's behalf (pick/pass, call,
-- under, each bury card, each card play), and the subset not held against
-- the player (the host closed the table, they were removed, the server
-- restarted). Whether a hand counts as abandoned is decided at query time
-- from these, so the threshold can change without rewriting history.
ALTER TABLE game_player
    ADD COLUMN IF NOT EXISTS ai_actions         SMALLINT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS ai_actions_excused SMALLINT NOT NULL DEFAULT 0;

-- Approximate backfill for human rows written before the counters: the
-- bidding phase was a single flag, so it counts as one decision. Guarded
-- on ai_actions = 0 so a re-run never overwrites exact counts.
UPDATE game_player gp
SET ai_actions = gp.is_substituted_pick::int + (
    SELECT count(*) FROM trick_card tc
    WHERE tc.game_player_id = gp.game_player_id AND tc.is_substituted
)
WHERE gp.player_id IS NOT NULL
  AND gp.ai_actions = 0
  AND (gp.is_substituted_pick OR EXISTS (
    SELECT 1 FROM trick_card tc
    WHERE tc.game_player_id = gp.game_player_id AND tc.is_substituted
  ));

-- Card point value (A 11, 10 10, K 4, Q 3, J 2, else 0), for scoring the
-- bury when deriving points taken from the trick history. A fresh
-- database seeds it from fixtures/afterReset.sql instead.
ALTER TABLE card ADD COLUMN IF NOT EXISTS points SMALLINT NOT NULL DEFAULT 0;
UPDATE card SET points = CASE left(code, length(code) - 1)
    WHEN 'A' THEN 11 WHEN '10' THEN 10 WHEN 'K' THEN 4 WHEN 'Q' THEN 3
    WHEN 'J' THEN 2 ELSE 0 END;
