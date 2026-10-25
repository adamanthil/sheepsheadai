--! Previous: sha1:9df74a01f5c96dbd4d0e804b42b53783d8cf740c
--! Hash: sha1:6fff0772cfd72d9bf709f8bc2a1d00e31287c752
--! Message: ai player identity is model plus sampling

-- AI player identity = model + deployment sampling.
--
-- is_deterministic was a two-valued version of the right key. The deployed
-- policy is sampled per head group with a softmax temperature (0 = argmax)
-- and an optional logit band (sheepshead/agent/sampling.py), and a tempered
-- player is a different player from the argmax one for stats and the
-- leaderboard. All temperatures 0 and no band is exactly the old
-- deterministic player, so existing rows keep their identity.
ALTER TABLE ai_player DROP CONSTRAINT IF EXISTS ai_player_model_determinism_unique;
ALTER TABLE ai_player DROP COLUMN IF EXISTS is_deterministic;
ALTER TABLE ai_player
    ADD COLUMN IF NOT EXISTS temperature_pick    REAL NOT NULL DEFAULT 0 CHECK (temperature_pick >= 0),
    ADD COLUMN IF NOT EXISTS temperature_partner REAL NOT NULL DEFAULT 0 CHECK (temperature_partner >= 0),
    ADD COLUMN IF NOT EXISTS temperature_bury    REAL NOT NULL DEFAULT 0 CHECK (temperature_bury >= 0),
    ADD COLUMN IF NOT EXISTS temperature_play    REAL NOT NULL DEFAULT 0 CHECK (temperature_play >= 0),
    -- nats below the best logit an action may sit and still be drawn; NULL = no band
    ADD COLUMN IF NOT EXISTS logit_band          REAL CHECK (logit_band >= 0);
ALTER TABLE ai_player DROP CONSTRAINT IF EXISTS ai_player_model_sampling_unique;
ALTER TABLE ai_player ADD CONSTRAINT ai_player_model_sampling_unique
    UNIQUE NULLS NOT DISTINCT (ai_model_id, temperature_pick, temperature_partner,
                               temperature_bury, temperature_play, logit_band);
