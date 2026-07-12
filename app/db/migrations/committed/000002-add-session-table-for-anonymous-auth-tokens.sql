--! Previous: sha1:89297f233da3433f15772b816c3431791338fd0c
--! Hash: sha1:0a47dfcadf90acb0d02d96e90421a35481c7f36f
--! Message: Add session table for anonymous auth tokens

-- Anonymous session tokens (public-internet hardening).
--
-- A session row binds a server-minted bearer token (stored only as a SHA-256
-- hex hash) to a player. TTL is sliding: the server bumps last_seen /
-- expires_at at most once an hour while the token is in use.

DROP TABLE IF EXISTS session;

CREATE TABLE session (
    session_id    BIGINT GENERATED ALWAYS AS IDENTITY  NOT NULL PRIMARY KEY,
    player_id     UUID                              NOT NULL REFERENCES player(player_id) ON DELETE CASCADE,
    token_hash    TEXT                              NOT NULL UNIQUE,
    time_created  TIMESTAMP(0) WITHOUT TIME ZONE    NOT NULL,
    last_seen     TIMESTAMP(0) WITHOUT TIME ZONE    NOT NULL,
    expires_at    TIMESTAMP(0) WITHOUT TIME ZONE    NOT NULL
);

CREATE INDEX session_player_id_idx ON session (player_id);
