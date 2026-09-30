--! Previous: sha1:02919953119cc055c0990b24a47f6ea99aceffe3
--! Hash: sha1:ea47fa086b612307fa0f3713850ddf39a58d8aea
--! Message: Add optional accounts and emailed tokens

-- Optional accounts. An account hangs off an existing player row (the
-- guest's own identity is upgraded in place, so hands already played
-- count toward it) and adds a unique username, an email, and a password.
-- player.name stays the non-unique in-game display name.
--
-- Stats, the leaderboard, and the in-game account badge unlock once the
-- email is verified (email_verified_at IS NOT NULL); play never waits on it.

DROP TABLE IF EXISTS email_token;
DROP TABLE IF EXISTS account;

CREATE TABLE account (
    player_id          UUID                            NOT NULL PRIMARY KEY REFERENCES player(player_id) ON DELETE CASCADE,
    -- Display case preserved; uniqueness is case-insensitive (index below).
    username           TEXT                            NOT NULL CHECK (username ~ '^[A-Za-z0-9_-]{3,20}$'),
    -- Stored trimmed and lowercased by the server.
    email              TEXT                            NOT NULL,
    -- argon2id PHC string.
    password_hash      TEXT                            NOT NULL,
    email_verified_at  TIMESTAMP(0) WITHOUT TIME ZONE  NULL,
    time_created       TIMESTAMP(0) WITHOUT TIME ZONE  NOT NULL,
    last_updated       TIMESTAMP(0) WITHOUT TIME ZONE  NOT NULL,
    last_login         TIMESTAMP(0) WITHOUT TIME ZONE  NULL
);

CREATE UNIQUE INDEX account_username_idx ON account (lower(username));
CREATE UNIQUE INDEX account_email_idx ON account (email);

-- Single-use emailed tokens (address verification, password reset). Like
-- session tokens, only the SHA-256 hex hash is stored.
CREATE TABLE email_token (
    token_hash  TEXT                            NOT NULL PRIMARY KEY,
    player_id   UUID                            NOT NULL REFERENCES player(player_id) ON DELETE CASCADE,
    purpose     TEXT                            NOT NULL CHECK (purpose IN ('verify', 'reset')),
    expires_at  TIMESTAMP(0) WITHOUT TIME ZONE  NOT NULL,
    used_at     TIMESTAMP(0) WITHOUT TIME ZONE  NULL
);

CREATE INDEX email_token_player_id_idx ON email_token (player_id);
