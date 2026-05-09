--! Previous: -
--! Hash: sha1:89297f233da3433f15772b816c3431791338fd0c
--! Message: Initial schema

-- Initial schema for sheepsheadai.
-- Translated from server/database/sheepshead-ai-db-schema.sql with the
-- adjustments specified in WebServerUpdatePlan_202604.md §3.3:
--   * BIGSERIAL primary keys on tables that previously required caller-supplied ids
--   * cardset.cards_hash UNIQUE for dedup (Phase 5 §5.3)
--   * lookup indexes for trick / trick_card / game_player / game
--   * UNIQUE (ai_model_id, is_deterministic) on ai_player

-- Reference tables ----------------------------------------------------------

CREATE TABLE suit (
    suit_id    SMALLINT NOT NULL PRIMARY KEY,
    code       TEXT     NOT NULL UNIQUE,
    name       TEXT     NOT NULL
);

CREATE TABLE card (
    card_id    SMALLINT NOT NULL PRIMARY KEY,
    suit_id    SMALLINT NOT NULL REFERENCES suit(suit_id),
    code       TEXT     NOT NULL UNIQUE,
    name       TEXT     NOT NULL
);

-- Player / table ------------------------------------------------------------

CREATE TABLE player (
    player_id     UUID                              NOT NULL PRIMARY KEY,
    name          TEXT                              NULL,
    time_created  TIMESTAMP(0) WITHOUT TIME ZONE    NOT NULL,
    last_updated  TIMESTAMP(0) WITHOUT TIME ZONE    NOT NULL
);

CREATE TABLE game_table (
    game_table_id UUID                              NOT NULL PRIMARY KEY,
    name          TEXT                              NOT NULL,
    time_created  TIMESTAMP(0) WITHOUT TIME ZONE    NOT NULL,
    time_closed   TIMESTAMP(0) WITHOUT TIME ZONE    NULL
);

-- AI bookkeeping ------------------------------------------------------------

CREATE TABLE ai_model (
    ai_model_id   BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    label         TEXT                              NOT NULL UNIQUE,
    time_created  TIMESTAMP(0) WITHOUT TIME ZONE    NOT NULL
);

CREATE TABLE ai_player (
    ai_player_id     BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    ai_model_id      BIGINT  NOT NULL REFERENCES ai_model(ai_model_id),
    is_deterministic BOOLEAN NOT NULL,
    CONSTRAINT ai_player_model_determinism_unique
        UNIQUE (ai_model_id, is_deterministic)
);

-- Cardsets (deduped via cards_hash) ----------------------------------------

CREATE TABLE cardset (
    cardset_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    cards_hash TEXT   NOT NULL UNIQUE
);

CREATE TABLE cardset_card (
    cardset_id BIGINT   NOT NULL REFERENCES cardset(cardset_id),
    card_id    SMALLINT NOT NULL REFERENCES card(card_id),
    PRIMARY KEY (cardset_id, card_id)
);

-- Game (one row per hand) ---------------------------------------------------

CREATE TABLE game (
    game_id               UUID                              NOT NULL PRIMARY KEY,
    game_table_id         UUID                              NOT NULL REFERENCES game_table(game_table_id),
    is_double_on_the_bump BOOLEAN                           NOT NULL,
    is_called_partner     BOOLEAN                           NOT NULL,
    is_alone              BOOLEAN                           NULL,
    is_leaster            BOOLEAN                           NULL,
    called_card_id        SMALLINT                          NULL REFERENCES card(card_id),
    under_card_id         SMALLINT                          NULL REFERENCES card(card_id),
    time_created          TIMESTAMP(0) WITHOUT TIME ZONE    NOT NULL,
    time_closed           TIMESTAMP(0) WITHOUT TIME ZONE    NULL,
    blind_id              BIGINT                            NOT NULL REFERENCES cardset(cardset_id),
    bury_id               BIGINT                            NULL REFERENCES cardset(cardset_id)
);

CREATE INDEX game_game_table_id_idx ON game (game_table_id);

CREATE TABLE game_player (
    game_player_id   BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    game_id          UUID     NOT NULL REFERENCES game(game_id),
    player_id        UUID     NULL REFERENCES player(player_id),
    ai_player_id     BIGINT   NULL REFERENCES ai_player(ai_player_id),
    name             TEXT     NOT NULL,
    position         SMALLINT NOT NULL,
    starting_hand_id BIGINT   NOT NULL REFERENCES cardset(cardset_id),
    is_picker        BOOLEAN  NULL,
    is_partner       BOOLEAN  NULL,
    score            SMALLINT NULL,
    CONSTRAINT game_player_game_id_player_id_unique UNIQUE (game_id, player_id),
    CONSTRAINT game_player_game_id_position_unique  UNIQUE (game_id, position),
    CONSTRAINT game_player_player_or_ai_not_null
        CHECK (player_id IS NOT NULL OR ai_player_id IS NOT NULL)
);

CREATE INDEX game_player_game_id_idx     ON game_player (game_id);
CREATE INDEX game_player_player_id_idx   ON game_player (player_id);
CREATE INDEX game_player_ai_player_idx   ON game_player (ai_player_id);

-- Tricks --------------------------------------------------------------------

CREATE TABLE trick (
    trick_id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    game_id            UUID     NOT NULL REFERENCES game(game_id),
    index              SMALLINT NOT NULL,
    lead_player_id     BIGINT   NOT NULL REFERENCES game_player(game_player_id),
    winning_player_id  BIGINT   NULL     REFERENCES game_player(game_player_id),
    points             SMALLINT NOT NULL,
    CONSTRAINT trick_game_id_index_unique UNIQUE (game_id, index)
);

CREATE INDEX trick_game_id_index_idx ON trick (game_id, index);

CREATE TABLE trick_card (
    trick_card_id   BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    trick_id        BIGINT   NOT NULL REFERENCES trick(trick_id),
    card_id         SMALLINT NOT NULL REFERENCES card(card_id),
    game_player_id  BIGINT   NOT NULL REFERENCES game_player(game_player_id),
    index           SMALLINT NOT NULL,
    CONSTRAINT trick_card_trick_id_index_unique
        UNIQUE (trick_id, index),
    CONSTRAINT trick_card_trick_id_card_id_unique
        UNIQUE (trick_id, card_id),
    CONSTRAINT trick_card_trick_id_game_player_id_unique
        UNIQUE (trick_id, game_player_id),
    CONSTRAINT trick_card_game_player_id_card_id_unique
        UNIQUE (game_player_id, card_id)
);

CREATE INDEX trick_card_trick_id_index_idx ON trick_card (trick_id, index);
