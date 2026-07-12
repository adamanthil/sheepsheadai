CREATE TABLE game(
    game_id UUID NOT NULL,
    game_table_id UUID NOT NULL,
    is_double_on_the_bump BOOLEAN NOT NULL,
    is_called_partner BOOLEAN NOT NULL,
    is_alone BOOLEAN NULL,
    is_leaster BOOLEAN NULL,
    called_card_id SMALLINT NULL,
    under_card_id SMALLINT NULL,
    time_created TIMESTAMP(0) WITHOUT TIME ZONE NOT NULL,
    time_closed TIMESTAMP(0) WITHOUT TIME ZONE NULL,
    blind_id BIGINT NOT NULL,
    bury_id BIGINT NULL
);
ALTER TABLE
    game ADD PRIMARY KEY(game_id);
CREATE TABLE player(
    player_id UUID NOT NULL,
    name TEXT NULL,
    time_created TIMESTAMP(0) WITHOUT TIME ZONE NOT NULL,
    last_updated TIMESTAMP(0) WITHOUT TIME ZONE NOT NULL
);
ALTER TABLE
    player ADD PRIMARY KEY(player_id);
CREATE TABLE ai_model(
    ai_model_id BIGINT NOT NULL,
    label TEXT NOT NULL,
    time_created TIMESTAMP(0) WITHOUT TIME ZONE NOT NULL
);
ALTER TABLE
    ai_model ADD CONSTRAINT ai_model_label_unique UNIQUE(label);
ALTER TABLE
    ai_model ADD PRIMARY KEY(ai_model_id);
CREATE TABLE game_table(
    game_table_id UUID NOT NULL,
    name TEXT NOT NULL,
    time_created TIMESTAMP(0) WITHOUT TIME ZONE NOT NULL,
    time_closed TIMESTAMP(0) WITHOUT TIME ZONE NULL
);
ALTER TABLE
    game_table ADD PRIMARY KEY(game_table_id);

CREATE TABLE game_player(
    game_player_id BIGINT NOT NULL,
    game_id UUID NOT NULL,
    player_id UUID NULL,
    ai_player_id BIGINT NULL,
    name TEXT NOT NULL,
    position SMALLINT NOT NULL,
    starting_hand_id BIGINT NOT NULL,
    is_picker BOOLEAN NULL,
    is_partner BOOLEAN NULL,
    score SMALLINT NULL
);
ALTER TABLE
    game_player ADD CONSTRAINT game_player_game_id_player_id_unique UNIQUE(game_id, player_id);
ALTER TABLE
    game_player ADD CONSTRAINT game_player_game_id_position_unique UNIQUE(game_id, position);
ALTER TABLE
    game_player ADD PRIMARY KEY(game_player_id);
ALTER TABLE
    game_player ADD CONSTRAINT game_player_player_id_ai_player_id_both_not_null CHECK (player_id IS NOT NULL OR ai_player_id IS NOT NULL);
CREATE TABLE cardset(cardset_id BIGINT NOT NULL);
ALTER TABLE
    cardset ADD PRIMARY KEY(cardset_id);
CREATE TABLE card(
    card_id SMALLINT NOT NULL,
    suit_id SMALLINT NOT NULL,
    code TEXT NOT NULL,
    name TEXT NOT NULL
);
ALTER TABLE
    card ADD CONSTRAINT card_code_unique UNIQUE(code);
ALTER TABLE
    card ADD PRIMARY KEY(card_id);
CREATE TABLE cardset_card(
    cardset_id BIGINT NOT NULL,
    card_id SMALLINT NOT NULL
);
ALTER TABLE
    cardset_card ADD PRIMARY KEY(cardset_id, card_id);
CREATE TABLE trick(
    trick_id BIGINT NOT NULL,
    game_id UUID NOT NULL,
    index SMALLINT NOT NULL,
    lead_player_id BIGINT NOT NULL,
    winning_player_id BIGINT NULL,
    points SMALLINT NOT NULL
);
ALTER TABLE
    trick ADD CONSTRAINT trick_game_id_index_unique UNIQUE(game_id, index);
ALTER TABLE
    trick ADD PRIMARY KEY(trick_id);
CREATE TABLE suit(
    suit_id SMALLINT NOT NULL,
    code TEXT NOT NULL,
    name TEXT NOT NULL
);
ALTER TABLE
    suit ADD CONSTRAINT suit_code_unique UNIQUE(code);
ALTER TABLE
    suit ADD PRIMARY KEY(suit_id);
CREATE TABLE trick_card(
    trick_card_id BIGINT NOT NULL,
    trick_id BIGINT NOT NULL,
    card_id SMALLINT NOT NULL,
    game_player_id BIGINT NOT NULL,
    index SMALLINT NOT NULL
);
ALTER TABLE
    trick_card ADD CONSTRAINT trick_card_trick_id_index_unique UNIQUE(trick_id, index);
ALTER TABLE
    trick_card ADD CONSTRAINT trick_card_trick_id_card_id_unique UNIQUE(trick_id, card_id);
ALTER TABLE
    trick_card ADD CONSTRAINT trick_card_trick_id_game_player_id_unique UNIQUE(trick_id, game_player_id);
ALTER TABLE
    trick_card ADD CONSTRAINT trick_card_game_player_id_card_id_unique UNIQUE(game_player_id, card_id);
ALTER TABLE
    trick_card ADD PRIMARY KEY(trick_card_id);
CREATE TABLE ai_player(
    ai_player_id BIGINT NOT NULL,
    ai_model_id BIGINT NOT NULL,
    is_deterministic BOOLEAN NOT NULL
);
ALTER TABLE
    ai_player ADD PRIMARY KEY(ai_player_id);
ALTER TABLE
    game ADD CONSTRAINT game_called_card_id_fk FOREIGN KEY(called_card_id) REFERENCES card(card_id);
ALTER TABLE
    game ADD CONSTRAINT game_game_table_id_fk FOREIGN KEY(game_table_id) REFERENCES game_table(game_table_id);
ALTER TABLE
    trick ADD CONSTRAINT trick_winning_player_id_fk FOREIGN KEY(winning_player_id) REFERENCES game_player(game_player_id);
ALTER TABLE
    trick ADD CONSTRAINT trick_lead_player_id_fk FOREIGN KEY(lead_player_id) REFERENCES game_player(game_player_id);
ALTER TABLE
    game ADD CONSTRAINT game_blind_id_fk FOREIGN KEY(blind_id) REFERENCES cardset(cardset_id);
ALTER TABLE
    trick_card ADD CONSTRAINT trick_card_card_id_fk FOREIGN KEY(card_id) REFERENCES card(card_id);
ALTER TABLE
    game_player ADD CONSTRAINT game_player_starting_hand_id_fk FOREIGN KEY(starting_hand_id) REFERENCES cardset(cardset_id);
ALTER TABLE
    trick ADD CONSTRAINT trick_game_id_fk FOREIGN KEY(game_id) REFERENCES game(game_id);
ALTER TABLE
    card ADD CONSTRAINT card_suit_id_fk FOREIGN KEY(suit_id) REFERENCES suit(suit_id);
ALTER TABLE
    ai_player ADD CONSTRAINT ai_player_ai_model_id_fk FOREIGN KEY(ai_model_id) REFERENCES ai_model(ai_model_id);
ALTER TABLE
    game_player ADD CONSTRAINT game_player_ai_player_id_fk FOREIGN KEY(ai_player_id) REFERENCES ai_player(ai_player_id);
ALTER TABLE
    trick_card ADD CONSTRAINT trick_card_game_player_id_fk FOREIGN KEY(game_player_id) REFERENCES game_player(game_player_id);
ALTER TABLE
    game_player ADD CONSTRAINT game_player_game_id_fk FOREIGN KEY(game_id) REFERENCES game(game_id);
ALTER TABLE
    trick_card ADD CONSTRAINT trick_card_trick_id_fk FOREIGN KEY(trick_id) REFERENCES trick(trick_id);
ALTER TABLE
    cardset_card ADD CONSTRAINT cardset_card_card_id_fk FOREIGN KEY(card_id) REFERENCES card(card_id);
ALTER TABLE
    game ADD CONSTRAINT game_bury_id_fk FOREIGN KEY(bury_id) REFERENCES cardset(cardset_id);
ALTER TABLE
    game_player ADD CONSTRAINT game_player_player_id_fk FOREIGN KEY(player_id) REFERENCES player(player_id);
ALTER TABLE
    game ADD CONSTRAINT game_under_card_id_fk FOREIGN KEY(under_card_id) REFERENCES card(card_id);
ALTER TABLE
    cardset_card ADD CONSTRAINT cardset_card_cardset_id_fk FOREIGN KEY(cardset_id) REFERENCES cardset(cardset_id);
