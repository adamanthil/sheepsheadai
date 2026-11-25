--! Previous: sha1:d51d2a906675d2cb5c0811e87fa0ed380c352315
--! Hash: sha1:78670a758683748bb76578e38c89c666debc007f
--! Message: add ai review tables and queue existing hands

-- AI review: each decision a registered player made in a finished hand,
-- compared with what the AI would have done (services/hand_review.py).
--
-- hand_review is both the queue and the per-hand summary: a row is added
-- for a seat when its hand ends (or when its player registers) and filled
-- in by the background worker. decision_review holds the decisions of a
-- reviewed seat. Both name the ai_model that produced the figures, and a
-- decision can only belong to a review made by the same model.
DROP TABLE IF EXISTS decision_review;
DROP TABLE IF EXISTS hand_review;

CREATE TABLE hand_review (
    game_player_id BIGINT       PRIMARY KEY REFERENCES game_player(game_player_id),
    -- queued:   waiting for the worker
    -- reviewed: done; the figures below and decision_review rows are set
    -- skipped:  not reviewable (the AI made the player's call or bury)
    -- failed:   the stored hand would not replay; see failure
    status         TEXT         NOT NULL DEFAULT 'queued'
                   CHECK (status IN ('queued', 'reviewed', 'skipped', 'failed')),
    time_queued    TIMESTAMP(0) WITHOUT TIME ZONE NOT NULL DEFAULT now(),
    attempts       SMALLINT     NOT NULL DEFAULT 0,
    failure        TEXT         NULL,

    ai_model_id    BIGINT       NULL REFERENCES ai_model(ai_model_id),
    time_reviewed  TIMESTAMP(0) WITHOUT TIME ZONE NULL,
    duration_ms    INTEGER      NULL,
    evaluations    INTEGER      NULL,

    -- Decisions where the player had a choice, and how they compare. A
    -- difference under a quarter point of expected score is "even", one
    -- of a full point or more is "costly" or "better", and what lies
    -- between is "slight".
    decisions      SMALLINT     NULL,
    agreed         SMALLINT     NULL,
    even           SMALLINT     NULL,
    slight         SMALLINT     NULL,
    costly         SMALLINT     NULL,
    better         SMALLINT     NULL,
    -- Expected score the player's choices cost and gained, both >= 0.
    cost           REAL         NULL,
    gained         REAL         NULL,
    -- decision_review.index of the decision the hand is summed up by: the
    -- costliest, else the best gain; NULL when none reaches a full point.
    headline_index SMALLINT     NULL,

    CONSTRAINT hand_review_reviewed_has_model
        CHECK (status <> 'reviewed' OR ai_model_id IS NOT NULL),
    CONSTRAINT hand_review_model_unique UNIQUE (game_player_id, ai_model_id)
);

-- The worker takes the newest queued hand first.
CREATE INDEX hand_review_queued_idx ON hand_review (game_player_id DESC)
    WHERE status = 'queued';
CREATE INDEX hand_review_time_reviewed_idx ON hand_review (time_reviewed)
    WHERE status = 'reviewed';

CREATE TABLE decision_review (
    game_player_id BIGINT   NOT NULL,
    -- Order among the seat's decisions in the hand.
    index          SMALLINT NOT NULL,
    ai_model_id    BIGINT   NOT NULL REFERENCES ai_model(ai_model_id),
    kind           TEXT     NOT NULL
                   CHECK (kind IN ('pick', 'call', 'under', 'bury', 'play')),
    trick_index    SMALLINT NULL,
    -- Card codes: what the seat held, and the [seat, card] plays already
    -- made to the trick.
    hand           TEXT[]   NOT NULL,
    table_cards    JSONB    NOT NULL,
    -- Engine action names; a bury is "BURY <card> <card>".
    action         TEXT     NOT NULL,
    model_action   TEXT     NOT NULL,
    -- [action, probability] over the legal actions, most likely first.
    policy         JSONB    NOT NULL,
    is_forced      BOOLEAN  NOT NULL,
    -- The AI's reading of the hand at the decision, from the seat's view
    -- and (oracle_score) with every card known.
    expected_score REAL     NOT NULL,
    win_prob       REAL     NOT NULL,
    oracle_score   REAL     NULL,
    -- Expected score after the player's choice and after the AI's; set
    -- only where the two differ.
    chosen_score   REAL     NULL,
    model_score    REAL     NULL,
    -- What the player's choice gained (+) or cost (-), as displayed.
    difference     REAL     GENERATED ALWAYS AS
                   (round((chosen_score - model_score)::numeric, 1)::real) STORED,

    PRIMARY KEY (game_player_id, index),
    FOREIGN KEY (game_player_id, ai_model_id)
        REFERENCES hand_review (game_player_id, ai_model_id) ON DELETE CASCADE
);

CREATE INDEX decision_review_difference_idx ON decision_review (difference)
    WHERE difference IS NOT NULL;

-- Queue every finished hand already played by a registered account.
INSERT INTO hand_review (game_player_id)
SELECT gp.game_player_id
FROM game_player gp
JOIN account a ON a.player_id = gp.player_id
JOIN game g ON g.game_id = gp.game_id
WHERE g.time_closed IS NOT NULL AND gp.score IS NOT NULL
ON CONFLICT DO NOTHING;
