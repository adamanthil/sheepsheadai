--! Previous: sha1:c0df1a5d33e906bb179ed9f29fe9e6fe5a3654fb
--! Hash: sha1:9ac77ac57e627bb50802204ad529db6aa7aae372
--! Message: index decision reviews for the review tab

-- The AI review tab totals one player's decisions, and decision_review is
-- keyed by seat (game_player_id), so each of its queries joined through
-- game_player and then read the whole table. These two let the lists be
-- read a seat at a time instead.

-- Pick decisions where the player and the AI went different ways: about
-- one decision in a hundred.
DROP INDEX IF EXISTS decision_review_pick_differs_idx;
CREATE INDEX decision_review_pick_differs_idx
    ON decision_review (game_player_id)
    WHERE kind = 'pick' AND action <> model_action;

-- Decisions where the two differed, with the difference, for the largest
-- differences lists. Deliberately not narrowed to the full point those
-- lists start at: that threshold lives in services/review_bands.py, and an
-- index that named it would stop being used the day it moved.
DROP INDEX IF EXISTS decision_review_seat_difference_idx;
CREATE INDEX decision_review_seat_difference_idx
    ON decision_review (game_player_id, difference)
    WHERE difference IS NOT NULL;

-- Superseded by the one above: ordered by difference across every player,
-- which no query asks for.
DROP INDEX IF EXISTS decision_review_difference_idx;
