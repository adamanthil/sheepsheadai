--! Previous: sha1:9ac77ac57e627bb50802204ad529db6aa7aae372
--! Hash: sha1:9e6060e271653bf91cfdc181dc1d25c4470c627a
--! Message: index substituted cards for the vs-AI leaderboard

-- The vs-AI leaderboard finds the hands played against the AI alone by
-- ruling out every seat a person touched, which includes any seat with a
-- substituted card. Nothing indexed that flag, so the check read every
-- trick card to find the few that carry it (well under one in a hundred),
-- and the cost the planner put on that pushed the whole ranking into a
-- plan that compared every hand with every other.
DROP INDEX IF EXISTS trick_card_substituted_idx;
CREATE INDEX trick_card_substituted_idx
    ON trick_card (game_player_id)
    WHERE is_substituted;
