--! Previous: sha1:2935d31b722a267e907988e1652566b498477883
--! Hash: sha1:c0df1a5d33e906bb179ed9f29fe9e6fe5a3654fb
--! Message: drop the stored band counts from hand reviews

-- hand_review no longer keeps how many of a hand's decisions fell in each
-- band, nor which decision headlines it. Those depend on the thresholds a
-- difference is banded by, so they are counted from decision_review when a
-- review is read (services/review_bands.py) and nothing banded is stored.
-- cost and gained stay: plain sums of the differences, whatever the bands.
ALTER TABLE hand_review
    DROP COLUMN IF EXISTS decisions,
    DROP COLUMN IF EXISTS agreed,
    DROP COLUMN IF EXISTS even,
    DROP COLUMN IF EXISTS slight,
    DROP COLUMN IF EXISTS costly,
    DROP COLUMN IF EXISTS better,
    DROP COLUMN IF EXISTS headline_index;
