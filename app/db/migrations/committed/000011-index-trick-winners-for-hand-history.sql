--! Previous: sha1:78670a758683748bb76578e38c89c666debc007f
--! Hash: sha1:2935d31b722a267e907988e1652566b498477883
--! Message: index trick winners for hand history

-- Enter migration here

-- A hand history row sums the points of the tricks its seat won; without
-- this every row on a page seq-scanned the whole trick table.
DROP INDEX IF EXISTS trick_winning_player_id_idx;
CREATE INDEX trick_winning_player_id_idx ON trick (winning_player_id);
