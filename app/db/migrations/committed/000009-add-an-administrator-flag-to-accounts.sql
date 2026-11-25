--! Previous: sha1:6fff0772cfd72d9bf709f8bc2a1d00e31287c752
--! Hash: sha1:d51d2a906675d2cb5c0811e87fa0ed380c352315
--! Message: add an administrator flag to accounts

-- Administrators: accounts allowed into the admin area.
--
-- Nobody is made one here. The flag is set by hand:
--   UPDATE account SET is_admin = true WHERE username = '...';
ALTER TABLE account ADD COLUMN IF NOT EXISTS is_admin BOOLEAN NOT NULL DEFAULT false;
