--! Previous: sha1:14492e0991b9f34ff05955917590b5f3ace76e3e
--! Hash: sha1:02919953119cc055c0990b24a47f6ea99aceffe3
--! Message: Record the last client IP on player rows

-- The client address the player was last seen from (join, websocket
-- connect, login), recorded for moderation. Raw address, not the /64
-- rate-limit key. It lives and dies with the player row: the identity
-- purge takes it along with the orphaned guest.
ALTER TABLE player
    ADD COLUMN IF NOT EXISTS last_ip INET NULL;
