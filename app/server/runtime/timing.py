"""Gameplay timing: how long the server waits on players, tables and the AI.

All in seconds. Kept in one place because the values only make sense
together -- how long a player may be away, how long their seat is held,
how long an abandoned table lives -- and are tuned as a set.

Client-side timings (reconnect backoff, the last-card auto-play) live in
the web app. TRICK_COLLECT_SECONDS must match its trick animation.
"""

from __future__ import annotations

# --- A player who is away -------------------------------------------------

# Default for SHEEPSHEAD_TURN_TIMEOUT_SECONDS: how long a human has to move
# before the AI moves for them and they take a strike.
TURN_TIMEOUT_SECONDS = 120.0
# Strikes in a row before a player is moved out of their seat to watch.
MAX_TIMEOUT_STRIKES = 3
# A seated player whose last tab disconnects keeps the seat this long before
# the AI takes it (they reclaim it on reconnect).
DISCONNECT_GRACE_SECONDS = 10.0
# The same, in the lobby before the first deal: nothing is held up by an
# empty seat there, so it is released at once.
LOBBY_DISCONNECT_GRACE_SECONDS = 0.0
# A host who disconnects keeps host this long before it passes on.
HOST_HANDOFF_GRACE_SECONDS = 10.0

# --- Table lifetime ---------------------------------------------------------

# A table with no human connected closes this long after the last one left.
ABANDONED_TABLE_SECONDS = 30.0
# A newly created table gets this long for its first connection.
NEW_TABLE_GRACE_SECONDS = 5 * 60
# A table never dealt closes this long after creation, players or not.
LOBBY_EXPIRY_SECONDS = 20 * 60
# A finished hand nobody deals on closes the table this long after it ended.
HAND_IDLE_SECONDS = 5 * 60
# How often the idle sweep checks the two expiries above.
SWEEP_INTERVAL_SECONDS = 30.0
# A disconnected, unseated client keeps its reclaim rights this long; after
# that its ClientConn (and any AI seat reservation) is swept.
STALE_CLIENT_SECONDS = 30 * 60

# --- AI pacing --------------------------------------------------------------

# How long the table client spends on a completed trick: a 2.0s look at the
# finished trick, then a 1.3s collect animation (see useTrickAnimation.ts).
# The next card must not land underneath it.
TRICK_COLLECT_SECONDS = 3.3
# Before the AI's first move of a fresh deal, so the hand can be read.
FIRST_AI_MOVE_DELAY_SECONDS = 2.0
# After a doublers throw-in, before the new deal's first bid, so the
# throw-in callout is readable rather than overrun by the next PICK.
THROW_IN_PAUSE_SECONDS = 1.2
# After each AI card play and each AI pass.
AI_PLAY_PAUSE_SECONDS = 0.5
AI_PASS_PAUSE_SECONDS = 0.5
