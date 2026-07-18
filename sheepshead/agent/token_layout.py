"""Single source of truth for the encoder token layout.

Base token sequence (19 tokens, emitted by ``CardReasoningEncoder``):
    [context, memory, hand x8, trick x5, blind x2, bury x2]
      index 0    1      2:10     10:15     15:17     17:19

The oracle encoder (``OracleCriticEncoder``) extends this with 32
opponent-hand tokens to a 51-token sequence:
    [context, memory, hand x8, trick x5, blind x2, bury x2, opp x32]
      index 0    1      2:10     10:15     15:17     17:19    19:51

Token-type-id embedding uses the same enumeration, plus one extra id
for the oracle's opponent-hand tokens.
"""

CONTEXT_TOKEN = 0
MEMORY_TOKEN = 1
HAND_TOKENS = slice(2, 10)
TRICK_TOKENS = slice(10, 15)
BLIND_TOKENS = slice(15, 17)
BURY_TOKENS = slice(17, 19)
BASE_TOKEN_COUNT = 19

OPPONENT_TOKENS = slice(19, 51)
ORACLE_TOKEN_COUNT = 51

CONTEXT_TYPE_ID = 0
MEMORY_TYPE_ID = 1
HAND_TYPE_ID = 2
TRICK_TYPE_ID = 3
BLIND_TYPE_ID = 4
BURY_TYPE_ID = 5
BASE_TYPE_COUNT = 6

OPPONENT_TYPE_ID = 6
ORACLE_TYPE_COUNT = 7
