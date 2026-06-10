# Sheepshead

Sheepshead is a trick-taking card game played with 5 players and a 32-card deck. That deck consists of all cards in a standard deck of 52 cards except for 6s and below. A 6 card hand is dealt to each player plus a 2 card blind. Teams are determined dynamically over the course of a hand, and are typically 2 vs 3 players. The "picker" who chooses to pick up the blind and (optionally) call a partner play against the other 3 players. The team with more than 60 points wins. Ties are won by the "defending" team of 3 players.

## Card Ranking

### Trump

Queen of Clubs (Highest)
Queen of Spades
Queen of Hearts
Queen of Diamonds
Jack of Clubs
Jack of Spades
Jack of Hearts
Jack of Diamonds
Ace of Diamonds
10 of Diamonds
K of Diamonds
9 of Diamonds
8 of Diamonds
7 of Diamonds

### Fail Suits

Ace
10
King
9
8
7

Note: 10s are higher in power than Kings in Sheepshead.

## Points

Aces - 11 points
10s - 10 points
Kings - 4 points
Queens - 3 points
Jacks - 2 points

All other cards - 0 points

There are a total of 120 points in the deck.

## Partners

Partners are determined either by the person who has the Jack of Diamonds in their hand in Jack of Diamonds partner mode or by the person who has the called card in their hand in called ace mode. In the called ace variant, the picker _must_ have another fail card of the same suit as the called card in hand after burying the blind.

The picker always has the option to call "Alone" instead of a partner, meaning they play by themselves 1 vs 4 against the other players.


# Architecture Design

Game logic is defined in sheepshead.py via the Game and Player classes.

In a given game, each player agent has 6-10 total actions. (6 if they don't have the opportunity to pick and simply play all 6 cards in their hand, 10 if they PICK).
