# Adversarial Search Problem with Reversi

The original project was based on a course by Mathias Broxvall and adapted at the ENSICAEN by Regis Clouard.

## How to play a game

Use the command line `./reversi.py -1 player1 -2 player2 -n searchLimit -f evaluationFunction`

### As of 1.0, the following players are available :

* Human player : don't use the parameter corresponding to the player you want to play as
* Random player : `ReversiRandomAI`
* Greedy player : `ReversiGreedyAI`
* Minimax player : `ReversiMinimaxAI`
* AlphaBeta player : `ReversiAlphaBetaAI`, implementation of minimax with alpha-beta pruning
* Negamax player : `ReversiNegaMaxAI`
* Negamax player using alpha-beta pruning : `ReversiNegaMaxPruningAI`
* Negamax player using alpha-beta pruning and transposition tables : `ReversiNegaMaxPruningTTAI`
* Negamax player using alpha-beta pruning, transposition tables, iterative deepening and a time
  limit : `ReversiIDNegaMaxPruningTTAI`
* Monte-Carlo Tree Search player : `ReversiMonteCarloTreeSearchAI`
* Monte-Carlo Tree Search player with a time limit : `ReversiMonteCarloTreeSearchTLAI`

The default value for `n` is 3.
For the player containing "TL" in their name, the parameter `n` is the number of seconds the player can use to find a move;

### As of 1.0, the following evaluation functions are available :

* Simple evaluation function : `SimpleEvaluationFunction`, using only the actual score (default value)
* Better evaluation function : `BetterEvaluationFunction`, using the score of the game and made much more important if
  the current state if a terminal state
* Random evaluation function : `RandomEvaluationFunction`, using a random number to evaluate the state if it is not
  terminal, else the score of the game
* Evaluation function based on the number of moves possible : `OtherEvaluationFunction`, using the score if the game is
  finished, else the number of possible moves
* Position related evaluation function : `MyEvaluationFunction`, based on the score if the game is finished, else on the
  position of all the pawns on the board

## How to play several games

Use the command line `./compete.py -1 player1 -2 player2 -n searchLimit -f evaluationFunction`
These two players will play ten games.