# Model-based artificial agents

This library focus on model-based artificial agent learning. The environment is modeled as a state-transition system, with a set of possible actions for each state and reward for state/action pairs, and the policy improvememt is done through dynamic programming.

### Examples

You can play the Tic-Tac-Toe game against an agent. The agent is trained every time the code is run, and theoretically takes the perfect steps to win a game.

Play Tic-Tac-Toe (using cargo):

```
cargo run --example tictactoe
```
