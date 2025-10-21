# Counterfactual Regret Minimization (CFR)

A Python 3 implementation of Counterfactual Regret Minimization algorithm for learning Nash equilibrium strategies in various games.

## Introduction

Counterfactual Regret Minimization (CFR) is a powerful algorithm for finding optimal strategies in imperfect information games. This project demonstrates CFR implementations across three different game types, showcasing how the algorithm adapts to various game structures from simple simultaneous games to complex sequential games with information sets.

CFR is particularly valuable in game theory and AI applications, especially in poker and other strategic games where players must make decisions with incomplete information about their opponents' states.

## Key Concepts

**What is CFR**: CFR is an iterative algorithm that learns optimal strategies by minimizing "regret" - the difference between what a player could have gained by playing optimally versus what they actually gained. Over many iterations, this process converges to Nash equilibrium strategies.

**Regret Matching**: The core mechanism that updates strategies based on accumulated regrets. Actions that would have been better are played more frequently in future iterations.

**Nash Equilibrium**: A stable state where no player can improve their outcome by unilaterally changing their strategy. CFR guarantees convergence to approximate Nash equilibrium in zero-sum games.

**Why it matters**: CFR enables AI systems to learn optimal strategies in complex games without requiring explicit game tree search, making it practical for real-world applications like poker AI and strategic decision making.

## Game Implementations

### Kuhn Poker (`Kuhn.py`)

A simplified poker variant with 3 cards and 2 players, perfect for understanding CFR fundamentals.

**Game Rules**:
- Each player gets one private card from a deck of 3 cards (1, 2, 3)
- Players can either pass (0) or bet (1)
- Higher card wins, with different payouts based on betting patterns

**Algorithm**: Vanilla CFR with sequential strategy manager

**Key Features**:
- Information sets based on private cards and action history
- Sequential decision making
- Demonstrates CFR convergence in imperfect information games

**Usage**:
```bash
python3 Kuhn.py
```

**Expected Output**: Average game value and learned strategies for each information set, showing how players should bet with different cards.

### Rock-Paper-Scissors (`RPS.py`)

Classic simultaneous-move game demonstrating regret matching in its simplest form.

**Description**: Two players simultaneously choose rock, paper, or scissors

**Algorithm**: Regret Matching for simultaneous-move games

**Key Features**:
- Demonstrates convergence to uniform Nash equilibrium (1/3, 1/3, 1/3)
- No information sets needed (simultaneous moves)
- Simple regret matching without reach probabilities

**Usage**:
```bash
python3 RPS.py
```

**Expected Output**: Both players' optimal strategies showing equal probability for each action.

### Liar's Dice (`liar_die.py`)

Simplified version of Liar's Dice with one die per player, showcasing CFR in more complex sequential games.

**Game Rules**:
- Each player rolls one die (1-6)
- Players take turns making claims about dice values
- Opponents can either accept or doubt claims
- Doubting reveals the truth and determines winner

**Algorithm**: CFR with forward/backward propagation

**Key Features**:
- More complex game tree with claim-response mechanism
- CSV export functionality for strategy analysis
- Forward and backward passes for efficient computation
- Progress reporting during training

**Usage**:
```bash
python3 liar_die.py
```

**Expected Output**: Detailed strategy tables showing optimal claiming and response strategies for different game states, saved as CSV files.

## Project Architecture

### Strategy Managers

#### `sequantial_strategy_manager.py`
For sequential games (Kuhn Poker, Liar's Dice):
- **`SequentialStrategyManager`**: Manages strategy and regret tracking for a single information set
- **`SequentialStrategyManagerMap`**: Container for multiple information sets
- **Key Features**: Reach probability tracking, regret matching, average strategy computation

#### `simultaneous_strategy_manager.py`
For simultaneous-move games (RPS):
- **`SimultaneousStrategyManager`**: Simpler regret matching without reach probabilities
- **Key Features**: Direct regret matching, strategy averaging

## Installation & Requirements

**Python Version**: Python 3.6+

**Dependencies**:
```
numpy
pandas
```

**Installation**:
```bash
pip3 install numpy pandas
```

## Usage Instructions

### Basic Usage

Run any of the three game implementations:

```bash
# Kuhn Poker
python3 Kuhn.py

# Rock-Paper-Scissors  
python3 RPS.py

# Liar's Dice
python3 liar_die.py
```

### Customization

**Kuhn Poker**: Modify `iterations` parameter in `train()` function
```python
avg_game_value = train(iterations=1000000)  # Default: 1,000,000
```

**Rock-Paper-Scissors**: Modify `iterations` parameter in `train()` function
```python
train(1000000)  # Default: 1,000,000
```

**Liar's Dice**: Modify `sides` and `iterations` parameters
```python
trainer = LiarDieTrainer(sides=6)  # Default: 6-sided die
trainer.train(iterations=ITERATION)  # Default: 100,000
```

## Results & Interpretation

### Understanding Output

**Kuhn Poker**:
- Average game value: Expected payoff for the first player
- Strategy tables: Probability of betting vs passing for each card/history combination

**Rock-Paper-Scissors**:
- Strategy percentages: Should converge to ~33.33% for each action
- Demonstrates uniform Nash equilibrium

**Liar's Dice**:
- Initial claim policies: How to start the game with different dice rolls
- Response strategies: When to doubt vs accept opponent claims
- Claim strategies: What to claim given current situation
- CSV files: Detailed strategy data for further analysis

### Expected Convergence

All implementations should show:
- Decreasing regret over iterations
- Convergence to Nash equilibrium strategies
- Stable average strategies in final iterations

Results are saved in the `SimulationResult/` directory for Liar's Dice.

## Algorithm Details

### CFR Iteration Process

1. **Initialize**: Set up game state with random deal/rolls
2. **Forward Pass**: Compute reach probabilities through game tree
3. **Backward Pass**: Calculate counterfactual values from terminal states
4. **Update Regrets**: Add regrets weighted by opponent reach probability
5. **Update Strategy**: Use regret matching to compute new strategy
6. **Accumulate**: Add to average strategy for convergence

### Key Concepts

- **Information Sets**: Game states that look identical to a player
- **Reach Probabilities**: Likelihood of reaching each game state
- **Counterfactual Values**: Expected utility if player deviated from current strategy
- **Regret Matching**: Strategy update rule based on accumulated regrets

## Code Quality Features

- **Comprehensive docstrings**: Every class and function documented
- **Type hints**: Full type annotation for better code clarity
- **Modular architecture**: Clean separation between game logic and strategy management
- **Error handling**: Input validation and edge case handling
- **Progress reporting**: Training progress updates for long-running algorithms

## References & Resources

- [Counterfactual Regret Minimization Tutorial](https://stevengong.co/notes/Counterfactual-Regret-Minimization) - Comprehensive CFR explanation and implementation guide

## Future Work / Extensions

Potential improvements and extensions:
- **Advanced CFR variants**: CFR+, Monte Carlo CFR (MCCFR)
- **Additional games**: Texas Hold'em, other poker variants
- **Performance optimizations**: Parallel processing, memory efficiency
- **Visualization tools**: Game tree visualization, strategy convergence plots
- **Interactive demos**: Web interface for exploring learned strategies

## License

This project is open source. Feel free to use, modify, and distribute according to your needs.

## Contributing

Contributions are welcome! Areas for improvement:
- Additional game implementations
- Performance optimizations
- Documentation improvements
- Test coverage
- Visualization tools
