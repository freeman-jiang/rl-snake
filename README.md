# RL-Snake

This project trains a model to play the Snake game using reinforcement learning with a policy gradient approach. The agent learns to play through trial and error, gradually improving its performance, eventually matching or exceeding average human gameplay.

Human:

https://github.com/user-attachments/assets/e66ccbbd-953c-4c63-a744-44b504e01261

AI:

https://github.com/user-attachments/assets/3945e1ea-e57b-4e89-ab44-d1f585610886

### Components

- **game.py**: Contains the Snake game implementation with a clean interface for reinforcement learning
- **book.ipynb**: A Jupyter notebook demonstrating the training process using PyTorch
- **snake_policy_weights.pth**: Pre-trained neural network weights for the Snake AI

### Training Process
The project uses a simple neural network policy that takes in the game state (danger sensors, current direction, food direction) and outputs probabilities for each possible action (up, down, left, right).

### Technical Implementation
- State representation: 12 binary features (danger in 4 directions, current direction one-hot, food location)
- Neural network: Simple 12→20→20→4 architecture with ReLU activations and softmax output in PyTorch. Network actually performs decently on a single linear transformation + softmax as well.
- Training algorithm: REINFORCE (Policy Gradient)
- Reward structure: +10 for food, -10 for collisions, small negative reward for each step
