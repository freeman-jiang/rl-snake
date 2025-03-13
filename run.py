import time

import numpy as np
import torch
import torch.nn as nn

from game import SnakeGame


# Define the PolicyNetwork class
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, output_dim),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        return self.layer(x)

def main():
    # Initialize the game
    game = SnakeGame(render=True)
    
    # Initialize policy network with the same dimensions as in training
    input_dim = 12
    output_dim = 4
    policy_network = PolicyNetwork(input_dim, output_dim)
    
    # Load trained weights
    try:
        policy_network.load_state_dict(torch.load('snake_policy_weights.pth'))
        policy_network.eval()  # Set to evaluation mode
        print("Successfully loaded trained model weights!")
    except FileNotFoundError:
        print("Could not find model weights file 'snake_policy_weights.pth'")
        return
    
    # Run game with trained policy
    state = game.reset()
    total_reward = 0
    done = False
    frame_count = 0
    
    print("Running game with trained policy...")
    print("Press Ctrl+C to stop")
    
    try:
        while not done:
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # Get action from policy
            with torch.no_grad():
                action_probs = policy_network(state_tensor)
                
            # Select best action (deterministic policy)
            action = torch.argmax(action_probs, dim=1).item()
            
            # Take step in environment
            state, reward, done, info = game.step(action)
            total_reward += reward
            frame_count += 1
            
            # Add small delay to make game viewable
            time.sleep(0.05)
            
            # Print status every 10 frames
            if frame_count % 10 == 0:
                print(f"Score: {info['score']}, Total Reward: {total_reward:.2f}")
        
        print(f"Game over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    
    except KeyboardInterrupt:
        print("\nGame stopped by user")
        print(f"Final Score: {info['score']}, Total Reward: {total_reward:.2f}")

if __name__ == "__main__":
    main()
