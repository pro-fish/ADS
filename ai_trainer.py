import torch
import torch.nn as nn
import torch.optim as optim
import json
from pathlib import Path

class DispatchAI(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DispatchAI, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class AITrainer:
    def __init__(self, city_name="Jeddah", input_size=10, hidden_size=20, output_size=5):
        self.city_name = city_name
        self.city_data_dir = Path("data") / city_name
        self.city_data_dir.mkdir(parents=True, exist_ok=True)

        self.model_file = self.city_data_dir / "dispatch_model.pth"
        self.training_stats_file = self.city_data_dir / "training_stats.json"

        self.model = self.initialize_model(input_size, hidden_size, output_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.training_stats = {"episodes": 0, "rewards": []}
        self.gamma = 0.99

    def initialize_model(self, input_size, hidden_size, output_size):
        return DispatchAI(input_size, hidden_size, output_size)

    def train_episode(self, state, action, reward, next_state, done=False):
        # Simple Q-learning update
        self.model.train()
        state_value = self.model(state)
        next_state_value = self.model(next_state)
        target = state_value.clone().detach()
        if done:
            target[0, action] = reward
        else:
            target[0, action] = reward + self.gamma * torch.max(next_state_value).item()

        loss = self.criterion(state_value, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.training_stats["episodes"] += 1
        self.training_stats["rewards"].append(reward)

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_file)

    def load_model(self):
        if self.model_file.exists():
            self.model.load_state_dict(torch.load(self.model_file))

    def save_training_stats(self):
        with open(self.training_stats_file, "w") as f:
            json.dump(self.training_stats, f, indent=4)
