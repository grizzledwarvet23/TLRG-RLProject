import numpy as np
import torch
import socket
#just import pytorch:
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import select
import time
import json


UDP_IP = "127.0.0.1"
UDP_PORT = 5065
MESSAGE = "YOUR DATA"

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

#set timeout to 5 seconds
sock.settimeout(5)

#create actor-critic network for PPO. just the network part for now:
class ActorCritic(nn.Module):
    def __init__(self,  state_dim): #just the continuous case for now.
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2), #2 actions. a mean and a standard deviation.
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1) #1 value output.
        )


    def forward(self, x):
        raw_actor_output = self.actor(x)
        mean, log_std = raw_actor_output.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()
        dist = Normal(mean, std)
        action = torch.tanh(dist.rsample())  # Sample an action
        value = self.critic(x)  # Get value estimation

        return action, dist.log_prob(action), value
    

#ok we'll just test making a nn and getting output.
state_dim = 2
fire_ac = ActorCritic(state_dim)
optimizer = optim.Adam(fire_ac.parameters(), lr=0.01)

# def preprocess_data(data):
#     data_dict = json.loads(data) #json->python dict
#     state = np.array([data_dict[key] for key in data_dict.keys()])
#     return torch.FloatTensor(state)

#we will just do reinforce update for now.
def reinforce_update(returns, log_probs):
    print("UPDATING!")
    policy_loss = []
    for log_prob, Gt in zip(log_probs, returns):
        policy_loss.append(-log_prob * Gt) #negative for gradient ascent
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    
def compute_returns(rewards):
    returns = []
    Gt = 0
    pw = 0
    for r in rewards:
        Gt += r
        returns.insert(0, Gt)
    returns = torch.tensor(returns)
    returns = returns.float()
    returns = (returns - returns.mean()) / (returns.std() + 1e-5) #normalize returns.
    return returns

timestep = 0

states, actions, rewards = [], [], []
log_probs = []

sock.sendto(bytes(MESSAGE, "utf-8"), (UDP_IP, UDP_PORT)) #one time send?

while True:
    data, addr = sock.recvfrom(1024)
    data = data.decode("utf-8")
    data_dict = json.loads(data)

    #data_dict["ClosestEnemyPosition"] is a 2 element float list. we will use this as the 2 dimensional state to feed into fire_ac.
    state = torch.tensor(data_dict["ClosestEnemyPosition"], dtype=torch.float32)
    action, log_prob, value = fire_ac(state)
    #print(action, log_prob, value)

    #let us send this action back to the game.
    action = action.detach().numpy().tolist()
    #instead of just sending the action, we will do "Action: " + action
    action = "Action: " + json.dumps(action)
    sock.sendto(bytes(action, "utf-8"), (UDP_IP, UDP_PORT))


    log_probs.append(log_prob)
    states.append(data_dict["ClosestEnemyPosition"]) 
    actions.append(action)
    rewards.append(data_dict["Reward"])


    timestep += 1
    print(timestep)
    if timestep == 200:
        returns = compute_returns(rewards)
        reinforce_update(returns, log_probs)
        timestep = 0




    


    







    