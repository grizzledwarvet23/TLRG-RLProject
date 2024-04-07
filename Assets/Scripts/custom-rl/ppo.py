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

#it might've been saved to the path "fire_ac", so load it if it exists:
try:
    fire_ac.load_state_dict(torch.load("fire_ac2"))
    print("loaded model")
except:
    print("no model found")
    pass




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
    policy_loss.backward(retain_graph=True)
    optimizer.step()

def ppo_update(states, actions, log_probs_old, returns, values_old, epsilon=0.2, beta=0.001):
    #convert values_old to tensor:
    values_old = torch.tensor(values_old, dtype=torch.float32)
    #convert log_probs_old from list to tensor:
    log_probs_old = torch.tensor(log_probs_old, dtype=torch.float32)
    advantages = returns - values_old


    for _ in range(3): #lets just say 3 epochs for now.
        _, log_probs, values = fire_ac( torch.tensor(states, dtype=torch.float32) )

        ratios = torch.exp(log_probs - log_probs_old) #pi_theta / pi_theta_old
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - epsilon, 1 + epsilon) * advantages

        policy_loss = -torch.min(surr1, surr2).mean() #we do negative for gradient ascent
        value_loss = F.mse_loss(values, returns)

        optimizer.zero_grad()
        (policy_loss + value_loss * beta).backward()
        optimizer.step()

    
def compute_returns(rewards, gamma=0.99):
    returns = []
    Gt = 0
    for r in reversed(rewards):
        Gt = r + gamma * Gt
        returns.append(Gt)
    returns.reverse()
    returns = torch.tensor(returns)
    print(returns)
    return returns
    
timestep = 0

states, actions, rewards = [], [], []
values = []
log_probs = []

sock.sendto(bytes(MESSAGE, "utf-8"), (UDP_IP, UDP_PORT)) #one time send?

while True:
    try:
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
        values.append(value)


        timestep += 1
        print(timestep, action)
        if timestep == 500:
            returns = compute_returns(rewards)
            # reinforce_update(returns, log_probs)
            ppo_update(states, actions, log_probs, returns, values)
            timestep = 0
            states, actions, rewards = [], [], []
            values = []
            log_probs = []

    except socket.timeout:
        torch.save(fire_ac.state_dict(), "fire_ac2")
        print("Socket timeout")
        break



    