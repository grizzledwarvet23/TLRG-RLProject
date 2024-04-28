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

import onnxruntime as ort




UDP_IP = "127.0.0.1"
UDP_PORT = 5065
MESSAGE = "YOUR DATA"

training_cycles = 0

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

#set timeout to 5 seconds
sock.settimeout(10)

#create actor-critic network for PPO. just the network part for now:
class ActorCritic(nn.Module):
    def __init__(self,  state_dim, discrete=True, action_dim=1): #just the continuous case for now.
        super(ActorCritic, self).__init__()
        if discrete:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, action_dim), #1 action output.
                nn.Softmax(dim=-1) #softmax for discrete actions.
            )
            self.action_dim = action_dim
        else:
                self.actor = nn.Sequential(
                    nn.Linear(state_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 64),
                    nn.ReLU(),
                    nn.Linear(64, 2), #2 actions. a mean and a standard deviation.
                )
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            #use relu for critic.
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1) #1 value output.
        )
        self.discrete = discrete


    def forward(self, x, regular_use=True):
        if not self.discrete:
            raw_actor_output = self.actor(x)
            mean, log_std = raw_actor_output.chunk(2, dim=-1)
            log_std = torch.clamp(log_std, -20, 2)
            std = log_std.exp()
            dist = Normal(mean, std)
            sample = dist.rsample()
            action = torch.tanh(sample)
            value = self.critic(x) 
            #jacobian stuff copied from chatgpt
            log_prob = dist.log_prob(sample) - torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(-1, keepdim=True)  # Sum (or mean) over all action dimensions if necessary
            return action, dist.log_prob(action), value
    

        else:
            raw_actor_output = self.actor(x)
            dist = Categorical(logits=raw_actor_output)
            action = dist.sample()
            value = self.critic(x)
            # action_normalized = torch.tensor(2 * (action.item() / (self.action_dim - 1)) - 1)
            return action, dist.log_prob(action), value

state_dim = 2
#when discrete is false, action dim is irrelevant.
fire_ac = ActorCritic(state_dim, action_dim=3, discrete=False)



optimizer = optim.Adam(fire_ac.parameters(), lr=0.01)


def ppo_update(states, actions, log_probs_old, returns, values_old, epsilon=0.2, beta=0.01):
    #access the training cycles variable.

    states = torch.stack(tuple(torch.tensor(state, dtype=torch.float32) for state in states))
    actions = torch.stack(tuple(torch.tensor(action, dtype=torch.float32) for action in actions))
    log_probs_old = torch.stack(tuple(torch.tensor(log_prob, dtype=torch.float32) for log_prob in log_probs_old))
    returns = torch.stack(tuple(torch.tensor(r, dtype=torch.float32) for r in returns))
    values_old = torch.stack(tuple(torch.tensor(v, dtype=torch.float32) for v in values_old))

    # log_probs_old = torch.tensor(log_probs_old, dtype=torch.float32)

    means, log_stds = fire_ac.actor(states).chunk(2, dim=-1)
    values_new = fire_ac.critic(states)
    dists = torch.distributions.Normal(means, log_stds.exp())
    log_probs_new = dists.log_prob(actions).sum(axis=-1)

    advantages = returns - values_old.detach()

    ratios = (log_probs_old - log_probs_new).exp()

    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1 - epsilon, 1 + epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    critic_loss = F.mse_loss(values_new, returns)

    total_loss = policy_loss + critic_loss
    print(total_loss, policy_loss, critic_loss)


    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return total_loss.item()


    # for _ in range(3): #lets just say 3 epochs for now.
    #     log_probs = []
    #     values = []
    #     for state in states:
    #         state = torch.tensor(state, dtype=torch.float32)
    #         _, log_prob, value = fire_ac(state)
    #         log_probs.append(log_prob)
    #         values.append(value)
    #     log_probs = torch.tensor(log_probs, dtype=torch.float32)
    #     values = torch.tensor(values, dtype=torch.float32, requires_grad=True)
    #     ratios = torch.exp(log_probs - log_probs_old) #pi_theta / pi_theta_old

    #     surr1 = ratios * advantages
    #     surr2 = torch.clamp(ratios, 1 - epsilon, 1 + epsilon) * advantages


    #     policy_loss = -torch.min(surr1, surr2).mean() #we do negative for gradient ascent
    #     value_loss = F.mse_loss(values, returns)
    #     optimizer.zero_grad()
    #     (policy_loss + value_loss * beta).backward()
    #     optimizer.step()
    
def compute_returns(rewards, gamma=0.99):
    returns = []
    Gt = 0
    for r in reversed(rewards):
        Gt = r + gamma * Gt
        returns.append(Gt)
    returns.reverse()
    return returns
    
timestep = 0



states, actions, rewards = [], [], []
values = []
log_probs = []



sock.sendto(bytes(MESSAGE, "utf-8"), (UDP_IP, UDP_PORT)) #one time send?

# fire_model_path = 'models/FireRotate.onnx' 
#fire_model_path = 'models/FireDivided.onnx'
# fire_model_path = 'models/FireDividedTimePenalty.onnx'
# fire_model_path = 'models/Fire6Divided.onnx'
# fire_model_path = 'models/Fire9Divided.onnx'
fire_model_path = 'models/Fire15Divided.onnx'

fire_session = ort.InferenceSession(fire_model_path)

fire_input_name = fire_session.get_inputs()[0].name
fire_input_shape = fire_session.get_inputs()[0].shape
fire_input_type = fire_session.get_inputs()[0].type

fire_action_mask = fire_session.get_inputs()[1].name

#crop_session = 'models/CropSorterNew.onnx'
crop_session = 'models/CropSorterDivided(Cont+Discrete).onnx'

crop_session = ort.InferenceSession(crop_session)

crop_input_name = crop_session.get_inputs()[0].name
crop_input_shape = crop_session.get_inputs()[0].shape
crop_input_type = crop_session.get_inputs()[0].type

crop_action_mask = crop_session.get_inputs()[1].name


# print(crop_input_name)
# print(crop_input_shape)
# print(crop_input_type)

#now to output information:
crop_output_name = crop_session.get_outputs()[0].name
crop_output_shape = crop_session.get_outputs()[0].shape
crop_output_type = crop_session.get_outputs()[0].type



#then the other output:
crop_output_name2 = crop_session.get_outputs()[1].name
crop_output_shape2 = crop_session.get_outputs()[1].shape
crop_output_type2 = crop_session.get_outputs()[1].type


# print(crop_session.get_outputs()[0])
# print(crop_session.get_outputs()[1])
# print(crop_session.get_outputs()[2])
# print(crop_session.get_outputs()[3])
# print(crop_session.get_outputs()[4])



while True:
    try:
        data, addr = sock.recvfrom(1024)
        data = data.decode("utf-8")
        data_dict = json.loads(data)
        
 
        fire_state_value = data_dict["PlayerHealth"] + data_dict["EnemyData"][:12]
        #also, add 3 more zeris to fire_state_value to make it 12.
        state = torch.tensor(fire_state_value, dtype=torch.float32)
        # action, log_prob, value = fire_ac(state)

        crop_state_value = data_dict["CropData"][:2]
        top_decision = data_dict["TopDecision"]

        #0 = water, 1 = fire.
        action_str = "Action: "
        if top_decision == 0:
            #THIS IS FOR WATER CASE. COMMENTED OUT DURING ROCK TESTING.
            crop_input_data = np.array([crop_state_value], dtype=np.float32)
            #we also have to put action_masks
            #action masks says got 1, expected 2 so lets fix it:
            action_mask = np.array([1, 1, 1], dtype=np.float32).reshape(1, 3)
            crop_outputs = crop_session.run(None, {crop_input_name: crop_input_data, crop_action_mask: action_mask})
            crop_continuous_action = crop_outputs[2][0] 
            crop_discrete_action = crop_outputs[5][0]
            action_str += "Water: " + json.dumps(crop_discrete_action.tolist()) + " " + json.dumps(crop_continuous_action.tolist())
        elif top_decision == 1:
            action_mask = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32).reshape(1, 15)
            fire_input_data = np.array([fire_state_value], dtype=np.float32)
            fire_outputs = fire_session.run(None, {fire_input_name: fire_input_data, fire_action_mask: action_mask})
            fire_continuous_action = fire_outputs[2][0]
            fire_discrete_action = fire_outputs[5][0] #lets test this with discrete deterministic.
            action_str += "Fire: " + json.dumps(fire_discrete_action.tolist()) + " " + json.dumps(fire_continuous_action.tolist())
    

        sock.sendto(bytes(json.dumps(action_str), "utf-8"), (UDP_IP, UDP_PORT))

        # log_probs.append(log_prob)
        # states.append(fire_state_value)
        # actions.append(action)
        # rewards.append(data_dict["Reward"])
        # values.append(value)

        # timestep += 1
        # if timestep == 256:
        #     total_reward = sum(rewards)
        #     print("UPDATE! Total reward: ", total_reward)
        #     print(rewards)
        #     returns = compute_returns(rewards)
        #     training_cycles += 1
        #     ppo_update(states, actions, log_probs, returns, values)
        #     timestep = 0
        #     states, actions, rewards = [], [], []
        #     values = []
        #     log_probs = []

    except socket.timeout:
        torch.save(fire_ac.state_dict(), "fire_ac2")
        print("Socket timeout")
        break



    