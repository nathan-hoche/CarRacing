import numpy as np

#class PolicyNetwork(nn.Module):
#    def __init__(self, input_size, output_size, hidden_size=64):
#        super(PolicyNetwork, self).__init__()
#        self.fc1 = nn.Linear(input_size, hidden_size)
#        self.fc2 = nn.Linear(hidden_size, output_size)

#    def forward(self, x):
#        x = F.relu(self.fc1(x))
#        x = F.softmax(self.fc2(x), dim=1)
#        return x

#    def sample_action(self, observation):
#        observation = torch.from_numpy(observation).float().unsqueeze(0)
#        probabilities = self.forward(observation)
#        action_probs = torch.distributions.Categorical(probabilities)
#        action = action_probs.sample()
#        log_prob = action_probs.log_prob(action)
#        return action.item(), log_prob.item()

class estimator:
    def __init__(self) -> None:
        self.name = "VPG"
        self.memory = []
        self.gamma = 0.95
        self.epsilon = 0.01
        self.lr = 0.002
        self.batch_size = 10

    def __str__(self) -> str:
        return self.name
    
    def memorize(self, observation=None, step=None, reward=None, nextObservation=None, check=False):
        if check:
            return
        self.memory.append([observation, step, reward, nextObservation])
    
    def update(self, brain:object=None, score=None, check=False):
        if check:
            return
        return self.VPG(brain, score)
    
    def VPG(self, brain:object=None, score=None):
        # get the list of observations, actions and rewards
        rewards = [x[2] for x in self.memory]
        #actions = [x[1] for x in self.memory]
        #observations = [x[0] for x in self.memory]

        # compute the discounted rewards
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add

        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)

        # get the weights
        weights = brain.getAllWeights()

        # update the weights
        for actual in self.memory:
            gradients = brain.predict(actual[0])
            for x in range(len(gradients)):
                for key in weights:
                    weights[key]["weight"] += self.epsilon * gradients[0][x] * discounted_rewards[x]
                    weights[key]["bias"] += self.epsilon * gradients[0][x] * discounted_rewards[x]

        return weights