import numpy as np

class estimator:
    def __init__(self) -> None:
        self.name = "VPG"
        self.memory = []
        self.gamma = 0.99
        self.epsilon = 0.9

    def __str__(self) -> str:
        return self.name
    
    def memorize(self, observation=None, step=None, reward=None, nextObservation=None, check=False):
        if check:
            return
        self.memory.append([observation, step, reward, nextObservation])
    
    def update(self, brain:object=None, score=None, check=False):
        if check:
            return
        return self.VPG(brain)
    
    def VPG(self, brain:object):
        
        # get the list of observations, actions and rewards
        rewards = [x[2] for x in self.memory]

        # compute the discounted rewards
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add

        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)

        weights = brain.getAllWeights()

        for actual in self.memory:
            gradients = brain.predict(actual[0])
            for x in range(len(gradients)):
                for key in weights:
                    weights[key]["weight"] += self.epsilon * gradients[0][x]
                    weights[key]["bias"] += self.epsilon * gradients[0][x]
        return weights