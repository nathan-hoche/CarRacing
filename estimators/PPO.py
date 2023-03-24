#from stable_baselines3 import PPO      #PPO -> Proximal Policy Optimization
#from stable_baselines3.common.vec_env import DummyVecEnv 
#from stable_baselines3.common.evaluation import evaluate_policy  #to evaluate the model 
#from stable_baselines3.common.callbacks import EvalCallback

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
import keras
import torch
from torch.distributions import MultivariateNormal

############# HYPERPARAMETER #############

lr = 3e-4
gamma = 0.95
clip = 0.2
n_updates_per_iteration = 5

############# UTILS #############


def get_model_critic():
    state_input = keras.Input(shape=(1, 96, 96), dtype=tf.float32)
    # Classification block
    x = Flatten(name='flatten')(state_input)
    x = Dense(1024, activation='relu', name='fc1')(x)
    out_actions = Dense(1, activation='linear')(x)
  
    # Define model
    model = keras.Model(inputs=[state_input], outputs=[out_actions])
    model.compile(optimizer=keras.optimizers.Adam(lr=1e-4), loss='mse')
    
    return model

def clearObservation(observation):
    obs = np.zeros((96, 96))
    for x in range(0, 96):
        for y in range(0, 96):
            obs[x][y] = observation[x][y].mean()

    return obs.reshape(1, 1, 96, 96)



############# BUFFER #############

class Buffer:
    def __init__(self):
        self.buffer_obs = []
        self.buffer_action = []
        self.buffer_logprob = []
        self.buffer_reward = []
        self.buffer_rtgs = []
        self.buffer_lens = []
        self.ep_rews = []
        self.size = 0
        return 

    def store(self, observation, reward, actor, critic):
        return 


    def compute_rtgs(self, buffer_reward):
        buffer_rtgs = []
        discounted_reward = 0 # The discounted reward so far

        # Iterate through all rewards in the episode. We go backwards for smoother calculation of each
        # discounted return (think about why it would be harder starting from the beginning)
        for rew in reversed(buffer_reward):
            discounted_reward = rew + discounted_reward * gamma
            buffer_rtgs.insert(0, discounted_reward)

		# Convert the rewards-to-go into a tensor
        buffer_rtgs = torch.tensor(buffer_rtgs, dtype=torch.float)
        return buffer_rtgs
    
    def reshape_data(self):
        #self.buffer_obs = torch.tensor(self.buffer_obs, dtype=torch.float)
        #self.buffer_action = torch.tensor(self.buffer_action, dtype=torch.float)
        #self.buffer_logprob = torch.tensor(self.buffer_logprob, dtype=torch.float)
        self.buffer_rtgs = self.compute_rtgs(self.buffer_reward)   
        return (self.buffer_obs, self.buffer_action, self.buffer_logprob, self.buffer_rtgs)


############# CLASS #############

class estimator:
    def __init__(self) -> None:
        self.name = "Custom"
        self.bestWeights, self.bestScore = None, -1

    def __str__(self) -> str:
        return self.name        

    def setup(self, brain: object):
        self.buffer = Buffer()

        self.actor = keras.models.clone_model(brain.model)
        self.actor.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse")
        self.critic = get_model_critic()

        #self.actor_optim = keras.optimizers.Adam(self.actor.trainable_weights, lr=lr)
        #self.critic_optim = keras.optimizers.Adam(self.critic.trainable_weights, lr=lr)

        # Initialize the covariance matrix used to query the actor for actions
        self.cov_var = torch.full(size=(3,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)
        return
    
        
    def get_logprob(self, action):
        action_torch = torch.tensor(action, dtype=torch.float32)

        dist = MultivariateNormal(action_torch, self.cov_mat)

        # Calculate the log probability for that action
        log_prob = dist.log_prob(action_torch)
        return log_prob.detach()


    def memorize(self, observation=None, action=None, reward=None, nextObservation=None, check=False):
        if check:
            return
        observation = clearObservation(observation)
        # Track observations in this batch
        self.buffer.buffer_obs.append(observation)
        
        log_prob = self.get_logprob(action)
        
        # Track recent reward, action, and action log probability
        self.buffer.buffer_reward.append(reward)
        self.buffer.buffer_action.append(action)
        self.buffer.buffer_logprob.append(log_prob)
        self.buffer.size += 1


    def update(self, brain:object=None, score=None, model=None, check=False):
        if check:
            return
        (buffer_obs, buffer_action, buffer_logprob, buffer_rtgs) = self.buffer.reshape_data()
        print("Updated")
        #print("shape of buffer_obs: ", self.buffer.buffer_obs.shape)
        #print("shape of buffer_action: ", self.buffer.buffer_action.shape)
        #print("shape of buffer_logprob: ", self.buffer.buffer_logprob.shape)
        print("shape of buffer_rtgs: ", self.buffer.buffer_rtgs.shape)

        # Calculate advantage at k-th iteration
        V = self.evaluate(buffer_obs)
        A_k = buffer_rtgs - V.detach()      

        # One of the only tricks I use that isn't in the pseudocode. Normalizing advantages
		# isn't theoretically necessary, but in practice it decreases the variance of 
		# our advantages and makes convergence much more stable and faster. I added this because
		# solving some environments was too unstable without it.
        
        A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
        print("V ===== ", V)
        print("A_k ===== ", A_k)

        for _ in range(n_updates_per_iteration): 
            
            #print("actor_loss ===== ", actor_loss)
            #critic_loss = torch.nn.MSELoss()(V, buffer_rtgs)
            #print("critic_loss ===== ", critic_loss)

            # Calculate gradients and perform backward propagation for actor network
            #tape = tf.GradientTape()
            #critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
            #self.critic.model.optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
            # Calculate gradients and perform backward propagation for critic network

            with tf.GradientTape(persistent=True) as tape:
                V, curr_log_probs = self.evaluate_all(buffer_obs, buffer_action)
                delta = self.actor.trainable_variables
                # Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
                ratios = np.exp(np.array(curr_log_probs, dtype=np.float32) - np.array(buffer_logprob, dtype=np.float32))

                # Calculate surrogate losses.
                surr1 = ratios * np.array(A_k, dtype=np.float32)
                surr2 = np.clip(ratios, 1 - clip, 1 + clip) * np.array(A_k, dtype=np.float32)

                # Calculate actor and critic losses.
                #actor_loss = -np.mean(np.minimum(surr1, surr2))
                actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

            actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor.optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
            tape.reset()

            print("Updated")
            print(self.actor.weights)

        return 
    
    def evaluate(self, batch_obs):
        # Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
        V = []
        for obs in batch_obs:
            # get only the value from the critic
            V.append(self.critic(obs)[0][0])
        V = torch.tensor(np.asarray(V), dtype=torch.float)
        return V
    
    def evaluate_all(self, batch_obs, batch_acts):
        # Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
        V = []
        log_probs = []
        for obs, acts in zip(batch_obs, batch_acts):
            # get only the value from the critic
            V.append(self.critic(obs)[0][0])

		    # Calculate the log probabilities of batch actions using most recent actor network.
		    # This segment of code is similar to that in get_action()
            mean = torch.tensor(np.asarray(self.actor(obs)), dtype=torch.float)
            dist = MultivariateNormal(mean, self.cov_mat)
            log_probs.append(dist.log_prob(torch.tensor(acts, dtype=torch.float)))

        V = torch.tensor(np.asarray(V), dtype=torch.float)


		# Return the value vector V of each observation in the batch
		# and log probabilities log_probs of each action in the batch
        return V, log_probs