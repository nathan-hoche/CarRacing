#from stable_baselines3 import PPO      #PPO -> Proximal Policy Optimization
#from stable_baselines3.common.vec_env import DummyVecEnv 
#from stable_baselines3.common.evaluation import evaluate_policy  #to evaluate the model 
#from stable_baselines3.common.callbacks import EvalCallback

import numpy as np
import tensorflow as tf
import keras
import tensorflow_probability as tfp

############# HYPERPARAMETER #############

lr = 3e-4
gamma = 0.95
clip = 0.2
n_updates_per_iteration = 5

############# CLASS MODEL #############

class CriticModel(keras.Model):
    def __init__(self, model=None, name="critic", save_dir='saves/PPO'):
        super(CriticModel, self).__init__(name=name)

        self.save_dir = save_dir
        self.save_file = save_dir + "/" + self.name + ".h5"

        # Load critic model
        try: 
            self.model = keras.models.load_model(self.save_file)
            return
        except:
            pass

        if model is not None:
            self.model = keras.models.clone_model(model)
            self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse")
            return 
        
        state_input = keras.Input(shape=(1, 96, 96))
        action_input = keras.Input(shape=(3))

        state_input = keras.layers.Input(shape=(1, 96, 96), name='state')
        action_input = keras.layers.Input(shape=(3), name='action')

        flatten = keras.layers.Flatten()
        concat = keras.layers.Concatenate()

        dense1 = keras.layers.Dense(256, activation='relu')
        dense2 = keras.layers.Dense(128, activation='relu')
        dense3 = keras.layers.Dense(64, activation='relu')
        q_value_output = keras.layers.Dense(1, activation=None, name='q_value')

        self.model = keras.Model(inputs=[state_input, action_input], outputs=q_value_output(dense3(dense2(dense1(concat([flatten(state_input), action_input]))))))
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse")
    
    def call(self, state, action):
        q_value = self.model([state, action], training=True)
        return q_value

class ActorModel(keras.Model):
    def __init__(self, model, name="actor", save_dir='saves/PPO'):
        super(ActorModel, self).__init__(name=name)

        self.save_dir = save_dir
        self.save_file = save_dir + "/" + self.name + ".h5"

        ## Load actor model
        try:
            self.model = keras.models.load_model(self.saveFile)
            return
        except:
            pass

        self.model = keras.models.clone_model(model)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse")

    def __call__(self, state):
        return self.model(state, training=True)



############# BUFFER #############

class Buffer:
    def __init__(self):
        self.buffer_obs = []
        self.buffer_action = []
        self.buffer_logprob = []
        self.buffer_reward = []
        self.buffer_rtgs = []
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
        return buffer_rtgs
    
    def get_data(self):
        self.buffer_rtgs = self.compute_rtgs(self.buffer_reward)   
        return (self.buffer_obs, self.buffer_action, self.buffer_logprob, self.buffer_rtgs)
    
    def store_data(self, observation, action, log_prob, reward):
        self.buffer_obs.append(observation)
        self.buffer_action.append(action)
        self.buffer_logprob.append(log_prob)
        self.buffer_reward.append(reward)
        return

############# UTILS ##############

def clearObservation(observation):
    obs = np.zeros((96, 96))
    for x in range(0, 96):
        for y in range(0, 96):
            obs[x][y] = observation[x][y].mean()

    return obs.reshape(1, 1, 96, 96)



############# CLASS #############

class estimator:
    def __init__(self) -> None:
        self.name = "PPO"
        self.bestWeights, self.bestScore = None, -1

    def __str__(self) -> str:
        return self.name        

    def setup(self, brain: object):
        self.buffer = Buffer()

        self.actor = ActorModel(model=brain.model, name=("actor" + "_" + brain.name))
        self.critic = CriticModel(name=("critic" + "_" +  brain.name))

        # Initialize the covariance matrix used to query the actor for actions
        cov_var = np.full((3,), fill_value=0.5)
        cov_mat = np.diag(cov_var)
        self.cov_mat_32 = tf.cast(tf.constant(cov_mat), dtype=tf.float32)
        return


    def memorize(self, observation=None, action=None, reward=None, nextObservation=None, check=False):
        if check:
            return
        observation = clearObservation(observation)

        dist = tfp.distributions.MultivariateNormalFullCovariance(loc=action, covariance_matrix=self.cov_mat_32)
        log_prob = dist.log_prob(action)
        # Track recent observation, reward, action, and action log probability
        self.buffer.store_data(observation, action, log_prob, reward)
        

    def update(self, brain:object=None, score=None, model=None, check=False):
        if check:
            return
        (buffer_obs, buffer_action, buffer_logprob, buffer_rtgs) = self.buffer.get_data()

        # Calculate advantage at k-th iteration
        V = self.evaluate(buffer_obs)
        A_k = buffer_rtgs - np.asarray(V)
        A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

        for _ in range(n_updates_per_iteration): 
            # Update the actor
            self.train_actor(buffer_obs, buffer_logprob, A_k)
            
            # Update the critic
            self.train_critic(buffer_obs, buffer_rtgs)

        if score > self.bestScore:
            self.bestScore = score
            self.saveNetworks(score, brain)
        return self.getAllWeights()
    
    @tf.function
    def train_actor(self, buffer_obs, buffer_logprob, A_k):
        with tf.GradientTape() as tape:
            curr_log_probs = []
            for obs in buffer_obs:
                # Calculate the log probabilities of batch actions using most recent actor network.
                # This segment of code is similar to that in get_action()
                mean = self.actor(obs)
                dist = tfp.distributions.MultivariateNormalFullCovariance(loc=mean, covariance_matrix=self.cov_mat_32)
                curr_log_probs.append(dist.log_prob(mean))

            ratios = tf.exp(tf.cast(curr_log_probs, dtype=tf.float32) - tf.cast(buffer_logprob, dtype=tf.float32))
            surr1 = ratios * tf.cast(A_k, dtype=tf.float32)
            surr2 = tf.clip_by_value(ratios, 1 - clip, 1 + clip) * tf.cast(A_k, dtype=tf.float32)
            actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

        # Compute gradients and update the actor weights
        grads = tape.gradient(actor_loss, self.actor.trainable_weights)
        self.actor.model.optimizer.apply_gradients(zip(grads, self.actor.trainable_weights))

    @tf.function
    def train_critic(self, buffer_obs, buffer_rtgs):
        with tf.GradientTape() as tape:
            V = self.evaluate(buffer_obs)
            critic_loss = tf.keras.losses.MeanSquaredError()(V, buffer_rtgs)

        grads = tape.gradient(critic_loss, self.critic.trainable_weights)
        self.critic.model.optimizer.apply_gradients(zip(grads, self.critic.trainable_weights))
                                                 
    def evaluate(self, batch_obs):
        # Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
        V = []
        for obs in batch_obs:
            # get only the value from the critic
            act = self.actor(obs)
            critic = self.critic(obs, act)
            V.append(critic[0][0])
        return V
    
    def saveNetworks(self, score, brain: object):
        self.actor.model.save(self.actor.save_file)
        self.critic.model.save(self.critic.save_file)
        brain.save(score)

    def getAllWeights(self):
        res = {}

        for layer in self.actor.model.layers:
            ## Ignore layers without weights or biases as they are not trainable
            if (not hasattr(layer, "get_weights") or not hasattr(layer, "get_biases")):
                continue
            res[layer.name] = {}
            # Get current layer weights
            res[layer.name]["weights"] = layer.get_weights()
            ## Get current layer biases
            res[layer.name]["biases"] = layer.get_biases()
        return res