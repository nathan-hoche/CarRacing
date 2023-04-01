import numpy as np
import tensorflow as tf
import keras
import tensorflow_probability as tfp

############# HYPERPARAMETER #############

lr = 0.002
gamma = 0.95
clip = 0.2
n_updates_per_iteration = 5

############# CLASS MODEL #############

class CriticModel(keras.Model):
    def __init__(self, model, name="critic", save_dir='saves/PPO'):
        super(CriticModel, self).__init__(name=name)

        self.save_dir = save_dir
        self.save_file = save_dir + "/" + self.name + ".h5"

        # Load critic model
        try: 
            self.model = keras.models.load_model(self.save_file)
            print("Critic model loaded")
            return
        except:
            pass

        self.model = keras.Sequential()
        for layer in model.layers:
            if layer == model.layers[-1]:
                self.model.add(keras.layers.Dense(1, activation='relu'))
            else:
                self.model.add(layer)
        
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse")
    
    def call(self, state):
        return self.model(state, training=True)

class ActorModel(keras.Model):
    def __init__(self, model, name="actor", save_dir='saves/PPO'):
        super(ActorModel, self).__init__(name=name)

        self.save_dir = save_dir
        self.save_file = save_dir + "/" + self.name + ".h5"

        ## Load actor model
        try:
            self.model = keras.models.load_model(self.save_file)
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
        self.buffer_act = []
        self.buffer_logprob = []
        self.buffer_rew = []
        self.buffer_rtg = []
        return 

    
    def store_data(self, observation, action, log_prob, reward):
        self.buffer_obs.append(observation)
        self.buffer_act.append(action)
        self.buffer_logprob.append(log_prob)
        self.buffer_rew.append(reward)
        return
    
    def reshape_data(self):
        self.buffer_logprob = tf.cast(self.buffer_logprob, dtype=tf.float32)
        self.buffer_rtg = self.compute_rtgs()
        return self.buffer_obs, self.buffer_act, self.buffer_logprob, self.buffer_rtg
    
    def compute_rtgs(self):
        batch_rtgs = []
        discounted_reward = 0 
        for rew in reversed(self.buffer_rew):
            discounted_reward = rew + discounted_reward * gamma
            batch_rtgs.insert(0, discounted_reward)
        return batch_rtgs
    
    def reset(self):
        self.buffer_obs = []
        self.buffer_act = []
        self.buffer_logprob = []
        self.buffer_rew = []
        self.buffer_rtg = []
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
        self.critic = CriticModel(model=brain.model, name=("critic" + "_" +  brain.name))
        self.cov_var = tf.fill(dims=(3,), value=0.5)
        self.cov_mat = tf.linalg.diag(self.cov_var)
        return

    def getAction(self, mean):
        dist = tfp.distributions.MultivariateNormalFullCovariance(loc=mean, covariance_matrix=self.cov_mat)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.numpy(), log_prob

    
    def memorize(self, observation=None, action=None, reward=None, nextObservation=None, check=False):
        if check:
            return
        observation = clearObservation(observation)
        action, log_prob = self.getAction(action)
        self.buffer.store_data(observation, action, log_prob, reward)

    def evaluate(self, buffer_obs):
        # Query critic network for a value V for each obs in batch_obs.
        V = []
        for obs in buffer_obs:
            V.append(self.critic(obs))
        return V

    def update(self, brain:object=None, score=None, model=None, check=False):
        if check:
            return
        (buffer_obs, buffer_act, buffer_logprob, buffer_rtg) = self.buffer.reshape_data()
        V = np.asarray(self.evaluate(buffer_obs))

        # Calculate advantage
        A_k = buffer_rtg - V
        A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

        for _ in range(n_updates_per_iteration): 
            self.train_actor(buffer_obs, buffer_act, buffer_logprob, A_k)
            self.train_critic(buffer_obs, buffer_rtg)

        if score > self.bestScore:
            self.bestScore = score
            self.saveNetworks(score, brain)

        self.buffer.reset()
        return self.getAllWeights()
    
    @tf.function
    def train_actor(self, buffer_obs, buffer_act, buffer_logprob, A_k):
        with tf.GradientTape() as tape:
            curr_log_probs = []
            for obs, act in zip(buffer_obs, buffer_act):
                mean = self.actor(obs)
                dist = tfp.distributions.MultivariateNormalFullCovariance(loc=mean, covariance_matrix=self.cov_mat)
                curr_log_probs.append(dist.log_prob(act))

            ratios = tf.exp(tf.cast(curr_log_probs, dtype=tf.float32) - buffer_logprob)
            surr1 = ratios * tf.cast(A_k, dtype=tf.float32)
            surr2 = tf.clip_by_value(ratios, 1 - clip, 1 + clip) * tf.cast(A_k, dtype=tf.float32)
            actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
            tf.print("ACTOR LOSS : ", actor_loss)

        grads = tape.gradient(actor_loss, self.actor.trainable_weights)
        self.actor.model.optimizer.apply_gradients(zip(grads, self.actor.trainable_weights))
        tf.print("GOOD BOY ACTOR !")

    @tf.function
    def train_critic(self, buffer_obs, buffer_rtg):
        with tf.GradientTape() as tape:
            V = tf.convert_to_tensor(self.evaluate(buffer_obs))
            critic_loss = tf.reduce_mean(tf.square(V - tf.convert_to_tensor(buffer_rtg)))
        grads = tape.gradient(critic_loss, self.critic.trainable_weights)
        self.critic.model.optimizer.apply_gradients(zip(grads, self.critic.trainable_weights))
        tf.print("GOOD BOY CRITIC !")
    
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
            res[layer.name]["weights"] = layer.get_weights()[0]
            ## Get current layer biases
            res[layer.name]["biases"] = layer.get_weights()[1]
        return res