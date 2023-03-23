import numpy as np
import keras
import tensorflow as tf
import json
import sys

############# UPDATE FUNC #############

def randomUpdate(narray):
    return narray + np.random.uniform(-0.1, 0.1, narray.shape)

def randomChoiceWeights(narray):
    return np.random.choice(narray.reshape(narray.size), narray.size).reshape(narray.shape)

def fullRandomWeights(narray):
    return np.random.uniform(-1, 1, narray.size).reshape(narray.shape)


############# UTILS #############

def clearObservation(observation):
    obs = np.zeros((96, 96))
    for x in range(0, 96):
        for y in range(0, 96):
            obs[x][y] = observation[x][y].mean()

    return obs.reshape(1, 96, 96)

def convertToNumpy(arr):
    score = arr.pop("score")
    for x in arr.keys():
        for y in arr[x].keys():
            arr[x][y] = np.array(arr[x][y])
    return arr, score

def loopAll(weights: dict, func) -> dict:
    res = {}

    for x in weights.keys():
        res[x] = {}
        for y in weights[x].keys():
            res[x][y] = func(weights[x][y])
    return res

def formatWeights(weights:dict) -> list:
    return [weights["weight"], weights["bias"]]

############# CLASS #############

class Memory:
    def __init__(self, capacity, inputShape, actionsNumber):
        # Ensure inputShape does not have None values
        self.inputShape = tuple(x for x in inputShape if x is not None)

        # print("Memory input shape: ", self.inputShape)
        self.capacity = capacity
        self.memoryCounter = 0
        self.memory = np.zeros((self.capacity, *self.inputShape))
        # print("Memory shape: ", self.memory.shape)
        self.newMemory = np.zeros((self.capacity, *self.inputShape))
        self.actionMemory = np.zeros((self.capacity, actionsNumber))
        self.rewardMemory = np.zeros(self.capacity)
        self.terminalMemory = np.zeros(self.capacity)

    def storeTransition(self, state, action, reward, newState, done):
        index = self.memoryCounter % self.capacity

        # print(f"Shape of state: {state.shape}")
        self.memory[index] = state
        self.newMemory[index] = newState
        self.actionMemory[index] = action
        self.rewardMemory[index] = reward
        self.terminalMemory[index] = done

        self.memoryCounter += 1

    def sampleBuffer(self, batchSize):
        maxMemory = min(self.memoryCounter, self.capacity)
        batch = np.random.choice(maxMemory, batchSize)

        states = self.memory[batch]
        actions = self.actionMemory[batch]
        rewards = self.rewardMemory[batch]
        newStates = self.newMemory[batch]
        dones = self.terminalMemory[batch]

        return states, actions, rewards, newStates, dones

class CriticNetwork(keras.Model):
    def __init__(self, model, name="critic", saveDir='tmp/ddpg'):
        super(CriticNetwork, self).__init__(name=name)

        self.saveDir = saveDir
        self.saveFile = self.saveDir + "/" + self.name + ".h5"

        ## Critic Network with state and action as input
        self.model = keras.Sequential()

        self.model.add(keras.layers.Dense(64, activation="relu", input_shape=(64, 9219)))
        self.model.add(keras.layers.Dense(32, activation="relu"))
        self.model.add(keras.layers.Dense(1, activation="linear"))

        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")

    def __call__(self, state, action):
        ## Flatten state and action
        state = tf.reshape(state, (state.shape[0], -1))
        action = tf.reshape(action, (action.shape[0], -1))

        ## Join state and action
        entry = np.concatenate((state, action), axis=1)

        actionValue = self.model.layers[0](entry, training=True)
        for layer in self.model.layers[1:]:
            actionValue = layer(actionValue, training=True)

        return actionValue

class ActorNetwork(keras.Model):
    def __init__(self, model, name="targetActor", saveDir='tmp/ddpg', actionsNumber=3):
        super(ActorNetwork, self).__init__(name=name)

        self.saveDir = saveDir
        self.saveFile = self.saveDir + "/" + self.name + ".h5"

        # Create actor model from brain model
        self.model = keras.Sequential()
        for layer in model.layers:
            self.model.add(layer)
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")

    def __call__(self, state):
        actionValue = self.model.layers[0](state, training=True)
        for layer in self.model.layers[1:]:
            actionValue = layer(actionValue, training=True)

        return actionValue

class estimator:
    def __init__(self) -> None:
        self.name = "DDPG"
        self.gamma = 0.99
        self.tau = 0.005
        self.batchSize = 64
        self.noise = 0.1
        ## raise NotImplementedError(f"{self.name} is not implemented yet")

    def __str__(self) -> str:
        return self.name

    def setup(self, brain: object):
        self.memory = Memory(100, brain.model.input_shape, 3)

        # Clone the model before passing it to each network (to prevent reference issues)
        actor_model = keras.models.clone_model(brain.model)
        target_actor_model = keras.models.clone_model(brain.model)
        critic_model = keras.models.clone_model(brain.model)
        target_critic_model = keras.models.clone_model(brain.model)

        self.actor = ActorNetwork(model=actor_model, name="actor")
        self.targetActor = ActorNetwork(model=target_actor_model, name="targetActor")
        self.critic = CriticNetwork(model=critic_model, name="critic")
        self.targetCritic = CriticNetwork(model=target_critic_model, name="targetCritic")

    def memorize(self, observation=None, step=None, reward=None, nextObservation=None, check=False):
        if check:
            return
        clearedObservation = clearObservation(observation)
        clearedNextObservation = clearObservation(nextObservation)
        self.memory.storeTransition(clearedObservation, step, reward, clearedNextObservation, False)

    def loadModels(self):
        print('Loading models...')
        self.targetActor.model.load_weights(self.targetActor.saveFile)
        self.critic.model.load_weights(self.critic.saveFile)
        self.targetCritic.model.load_weights(self.targetCritic.saveFile)

    def updateNetworkParameters(self):
        weights = []
        targets = self.targetActor.weights
        for i, weight in enumerate(self.actor.model.weights):
            weights.append(weight * self.tau + targets[i]*(1-self.tau))
        self.targetActor.model.set_weights(weights)

        weights = []
        targets = self.targetCritic.weights
        for i, weight in enumerate(self.critic.model.weights):
            weights.append(weight * self.tau + targets[i]*(1-self.tau))
        self.targetCritic.model.set_weights(weights)

    def chooseAction(self, observation, evaluate=False):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        actions = self.actor(state)
        if not evaluate:
            actions += tf.random.normal(shape=[self.n_actions],
                                        mean=0.0, stddev=self.noise)  # Add noise for exploration

        actions = tf.clip_by_value(actions, -1, 1)

        return actions[0]

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

    def update(self, brain:object=None, score=None, model=None, check=False):
        if check:
            return

        if self.memory.memoryCounter < self.batchSize:
            return

        # Sample a batch from the memory buffer
        states, actions, rewards, newStates, dones = self.memory.sampleBuffer(self.batchSize)

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        newStates = tf.convert_to_tensor(newStates, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        max_gradient_norm = 1.0

        # Update critic (Q-value function)
        with tf.GradientTape() as tape:
            target_actions = self.targetActor(newStates)
            target_q_values = self.targetCritic(newStates, target_actions)
            y = rewards + self.gamma * target_q_values * (1 - dones)
            q_values = self.critic(states, actions)
            critic_loss = tf.reduce_mean(tf.square(y - q_values))
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        # Clip gradients to prevent exploding gradients
        critic_grads, _ = tf.clip_by_global_norm(critic_grads, max_gradient_norm)
        self.critic.model.optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        # Update actor (policy function)
        with tf.GradientTape() as tape:
            actions = self.actor(states)
            q_values = self.critic(states, actions)
            # For doing the tape.gradient, we need to use actions in the calcuation of the actor loss
            actor_loss = -tf.reduce_mean(q_values + actions)
        # Compute gradients
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)

        #print(f"actor_grads: {actor_grads}")
        if not None in actor_grads:
            self.actor.model.optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        # Update target networks with soft update
        self.updateNetworkParameters()
        return self.getAllWeights()