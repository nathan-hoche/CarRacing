import numpy as np
import keras
import tensorflow as tf
import json
import sys
import pickle
import os

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
    def __init__(self, capacity, inputShape, actionsNumber, filePath=None):
        # Ensure inputShape does not have None values
        self.inputShape = tuple(x for x in inputShape if x is not None)
        self.filePath = filePath

        if filePath and os.path.isfile(filePath):
            self.load_memory(filePath)
        else:
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

    def saveMemory(self):
        if not self.filePath:
            return
        # Create the directory if it does not exist
        directory = os.path.dirname(self.filePath)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save only up to the memoryCounter index
        memory_end = min(self.memoryCounter, self.capacity)
        np.savez_compressed(
            self.filePath,
            memory=self.memory[:memory_end],
            newMemory=self.newMemory[:memory_end],
            actionMemory=self.actionMemory[:memory_end],
            rewardMemory=self.rewardMemory[:memory_end],
            terminalMemory=self.terminalMemory[:memory_end],
            memoryCounter=self.memoryCounter
        )

    def loadMemory(self):
        if not self.filePath:
            return
        with open(self.filePath, 'rb') as file:
            data = pickle.load(file)
            self.memoryCounter = data['memoryCounter']
            self.memory = data['memory']
            self.newMemory = data['newMemory']
            self.actionMemory = data['actionMemory']
            self.rewardMemory = data['rewardMemory']
            self.terminalMemory = data['terminalMemory']

class CriticNetwork(keras.Model):
    def __init__(self, model=None, name="critic", saveDir='saves/DDPG', learningRate=0.002):
        super(CriticNetwork, self).__init__(name=name)

        self.saveDir = saveDir
        self.saveFile = self.saveDir + "/" + self.name + ".h5"

        ## Load critic model
        try:
            self.model = keras.models.load_model(self.saveFile)
            return
        except:
            pass

        if model is not None:
            self.model = keras.models.clone_model(model)
            self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=learningRate), loss="mse")
            return

        kernelInitializer = keras.initializers.RandomUniform(minval=-0.003, maxval=0.003)

        self.state_input = keras.layers.Input(shape=(1, 96, 96), name='state')
        self.action_input = keras.layers.Input(shape=(3), name='action')

        self.flatten = keras.layers.Flatten()
        self.concat = keras.layers.Concatenate()

        self.dense1 = keras.layers.Dense(256, activation='relu')
        self.dense2 = keras.layers.Dense(256, activation='relu')
        self.dense3 = keras.layers.Dense(64, activation='relu')
        self.q_value_output = keras.layers.Dense(1, activation=None, name='q_value', kernel_initializer=kernelInitializer)

        self.model = keras.Model(inputs=[self.state_input, self.action_input], outputs=self.q_value_output(self.dense3(self.dense2(self.dense1(self.concat([self.flatten(self.state_input), self.action_input]))))))

        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=learningRate), loss="mse")

    def call(self, state, action):
        q_value = self.model([state, action], training=True)
        return q_value

class ActorNetwork(keras.Model):
    def __init__(self, model, name="targetActor", saveDir='saves/DDPG', actionsNumber=3, learningRate=0.001):
        super(ActorNetwork, self).__init__(name=name)

        self.saveDir = saveDir
        self.saveFile = self.saveDir + "/" + self.name + ".h5"

        ## Load critic model
        try:
            self.model = keras.models.load_model(self.saveFile)
            return
        except:
            pass

        ## Random very small weights to prevent high, misleading values
        kernelInitializer = keras.initializers.RandomUniform(minval=-0.003, maxval=0.003)

        # Create actor model from brain model
        self.model = keras.Sequential()
        for layer in model.layers:
            if layer == model.layers[-1]:
                self.model.add(keras.layers.Dense(actionsNumber, model.layers[-1].activation, kernel_initializer=kernelInitializer))
            else:
                self.model.add(layer)


        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=learningRate), loss="mse")

    def __call__(self, state):
        return self.model(state, training=True)

class estimator:
    def __init__(self) -> None:
        self.name = "DDPG"
        self.gamma = 0.99
        self.tau = 0.005
        self.batchSize = 128
        self.noise = 0.1
        self.bestScore = 0
        self.currentSimulation = 0

    def __str__(self) -> str:
        return self.name

    def setup(self, brain: object):
        self.brain = brain
        self.memory = Memory(100000, brain.model.input_shape, 3, filePath="saves/DDPG/memory.pickle")

        self.actor = ActorNetwork(model=brain.model, name=("actor" + "_" +  brain.name))
        self.targetActor = ActorNetwork(model=self.actor.model, name=("targetActor" + "_" +  brain.name))
        self.critic = CriticNetwork(name=("critic" + "_" +  brain.name))
        self.targetCritic = CriticNetwork(model=self.critic.model, name=("targetCritic" + "_" +  brain.name))

    def memorize(self, observation=None, step=None, reward=None, nextObservation=None, check=False):
        if check:
            return
        reward *= 10
        clearedObservation = clearObservation(observation)
        clearedNextObservation = clearObservation(nextObservation)
        self.memory.storeTransition(clearedObservation, step, reward, clearedNextObservation, False)

    def learn(self):
        if self.memory.memoryCounter < self.batchSize:
            return

        # Sample a batch from the memory buffer
        states, actions, rewards, newStates, dones = self.memory.sampleBuffer(self.batchSize)

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        rewards = tf.cast(rewards, tf.float32)
        newStates = tf.convert_to_tensor(newStates, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        # Update critic (Q-value function)
        with tf.GradientTape() as tape:
            target_actions = self.targetActor(newStates)
            y = rewards + self.gamma * self.targetCritic(newStates, target_actions) * (1 - dones)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - self.critic(states, actions)))
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.model.optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        # Update actor (policy function)
        with tf.GradientTape() as tape:
            q_values = self.critic(states, self.actor(states))
            actor_loss = -tf.math.reduce_mean(q_values) # Maximize Q-value
        # Compute gradients
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)

        if not None in actor_grads:
            self.actor.model.optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        # Update target networks with soft update
        self.updateNetworkParameters()
        self.brain.train(self.getAllWeights())

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

    def saveNetworks(self, score, brain: object):
        self.targetActor.model.save(self.targetActor.saveFile)
        self.critic.model.save(self.critic.saveFile)
        self.targetCritic.model.save(self.targetCritic.saveFile)
        brain.save(score)

    def update(self, brain:object=None, score=None, model=None, check=False):
        if check:
            return
        self.currentSimulation += 1

        ## Save memory every 100 simulations
        ## if self.currentSimulation % 100 == 0:
        ##    self.memory.saveMemory()

        ## Save best network
        if score > self.bestScore:
            self.bestScore = score
            self.saveNetworks(score, brain)

        ## Set last memory state to done
        self.memory.terminalMemory[-1] = True
        self.learn()

        return self.getAllWeights()