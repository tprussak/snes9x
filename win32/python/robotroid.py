import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
#from tensorflow.keras import models

input_size = 41839

# Used for output since prints get eaten
fout = open('python.out', 'w')

# set some constants
epsilon = 0.1
sigma = 0.1
epsilondecay = 0.99
maxFramesPerTrain = 30

# Define the model
model = tf.keras.Sequential([
  layers.Dense(5000, activation='tanh', input_shape=(input_size,)),
  layers.Dense(2500, activation='relu'),
  layers.Dense(1000, activation='tanh'),
  layers.Dense(200, activation='relu'),
  layers.Dense(12, activation='tanh')
])
# Optionally load weights from file
# model.load_weights('model.h5')

model.compile(loss='mean_squared_error', optimizer='sgd')

results = np.empty((1, 12))
inputs = np.empty((1, input_size))
inputHistory = []
resultHistory = []

def step(data):
    inputs = np.asarray([np.asarray(data)])
    inputHistory.append(inputs[0]);
    if np.random.rand() > epsilon:
        results = model.predict_on_batch(inputs);
    else:
        results = np.asarray([np.random.random_sample(12) - 0.5]);

    resultHistory.append(results[0]);
    output = 0;
    for i in range(12):
        if results[0][i] > 0:
            output = output + (1 << i)
    return output

totalReward = 0
def learn(reward):
    # fout.write(f'Received reward of: {reward}\n')
    global inputHistory, resultHistory, totalReward, epsilon, maxFramesPerTrain
    totalReward += reward
    # If there is 0 reward, we are paused, delete the last frame
    if(reward == 0):
        inputHistory.pop(len(inputHistory) - 1)
        resultHistory.pop(len(resultHistory) - 1)
    # Apply the reward to the individual frame
    for i in range(0, 12):
        resultHistory[len(resultHistory) - 1][i] *= reward
    
    # Run training
    if(reward > 0 or len(inputHistory) >= maxFramesPerTrain):
        # Update the results with the total reward
        for j in range(len(resultHistory)):
            for i in range(0, 12):
                resultHistory[j][i] = resultHistory[j][i] + resultHistory[j][i] * totalReward * sigma

        # Run training
        model.train_on_batch(np.asarray(inputHistory), np.asarray(resultHistory))
        inputHistory = []
        resultHistory = []
        if(reward > 0):
            epsilon *= epsilondecay
            maxFramesPerTrain += 1
        totalReward = 0

def test():
    step(np.ones(input_size).tolist())
    learn(-1)
    step(np.ones(input_size).tolist())
    learn(0)
    step(np.ones(input_size).tolist())
    learn(1)

def cleanup():
    fout.write('Cleaning up...')
    fout.close()
    model.save_weights('model.h5')
    
