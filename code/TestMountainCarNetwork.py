import gym
import matplotlib.pyplot as plt
import numpy as np
import random as rand
from os.path import exists

env = gym.make('MountainCarContinuous-v0')
filepath = "/Users/henrydemarest/Documents/Random Coding Experiments/MachineLearningExperiments/OpenAI-Gym-MountainCar/Saved Networks/MC_gens-20_children-100_layers-0_layerHeight-3_networkTests-5_wMax-100_bMax-100.txt"

#open file:
file = open(filepath, "r")
lines = file.readlines()
file.close()

#get constants:
constants = lines[0].split(",")
neurons_per_layer = int(constants[0])
num_layers = int(constants[1])
output_space = int(constants[2])
NN = np.zeros((neurons_per_layer*num_layers+output_space, neurons_per_layer, 2))


#get Neural network:
for l in range(neurons_per_layer*num_layers+output_space):
    sets = lines[l+1].split(";")
    for i in range(neurons_per_layer):
        nums = sets[i].split(",")
        NN[l][i][0] = float(nums[0])
        NN[l][i][1] = float(nums[1])

#get old performances
if neurons_per_layer*num_layers+output_space + 1 < len(lines):
    print("\nPrevious performances:")
    num = 1
    for p in range(neurons_per_layer*num_layers+output_space+1, len(lines)):
        performance = lines[p].split(",")
        old_success_rate = float(performance[0])
        old_avg_reward = float(performance[1])
        old_tests_run = int(performance[2])
        print("- Saved Performance #", num, "- total tests =", old_tests_run, "success rate =", old_success_rate, ", average reward =", old_avg_reward)
        num+=1
    print()

#other constants:
num_testing_iterations = 1000
num_display_iterations = 5

#environment constants:
max_episode_length = 999


# DEFINING FUNCTIONS:
#Get output of neuronal network:
def neuralNetworkOutput(NN, input):
    last_layer = input
    for l in range(num_layers):
        next_layer = np.zeros(neurons_per_layer)
        for i in range(neurons_per_layer):
            value = 0
            for j in range(len(last_layer)):
                value += NN[l*neurons_per_layer + i][j][0] * last_layer[j] +  NN[l*neurons_per_layer + i][j][1]
            next_layer[i] = value
        last_layer = next_layer
    output = 0
    for i in range(len(last_layer)):
        output += last_layer[i] * NN[num_layers*neurons_per_layer][i][0] + NN[num_layers*neurons_per_layer][i][1]
    return output

print("Testing Model...")
print()
successes = 0
avg_reward = 0
for iteration in range(num_testing_iterations):
    obs = env.reset()
    total_reward = 0
    for step in range(max_episode_length):
        obsArray = np.array(obs)
        action = neuralNetworkOutput(NN, obsArray)
        if action > 1:
            action = 1
        elif action < -1:
            action = -1
        obs, reward, done, info = env.step([action])
        total_reward += reward
        if done:
            break
    if total_reward > 0:
        successes += 1
    avg_reward += total_reward

avg_reward /= num_testing_iterations
print("Model Results:")
print()
print("Final Success rate =", successes, "/", num_testing_iterations, "=", (successes/num_testing_iterations*100),"%")
print("Average Reward =", avg_reward)
print()

#Displaying Final Model
print("Displaying Sample Runs...\n")
for iteration in range(num_display_iterations):
    obs = env.reset()
    for step in range(max_episode_length):
        obsArray = np.array(obs)
        action = neuralNetworkOutput(NN, obsArray)
        if action > 1:
            action = 1
        elif action < -1:
            action = -1
        obs, reward, done, info = env.step([action])
        env.render()
        if done:
            break

#Save performance:
print("Saving performance...")
file = open(filepath, "a")

lastline = []
lastline.append(str(successes/num_testing_iterations*100)+",")
lastline.append(str(avg_reward)+",")
lastline.append(str(num_testing_iterations)+"\n")
file.writelines(lastline)
file.close()

env.close()