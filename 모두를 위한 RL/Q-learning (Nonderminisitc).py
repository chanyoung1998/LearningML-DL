import numpy as np
import gym
import random as pr
import matplotlib.pyplot as plt

'''gym.envs.registration.register(id='FrozenLake-v3',
entry_point='gym.envs.toy_text:FrozenLakeEnv', kwargs={'map_name':'4x4', 'is_slippery':True})
'''
env = gym.make('FrozenLake-v0')

Q = np.zeros([env.observation_space.n,env.action_space.n])
num_episodes = 2000
learning_rate = 0.85
discounted = 0.99

rList = []
for i in range(num_episodes):
    state = env.reset()
    rAll = 0
    done = False

    while not done:
        action = np.argmax(Q[state,:] + np.random.randn(1,env.action_space.n) / (i+1))
        new_state,reward,done,_ = env.step(action)
        #update q-table
        Q[state,action] = (1-learning_rate) * Q[state,action] + learning_rate * (reward + discounted * np.max(Q[new_state,:]))

        rAll += reward
        state = new_state

    rList.append(rAll)

print("Success rate: " + str(sum(rList) / num_episodes))
print("Final Q-Table Values")
print("LEFT DOWN RIGHT UP")
print(Q)

# num_episodes 중 실행한 episodes 출력
plt.bar(range(len(rList)), rList, color="blue")
plt.show()
