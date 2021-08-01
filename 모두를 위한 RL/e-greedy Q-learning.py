import numpy as np
import gym
import random as pr
import matplotlib.pyplot as plt

# https://gist.github.com/stober/1943451

# chooses randomly among eligible maximum indices
def rargmax(vector):
    m = np.amax(vector)
    # np.nonzero() : 기본은 0이 아닌 index들을 반환한다. 하지만 매개변수로 조건이 들어간다면
    # 해당 조건을 만족시키는 index를 반환한다
    # (array([0, 1, 2, 3], dtype=int64),): np.nonzero(m == vector) return 값
    indices = np.nonzero(m == vector)[0]

    return pr.choice(indices)

gym.envs.registration.register(id='FrozenLake-v3',
entry_point='gym.envs.toy_text:FrozenLakeEnv', kwargs={'map_name':'4x4', 'is_slippery':False})

env = gym.make('FrozenLake-v3')

Q = np.zeros([env.observation_space.n,env.action_space.n])
num_episodes = 2000

rList = []
for i in range(num_episodes):
    state = env.reset()
    rAll = 0
    done = False

    while not done:
        e = 0.1 / (i + 1)
        if pr.random() < e:
            # exploration
            action = env.action_space.sample()
        else:
            # exploit
            action = rargmax(Q[state, :])

        new_state,reward,done,_ = env.step(action)
        #update q-table
        Q[state,action] = reward + np.max(Q[new_state,:])

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
