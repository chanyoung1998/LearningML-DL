import gym
import tensorflow as tf
import numpy as np
import random
from collections import deque

# 4개의 입력, 2개의 출력, 2개의 은닉측은 각 각 24개의 뉴런을 가진다.
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(24,input_shape=(4,),activation=tf.nn.relu),
    tf.keras.layers.Dense(24,activation=tf.nn.relu),
    tf.keras.layers.Dense(2)
])
model.compile(optimizer='adam',loss='mean_squared_error')

score = []
memory = deque(maxlen=2000)

env = gym.make('CartPole-v0')
env.reset()

num_episodes = 1000
time_steps = 200
for i in range(num_episodes):
    state = env.reset()
    state = np.reshape(state,[1,4])
    eps = 1 / (i/50 + 10)

    reward_sum = 0
    for t in range(time_steps):
        if np.random.rand() < eps:
            action = np.random.randint(0,2)
        else:
            predict = model.predict(state)
            action = np.argmax(predict)

        new_state,reward,done,_ = env.step(action)
        new_state = np.reshape(new_state,[1,4])

        memory.append((state,action,reward,new_state,done))
        state = new_state

        if done or t == time_steps -1:
            #print('Episode',i,'Score',t+1)
            score.append(t+1)

    if i > 10:
        minibatch = random.sample(memory,16)

        for state,action,reward,new_state,done in minibatch:

            target = reward
            if not done:
                target = reward + 0.9 * np.amax(model.predict(new_state)[0])

            target_outputs = model.predict(state)

            target_outputs[0][action] = target
            model.fit(state, target_outputs, epochs=1, verbose=0)

env.close()
print(score)