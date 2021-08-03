import tensorflow as tf
import numpy as np
import gym
import matplotlib.pyplot as plt
tf.compat.v1.disable_v2_behavior()

def one_hot(x):
    return np.identity(16)[x:x+1]

env = gym.make('FrozenLake-v0')


input_size = env.observation_space.n
output_size = env.action_space.n
learning_rate = 0.1

X = tf.compat.v1.placeholder(shape=[1,input_size],dtype = tf.float32)
W = tf.Variable(tf.compat.v1.random_uniform([input_size,output_size],0,0.01))

Qpred = tf.matmul(X,W)
Y = tf.compat.v1.placeholder(shape=[1,output_size],dtype=tf.float32)

loss = tf.reduce_sum(tf.square(Y-Qpred))

train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

dis = 0.99
num_episodes = 2000

rList = []
init = tf.compat.v1.global_variables_initializer()
with tf.compat.v1.Session() as sess:
    sess.run(init)

    for i in range(num_episodes):
        s = env.reset()
        e = 1./((i/50) + 10)
        rAll = 0
        done =False
        local_loss = []

        while not done:
            Qs = sess.run(Qpred,feed_dict={X:one_hot(s)})
            if np.random.rand(1) < e:
                a = env.action_space.sample()
            else:
                a = np.argmax(Qs)

            s1,reward,done,_ = env.step(a)

            if done:
                Qs[0,a] = reward
            else:
                Qs1 = sess.run(Qpred,feed_dict={X:one_hot(s1)})
                Qs[0,a] = reward + dis * np.max(Qs1)

            sess.run(train,feed_dict={X:one_hot(s),Y:Qs})

            rAll += reward

        rList.append(rAll)


print("Success rate: " + str(sum(rList) / num_episodes) +'%')

# num_episodes 중 실행한 episodes 출력
plt.bar(range(len(rList)), rList, color="blue")
plt.show()
