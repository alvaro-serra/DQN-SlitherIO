from collections import deque
import tensorflow as tf
import numpy as np
import gym
import random



ACTIONS = 2 # number of valid actions
GAMMA = 0.95 # decay rate of past observations
OBSERVATION = 1000. # timesteps to observe before training
EXPLORE = 3000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.01 # final value of epsilon
EPSILON_DECAY = 0.995
INITIAL_EPSILON = 1.0 # starting value of epsilon
REPLAY_MEMORY = 320000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1
LEARNING_RATE = 0.001
KEEP_PROB = 0.9


#img_rows , img_cols = 80, 80
#Convert image into Black and white
#img_channels = 4 #We stack 4 frames

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

def conv2d_1(x,W):
    return tf.nn.conv2d(x, W, strides = [1,4,4,1], padding = 'SAME')

def conv2d_2(x,W):
    return tf.nn.conv2d(x, W, strides = [1,2,2,1], padding = 'SAME')

def conv2d_3(x,W):
    return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool_2x2(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

fConNet = tf.Graph()

with fConNet.as_default():
    input_dim = 4
    state = tf.placeholder(tf.float32,shape = (None, input_dim))
    #x_image = image
    #x_image = tf.reshape(image, [-1,80,80,4])
    actions_ref = tf.placeholder(tf.float32, shape = (None,2))#not yet known
    
    keep_prob = tf.placeholder(tf.float32)
    #Fully connected layer of 300 neurons (activation function: ReLU) with dropout(prob = keep_prob)
    W_fc1 = weight_variable([input_dim,24])
    b_fc1 = bias_variable([24])
    h_fc1=tf.nn.relu(tf.matmul(state, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    #Fully connected layer into 150 neurons (activation function: Relu)
    W_fc2 = weight_variable([24, 24])
    b_fc2 = bias_variable([24])
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2) #real value
    h_fc2_drop = tf.nn.dropout(h_fc2,keep_prob)
    
    #Fully connected layer into 2 actions (activation function: Linear)
    W_fc3 = weight_variable([24,2])
    b_fc3 = bias_variable([2])
    actions_pred = tf.matmul(h_fc2_drop,W_fc3)+b_fc3
    
    #loss_value = tf.reduce_mean(tf.squared_difference(actions_pred, actions_ref))
    loss_value = tf.losses.mean_squared_error(labels = actions_ref, predictions = actions_pred)

    #Optimization algorithm: ADAM, see https://arxiv.org/abs/1412.6980
    train_step = tf.train.AdamOptimizer(learning_rate = LEARNING_RATE).minimize(loss_value)
    #train_step = tf.train.RMSPropOptimizer(learning_rate = LEARNING_RATE).minimize(loss_value)
    #train_step = tf.train.GradientDescentOptimizer(learning_rate = LEARNING_RATE).minimize(loss_value)


with tf.Session(graph = fConNet) as sess:
    env = gym.make('CartPole-v0')
    env._max_episode_steps = None
    tf.global_variables_initializer().run() #init Q(s,a|w) with random weights
    D = deque() #initialize replay memory D
    epsilon = INITIAL_EPSILON
    OBSERVE = OBSERVATION
    t = 0
    reward100 = np.zeros(100)
    for i_episode in range(2000):
        observation = env.reset()
        state_p0 = observation
        #state_p0 = np.append(state_p0,observation)
        #state_p0 = np.append(state_p0,observation)
        #state_p0 = np.append(state_p0,observation)
        state_p0 = [state_p0]
        rr = 0
        done = False
        while not done:
            action_array = 0
            current_loss = 0
            #env.render()
            if random.random() < epsilon:
                action = env.action_space.sample()
                #print("------------- Random Action -------------")
            else:
                feed = {state: state_p0, keep_prob: KEEP_PROB}
                action_array = sess.run(actions_pred, feed_dict = feed)
                action = np.argmax(action_array)
            if epsilon > FINAL_EPSILON:
                epsilon *= EPSILON_DECAY
            
            observation, reward, done, info = env.step(action)
            if done:# Punish hard when failing
                reward = -50
                observation = np.zeros_like(observation)
            if not done:
                rr += reward
            state_p1 = observation
            #state_p1 = np.append(state_p1 , state_p0[0][0:-len(observation)])
            state_p1 = [state_p1]
            D.append((state_p0,action,reward,state_p1,done))
            if len(D) > REPLAY_MEMORY:
                D.popleft()
            if t>BATCH:
                minibatch = random.sample(D,BATCH)
                inputs = np.zeros((BATCH, len(state_p0[0])))
                targets = np.zeros((BATCH,ACTIONS))
                for i in range(0,len(minibatch)):
                    state_t = minibatch[i][0]            
                    action_t = minibatch[i][1]   #This is action index
                    reward_t = minibatch[i][2]
                    state_t1 = minibatch[i][3]
                    terminal = minibatch[i][4]
                    inputs[i] = state_t[0]
                    #print("inputs:",state_t)
                    #print("INPUTS:",inputs[i])
                    feed_state_t = {state: state_t, keep_prob: KEEP_PROB}
                    feed_state_t1 = {state: state_t1, keep_prob: KEEP_PROB}
                    targets[i] = sess.run(actions_pred, feed_dict = feed_state_t)
                    Q_sa = sess.run(actions_pred, feed_dict = feed_state_t1)
                    if terminal:
                        targets[i, action_t] = reward_t
                    else:
                        #print("TargetS_i:", targets[i])
                        #print("TARGETS_i:", targets[i,action_t])
                        targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa)
                        #print("Targets_i+1", targets[i])
                    feed_train = {state: [inputs[i]], actions_ref: [targets[i]], keep_prob: KEEP_PROB}
                    _, current_loss = sess.run([train_step, loss_value], feed_dict = feed_train)
            state_p0 = state_p1
            t = t+1
            monitor = ""
            if t <= BATCH:
                monitor = "observe"
            elif t > BATCH and t <= BATCH+EXPLORE:
                monitor = "explore"
            else:
                monitor = "train"
            #print("EPISODE:",i_episode,"/TIMESTEP:",t,"/STATE: ",monitor, \
                    #"/EPSILON:",epsilon, "/ACTION: ",action,"/REWARD: ",rr, \
                    #"/Q_MAX:", action_array,"/MEAN REWARD:",np.sum(reward100)/100 , \
                    #"/Loss: ", current_loss)
            if done:
                print("Episode {} finished after {} timesteps".format(i_episode,rr))    
                break
        reward100[i_episode%100] = rr
        #print("EPISODE:",i_episode,"/TIMESTEP:",t,"/STATE: ",monitor, \
                    #"/EPSILON:",epsilon, "/ACTION: ",action,"/REWARD: ",rr, \
                    #"/MEAN REWARD:",np.sum(reward100)/100 , \
                    #"/Loss: ", current_loss)
    

