##############
### Remark ###
##############

#export OPENAI_REMOTE_VERBOSE=0 ##put this in the terminal to remove verbose

#################
### Libraries ###
#################
import gym
import universe
import numpy as np
import tensorflow as tf

from collections import deque
import random
import time

from skimage.exposure import rescale_intensity
from skimage.transform import resize
from skimage import data
from skimage.color import rgb2grey
from skimage.color import rgb2gray


########################
### Global Variables ###
########################
"""
There are 12 possible defined actions. The original number of actions are #pixels * 2, meaning we can
press any pixel on the screen to define directions and either press space at the same time or not. In order
to lower the number of allowed actions and still get a smooth direction control we decide to define 12 regions
in the border of the screen where we will press either the center of the region or a random pixel in that region
(depending on the strategy taken) so as to move in that direction.
"""
##y = 85-386, x = 18-522 --> screen borders
topleft = (19,86)
bottomright = (522,386)
div_x = 5
div_y = 3
interx = int((bottomright[0]-topleft[0])/div_x)
intery = int((bottomright[1]-topleft[1])/div_y)
pointers = ([(topleft[0],topleft[1]+intery*i) for i in range(div_y-1,-1,-1)]+
           [(topleft[0]+interx*i,topleft[1]) for i in range(1,div_x-1)]+
           [(topleft[0]+interx*4,topleft[1]+intery*i) for i in range(div_y)]+
           [(topleft[0]+interx*i,topleft[1]+intery*2) for i in range(div_x-2,0,-1)])

ACTIONS = 12 # number of valid actions
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

#################
### Functions ###
#################

def idx2act(idx):
    p = [[('PointerEvent', x+int(interx/2), y+int(intery/2), 0)] for (x,y) in pointers]
    return p[idx]

def preprocess_obs_ini(obs_n):
    color_s_t = obs_n[0]['vision'][85:386,18:522,:]
    grey_s_t = rgb2grey(color_s_t)
    s_t = resize(grey_s_t, (80,80))
    s_t = rescale_intensity(s_t,out_range=(0,255))
    s_t = np.stack((s_t, s_t, s_t, s_t), axis=2)
    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])
    return s_t

def preprocess_obs(obs_n):
    color_s_t = obs_n[0]['vision'][85:386,18:522,:]
    grey_s_t = rgb2grey(color_s_t)
    s_t = resize(grey_s_t, (80,80))
    s_t = rescale_intensity(s_t,out_range=(0,255))
    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], 1) #1x80x80x1
    #s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3) #TODO IN THE MAIN SECTION
    return s_t

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

    
#############
### Graph ###
#############

ConNet = tf.Graph()

with ConNet.as_default():
    image = tf.placeholder(tf.float32,shape = (None, 80,80,4))
    x_image = image
    #x_image = tf.reshape(image, [-1,80,80,4])
    actions_ref = tf.placeholder(tf.float32, shape = (None,ACTIONS))#not yet known

    #5x5 convolution layer, pool 2x2, depth 1 --> depth 32
    W_conv1 = weight_variable([8,8,4,32])
    b_conv1 = bias_variable([20,20,32])
    h_conv1 = tf.nn.relu(conv2d_1(x_image,W_conv1) + b_conv1)
    #h_pool1 = max_pool_2x2(h_conv1)

    #5x5 convolution layer, pool 2x2, depth 32 --> depth 64
    W_conv2 = weight_variable([4,4,32,64])
    b_conv2 = bias_variable([10,10,64])
    h_conv2 = tf.nn.relu(conv2d_2(h_conv1, W_conv2) + b_conv2)
    #h_pool2 = max_pool_2x2(h_conv2)

    #5x5 convolution layer, pool 2x2, depth 32 --> depth 64
    W_conv3 = weight_variable([3,3,64,64])
    b_conv3 = bias_variable([10,10,64])
    h_conv3 = tf.nn.relu(conv2d_3(h_conv2, W_conv3) + b_conv3)
    #h_pool3 = max_pool_2x2(h_conv2)

    input_dim = 10*10*64
    #Flatten the filtered images in a vector
    h_conv3_flat = tf.reshape(h_conv3, [-1, input_dim])

    keep_prob = tf.placeholder(tf.float32)
    #Fully connected layer of 1024 neurons (activation function: ReLU) with dropout(prob = keep_prob)
    W_fc1 = weight_variable([input_dim,512])
    b_fc1 = bias_variable([512])
    h_fc1=tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    #Fully connected layer into 2 labels (activation function: Linear)
    W_fc2 = weight_variable([512, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])
    q_values = tf.matmul(h_fc1_drop, W_fc2) + b_fc2 #real value

    #Loss function: cross-entropy with softmax loss; numerically stable way
    #loss = tf.losses.mean_squared_error(label,scores)
    loss_value = tf.reduce_mean(tf.squared_difference(q_values, actions_ref))
    #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = label, logits = scores))

    #Optimization algorithm: ADAM, see https://arxiv.org/abs/1412.6980
    train_step = tf.train.AdamOptimizer(1e-3).minimize(loss_value)

###############
### Session ###
###############

with tf.Session(graph = ConNet) as sess:
    env = gym.make('internet.SlitherIO-v0')
    env.configure(remotes=1)  # automatically creates a local docker container
    tf.global_variables_initializer().run() #init Q(s,a|w) with random weights
    D = deque() #initialize replay memory D
    epsilon = INITIAL_EPSILON
    OBSERVE = OBSERVATION
    verbose = True
    debug = False;
    t = 0
    reward100 = np.zeros(100)
    for i_episode in range(2000):
        if debug:
            break
        if i_episode == 0:
            observation_n = env.reset()
        while observation_n[0] == None:  # When Slither hasn't started
            action = idx2act(0)
            action_n = [action for ob in observation_n]
            observation_n, reward_n, done_n, info = env.step(action_n)
            env.render()
        state_p0 = preprocess_obs_ini(observation_n)
        rr = 0
        done_n = [False]
        echo_ts = 0
        while not done_n[0]:
            action_array = 0
            current_loss = 0
            env.render()
            echo_ts += 1
            t+=1
            #if echo_ts%10 == 0 and verbose:
            #    print(echo_ts)
            if random.random() < epsilon:
                actionidx = int(random.random()*12)
                action = idx2act(actionidx)
                action_n = [action for ob in observation_n]
                print("------------- Random Action -------------")
            else:
                feed = {image: state_p0, keep_prob: KEEP_PROB}
                print("------------- Not Random ----------------")
                action_array = sess.run(q_values, feed_dict = feed)
                if np.isnan(action_array[0][0]):
                    print("NAN IN THE HOLE!!!!")
                    debug = True
                actionidx = np.argmax(action_array)
                action = idx2act(actionidx)
                action_n = [action for ob in observation_n]
            if epsilon > FINAL_EPSILON:
                epsilon *= EPSILON_DECAY
            
            observation_n, reward_n, done_n, info_n = env.step(action_n)
            
            if observation_n[0] == None:
                continue;

            if done_n[0]:# Punish hard when failing
                reward_n[0] = -50
                #print(state_p0)
                state_p1 = np.zeros_like(state_p0)
                #print('FINALLY OVER!!!')
                #print(state_p1)
                D.append((state_p0,actionidx,reward_n[0],state_p1,done_n[0]))

            if not done_n[0]:
                rr += reward_n[0]
                state_p1 = preprocess_obs(observation_n)
                state_p1 = np.append(state_p1, state_p0[:, :, :, :3], axis=3)
                D.append((state_p0,actionidx,reward_n[0],state_p1,done_n[0]))
            if len(D) > REPLAY_MEMORY:
                D.popleft()
            if t>BATCH:
                minibatch = random.sample(D,BATCH)
                inputs = np.zeros((BATCH, state_p0.shape[1],state_p0.shape[2],state_p0.shape[3]))
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
                    feed_state_t = {image: state_t, keep_prob: KEEP_PROB}
                    feed_state_t1 = {image: state_t1, keep_prob: KEEP_PROB}
                    targets[i] = sess.run(q_values, feed_dict = feed_state_t)
                    Q_sa = sess.run(q_values, feed_dict = feed_state_t1)
                    if terminal:
                        targets[i, action_t] = reward_t
                    else:
                        targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa)
                    feed_train = {image: [inputs[i]], actions_ref: [targets[i]], keep_prob: KEEP_PROB}
                    _, current_loss = sess.run([train_step, loss_value], feed_dict = feed_train)
            if done_n[0]:
                break;
            state_p0 = state_p1
            monitor = ""
            if t <= BATCH:
                monitor = "observe"
            elif t > BATCH and t <= BATCH+EXPLORE:
                monitor = "explore"
            else:
                monitor = "train"
            if verbose and (t%30 == 0 or done_n[0]) :
                print("EPISODE:",i_episode,"/TOTAL TIMESTEP:",t,"/TIMESTEP:",echo_ts,"/STATE: ",monitor, \
                        "/EPSILON:",epsilon, "/ACTION: ",action,"/REWARD: ",rr, \
                        "/Q_MAX:", action_array,"/MEAN REWARD:",np.sum(reward100)/100 , \
                        "/Loss: ", current_loss)
        print("Episode {} finished after {} timesteps with score {}".format(i_episode,echo_ts,rr))    
        reward100[i_episode%100] = rr
        #if verbose:
        #    print("EPISODE:",i_episode,"/TIMESTEP:",t,"/STATE: ",monitor, \
        #                "/EPSILON:",epsilon, "/ACTION: ",action,"/REWARD: ",rr, \
        #                "/MEAN REWARD:",np.sum(reward100)/100 , \
        #                "/Loss: ", current_loss)