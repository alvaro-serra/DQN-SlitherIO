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

BATCH = 32 # size of minibatch
REPLAY_MEMORY = 1000000 # number of previous transitions to remember
GA #TODOMMA = 0.99 # decay rate of past observations
TARGET_NETWORK_UP_FREQ == 0:    ##                                          then Q'm = 10000 #TODO# frequency with which the target network is updated. Parameter C.
k = 4 #TODO
UPDATE_FREQ = 4 #TODO# update of the network
LEARNING_RATE = 0.00025 
GRADIENT_MOMENTUM = 0.95 
SQUARED_GRADIENT_MOMENTUM = 0.95 
MIN_SQUARED_GRADIENT = 0.01 
INITIAL_EPSILON = 1.0 # starting value of epsilon
EXPLORE = 1000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.10 # final value of epsilon

EPSILON_DECAY = (INITIAL_EPSILON - FINAL_EPSILON)/EXPLORE
OBSERVATION = 50000. #TODO# timesteps to observe before training


ACTIONS = 12 # number of valid actions
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
    actions_ref = tf.placeholder(tf.float32, shape = (None,ACTIONS))
    #TODO ## add placeholder with previous actions vector

    #8x8 convolution layer, depth 1 --> depth 32
    W_conv1 = weight_variable([8,8,4,32])
    b_conv1 = bias_variable([20,20,32])
    h_conv1 = tf.nn.relu(conv2d_1(x_image,W_conv1) + b_conv1)

    #4x4 convolution layer, depth 32 --> depth 64
    W_conv2 = weight_variable([4,4,32,64])
    b_conv2 = bias_variable([10,10,64])
    h_conv2 = tf.nn.relu(conv2d_2(h_conv1, W_conv2) + b_conv2)

    #3x3 convolution layer,  depth 64 --> depth 64
    W_conv3 = weight_variable([3,3,64,64])
    b_conv3 = bias_variable([10,10,64])
    h_conv3 = tf.nn.relu(conv2d_3(h_conv2, W_conv3) + b_conv3)

    input_dim = 10*10*64
    #Flatten the filtered images in a vector  #TODO add actions
    h_conv3_flat = tf.reshape(h_conv3, [-1, input_dim])

    keep_prob = tf.placeholder(tf.float32)
    #Fully connected layer of 10*10*64 neurons (activation function: ReLU) with dropout(prob = keep_prob)
    W_fc1 = weight_variable([input_dim,512])
    b_fc1 = bias_variable([512])
    h_fc1=tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    #Fully connected layer into #nactions labels (activation function: Linear)
    W_fc2 = weight_variable([512, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])
    q_values = tf.matmul(h_fc1_drop, W_fc2) + b_fc2 #real value

    #Loss function: cross-entropy with softmax loss; numerically stable way
    #loss = tf.losses.mean_squared_error(label,scores)
    loss_value = tf.reduce_mean(tf.squared_difference(q_values, actions_ref))
    #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = label, logits = scores))

    #Optimization algorithm: ADAM, see https://arxiv.org/abs/1412.6980
    #train_step = tf.train.AdamOptimizer(1e-3).minimize(loss_value)

    #Optimization algorithm: RMSProp, see http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    train_step = tf.train.RMSPropOptimizer(LEARNING_RATE,
                                momentum = GRADIENT_MOMENTUM,
                                epsilon = MIN_SQUARED_GRADIENT,
                                decay = SQUARED_GRADIENT_MOMENTUM).minimize(loss_value)




###############
### Session ###
###############

with tf.Session(graph = ConNet) as sess:

    ##Initialisation variables outside from episodes
    env = gym.make('internet.SlitherIO-v0')
    env.configure(remotes=1)  # automatically creates a local docker container
    tf.global_variables_initializer().run() #init Q(s,a|w) with random weights
    D = deque() #initialize replay memory D
    epsilon = INITIAL_EPSILON
    verbose = True
    debug = False;
    t = 0
    reward100 = np.zeros(100)

    ## Starting the episodes pipeline
    for i_episode in range(2000):

        ## OUT EPISODE: break condition in case a targeted bug is found
        if debug:
            break

        ## OUT EPISODE: If it's the first episode initialise the environment
        if i_episode == 0:
            observation_n = env.reset()

        ## OUT EPISODE: When the episode is initialising and no observation can be extracted
        while observation_n[0] == None:  # When Slither hasn't started
            action = idx2act(0)
            action_n = [action for ob in observation_n]
            observation_n, reward_n, done_n, info = env.step(action_n)
            env.render()

        ## OUT EPISODE: First observation extracted + initialization of episode parameters
        state_p0 = preprocess_obs_ini(observation_n)
        rr = 0
        done_n = [False]
        echo_ts = 0
        last_action_idx = 0 # memory of last action

        ## OUT EPISODE: Episode start
        while not done_n[0]:

            ## IN EPISODE: Episode initialization of vars. action_array and current_loss
            action_array = 0
            current_loss = 0

            ## IN EPISODE: Render the environment
            env.render()

            ## IN EPISODE: update timestep tracking var. and total_timestep trackin var.
            echo_ts += 1
            t+=1
            
            ## IN EPISODE: choose action with e-greedy algorithm with Q_MODEL - this changes the action
            ##TODO: Include actions inside the state_representation
            if random.random() < epsilon or OBSERVATION:
                actionidx = int(random.random()*12)
                action = idx2act(actionidx)
                action_n = [action for ob in observation_n]
                #print("------------- Random Action -------------")
            else:
                feed = {image: state_p0, keep_prob: KEEP_PROB}
                #print("------------- Not Random ----------------")
                action_array = sess.run(q_values, feed_dict = feed)
                if np.isnan(action_array[0][0]):
                    print("NAN IN THE HOLE!!!!")
                    debug = True
                    break
                actionidx = np.argmax(action_array)
                action = idx2act(actionidx)
                action_n = [action for ob in observation_n]

            ## IN EPISODE: Epsilon Decay
            if epsilon > FINAL_EPSILON:
                epsilon -= EPSILON_DECAY
            
            ## IN EPISODE: take action and sample environment
            observation_n, reward_n, done_n, info_n = env.step(action_n)
            
            ## IN EPISODE: if no observation has been taken
            if observation_n[0] == None and verbose:
                print("No observation has been made")
                continue;

            ## IN EPISODE: Check if state is terminal if so: - map final state to zeros, 
            ##                                               - punish hard 
            ##                                               - append s,a,r,s+1,terminal to the database                                                  
            if done_n[0]:# Punish hard when failing
                reward_n[0] = -50
                #print(state_p0)
                state_p1 = np.zeros_like(state_p0)
                #print('FINALLY OVER!!!')
                #print(state_p1)
                D.append((state_p0,actionidx,reward_n[0],state_p1,done_n[0]))

            ## IN EPISODE: If state is not terminal: - add r to total_reward
            ##                                       - preprocess obs to state_p1
            ##                                       - append s,a,r,s+1,terminal
            if not done_n[0]:
                rr += reward_n[0]
                state_p1 = preprocess_obs(observation_n)
                state_p1 = np.append(state_p1, state_p0[:, :, :, :3], axis=3)
                D.append((state_p0,actionidx,reward_n[0],state_p1,done_n[0]))

            ## IN EPISODE: Check that replay memory buffer does not surpass max. cap.
            if len(D) > REPLAY_MEMORY:
                D.popleft()

            ## IN EPISODE: If we have filled enough the buffer we perform replay memory
            ##             START OF MEMORY REPLAY
            if t>OBSERVATION:

                ## IN MEMORY REPLAY: - Sample memory from trajectories
                ##                   - Init: - neural net inputs (states)
                ##                           - targets (Q-values)
                minibatch = random.sample(D,BATCH)
                inputs = np.zeros((BATCH, state_p0.shape[1],state_p0.shape[2],state_p0.shape[3]))
                targets = np.zeros((BATCH,ACTIONS))

                ## IN MEMORY REPLAY: define inputs and labels and train Q-model using SGD from a minibatch
                for i in range(0,len(minibatch)):

                    ## IN TRAINING: define st, at, rt, st+1, terminal of this sample
                    state_t = minibatch[i][0]            
                    action_t = minibatch[i][1]   #This is action index
                    reward_t = minibatch[i][2]
                    state_t1 = minibatch[i][3]
                    terminal = minibatch[i][4]
                    
                    ## IN TRAINING: define the input to the Q-model
                    inputs[i] = state_t[0]
                    #print("inputs:",state_t)
                    #print("INPUTS:",inputs[i])

                    ## IN TRAINING: define feeds for the estimations of Qtable and Q'table
                    feed_state_t = {image: state_t, keep_prob: KEEP_PROB}
                    feed_state_t1 = {image: state_t1, keep_prob: KEEP_PROB}

                    ## IN TRAINING: Building the labels with Q_model we will variate just the value of the action taken
                    targets[i] = sess.run(q_values, feed_dict = feed_state_t)
                    
                    ## IN TRAINING: Building st+1 q-values with Q'_model #TODO
                    Q_sa = sess.run(q_values, feed_dict = feed_state_t1)

                    ## IN TRAINING: Build the label
                    if terminal:
                        targets[i, action_t] = reward_t
                    else:
                        targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa)
                    
                    ## IN TRAINING: Training step
                    feed_train = {image: [inputs[i]], actions_ref: [targets[i]], keep_prob: KEEP_PROB}
                    _, current_loss = sess.run([train_step, loss_value], feed_dict = feed_train)

                ## IN MEMORY REPLAY: #TODO If timesteps % TARGET_NETWORK_UP_FREQ == 0:
                ##                                          then Q'model = Qmodel

            ## IN EPISODE: if terminal stop and get out of episode       
            if done_n[0]:
                break;

            ## IN EPISODE: make state_t = state_t+1
            state_p0 = state_p1

            ## IN EPISODE: Verbose of the state of the training of DQN
            monitor = ""
            if t <= OBSERVATION:
                monitor = "observe"
            elif t > OBSERVATION and t <= OBSERVATION+EXPLORE:
                monitor = "explore"
            else:
                monitor = "train"
            if verbose and (t%30 == 0 or done_n[0]) :
                print("EPISODE:",i_episode,"/TOTAL TIMESTEP:",t,"/TIMESTEP:",echo_ts,"/STATE: ",monitor, \
                        "/EPSILON:",epsilon, "/ACTION: ",action,"/REWARD: ",rr, \
                        "/Q_MAX:", action_array,"/MEAN REWARD:",np.sum(reward100)/100 , \
                        "/Loss: ", current_loss)

            ## IN EPISODE: #TODO Print variables to a file
        
        ## OUT EPISODE: check performance through the episode
        print("Episode {} finished after {} timesteps with score {}".format(i_episode,echo_ts,rr))    
        
        ## OUT EPISODE: update last 100 episode cummulative rewards array
        reward100[i_episode%100] = rr
        
        ## OUT EPISODE: #TODO Cross-episode plots, prints to a file... 