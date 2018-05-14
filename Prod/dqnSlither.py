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
import os
import csv
import pickle

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
GAMMA = 0.99 # decay rate of past observations
TARGET_NETWORK_UP_FREQ = 10000 # frequency with which the target network is updated. Parameter C. #test 100
k = 4 # freq of taking actions
UPDATE_FREQ = 4 # update of the network
LEARNING_RATE = 0.00025 
GRADIENT_MOMENTUM = 0.95 
SQUARED_GRADIENT_MOMENTUM = 0.95 
MIN_SQUARED_GRADIENT = 0.01 
INITIAL_EPSILON = 1.0 # starting value of epsilon
EXPLORE = 1000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.10 # final value of epsilon

EPSILON_DECAY = (INITIAL_EPSILON - FINAL_EPSILON)/EXPLORE
OBSERVATION = 50000. # timesteps to observe before training #TOREPAIR --> REPAIRED
MODEL_SAVER = 250000 #TOREPAIR --> REPAIRED

ACTIONS = 12 # number of valid actions
KEEP_PROB = 1.0#0.9 as we train very little


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

def find_last_model():
    model_list = next(os.walk("./train"))[1]
    model_list = [int(element[9:]) for element in model_list]
    model_list.sort()
    return model_list[-1]

#############
### Graph ###
#############

ConNet = tf.Graph()

with ConNet.as_default():

    image = tf.placeholder(tf.float32,shape = (None, 80,80,4))
    x_image = image
    actions_ref = tf.placeholder(tf.float32, shape = (None,ACTIONS))
    keep_prob = tf.placeholder(tf.float32)

    ## 1st layer
    # Q_model #8x8 convolution layer, depth 1 --> depth 32
    W_conv1 = weight_variable([8,8,4,32])
    b_conv1 = bias_variable([20,20,32])
    h_conv1 = tf.nn.relu(conv2d_1(x_image,W_conv1) + b_conv1)

    # Q'_model #8x8 convolution layer, depth 1 --> depth 32
    Wp_conv1 = weight_variable([8,8,4,32])
    bp_conv1 = bias_variable([20,20,32])
    hp_conv1 = tf.nn.relu(conv2d_1(x_image,Wp_conv1) + bp_conv1)

    ## 2nd layer
    # Q_model #4x4 convolution layer, depth 32 --> depth 64
    W_conv2 = weight_variable([4,4,32,64])
    b_conv2 = bias_variable([10,10,64])
    h_conv2 = tf.nn.relu(conv2d_2(h_conv1, W_conv2) + b_conv2)

    # Q'_model #4x4 convolution layer, depth 32 --> depth 64
    Wp_conv2 = weight_variable([4,4,32,64])
    bp_conv2 = bias_variable([10,10,64])
    hp_conv2 = tf.nn.relu(conv2d_2(hp_conv1, Wp_conv2) + bp_conv2)

    ## 3rd layer
    # Q_model #3x3 convolution layer,  depth 64 --> depth 64
    W_conv3 = weight_variable([3,3,64,64])
    b_conv3 = bias_variable([10,10,64])
    h_conv3 = tf.nn.relu(conv2d_3(h_conv2, W_conv3) + b_conv3)

    # Q'_model #3x3 convolution layer,  depth 64 --> depth 64
    Wp_conv3 = weight_variable([3,3,64,64])
    bp_conv3 = bias_variable([10,10,64])
    hp_conv3 = tf.nn.relu(conv2d_3(hp_conv2, Wp_conv3) + bp_conv3)
    
    input_dim = 10*10*64
    input_dimp4 = input_dim + 12*4

    ## Flatten resulting images
    # Q_model #Flatten the filtered images in a vector - added actions
    h_conv3_flat = tf.reshape(h_conv3, [-1, input_dim])

    # Q'_model #Flatten the filtered images in a vector - added actions
    hp_conv3_flat = tf.reshape(hp_conv3, [-1, input_dim])

    #Concatenating the action history: each one encoded as a one-hot vector
    # in order to keep action history in check. Easy to comment if we prefer to omit
    actionhist = tf.placeholder(tf.float32, shape = (None,4,ACTIONS)) ## add placeholder with previous actions vector as one-hot vectors
    actionhist2 = tf.reshape(actionhist,[-1,4*ACTIONS])
    h_conv3_flat = tf.concat([h_conv3_flat , actionhist2],1)
    hp_conv3_flat = tf.concat([hp_conv3_flat , actionhist2],1)

    ## 4th layer
    # Q_model #Fully connected layer of 10*10*64 neurons (activation function: ReLU) with dropout(prob = keep_prob)
    W_fc1 = weight_variable([input_dimp4,512])
    b_fc1 = bias_variable([512])
    h_fc1=tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Q'_model #Fully connected layer of 10*10*64 neurons (activation function: ReLU) with dropout(prob = keep_prob)
    Wp_fc1 = weight_variable([input_dimp4,512])
    bp_fc1 = bias_variable([512])
    hp_fc1=tf.nn.relu(tf.matmul(hp_conv3_flat, Wp_fc1) + bp_fc1)
    hp_fc1_drop = tf.nn.dropout(hp_fc1, keep_prob)

    ## 5th layer
    # Q_model #Fully connected layer into #nactions labels (activation function: Linear)
    W_fc2 = weight_variable([512, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])
    q_values = tf.matmul(h_fc1_drop, W_fc2) + b_fc2 #real value

    # Q'_model #Fully connected layer into #nactions labels (activation function: Linear)
    Wp_fc2 = weight_variable([512, ACTIONS])
    bp_fc2 = bias_variable([ACTIONS])
    qp_values = tf.matmul(hp_fc1_drop, Wp_fc2) + bp_fc2 #real value


    # Just Q_model #Loss function: MSE
    loss_value = tf.reduce_mean(tf.squared_difference(q_values, actions_ref))
    
    #Optimization algorithm: ADAM, see https://arxiv.org/abs/1412.6980
    #train_step = tf.train.AdamOptimizer(1e-3).minimize(loss_value)

    # Just Q_model #Optimization algorithm: RMSProp, see http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    train_step = tf.train.RMSPropOptimizer(LEARNING_RATE,
                                momentum = GRADIENT_MOMENTUM,
                                epsilon = MIN_SQUARED_GRADIENT,
                                decay = SQUARED_GRADIENT_MOMENTUM).minimize(loss_value)

    saver = tf.train.Saver(max_to_keep = None)


####### Session Parameters #######
# Now it is automatatised ###loading = False #First training session --> False #Otherwise --> True
train = True # Training --> True  # Test --> False
print_to_csv = True
###############
### Session ###
###############

with tf.Session(graph = ConNet) as sess:
    first_step_sess = True
    ## Weight and csv file writers initialisation
    if next(os.walk("./train"))[1] == []:
        timesteps_csvfile = open('train/timestep_data.csv','a')
        episodes_csvfile = open('train/episode_data.csv','a')
        listwriter_ts = csv.writer(timesteps_csvfile,delimiter = ',')
        listwriter_ep = csv.writer(episodes_csvfile,delimiter = ',')
        header_ts = "EPISODE|TOTAL FRAMES|FRAME|STATE|EPSILON|ACTION|REWARD|Q_MAX|MEAN REWARD|Last Batch Loss Mean"
        header_ep = "#Episode|#Frame|#Epsode Timesteps|Total episode score|Last Mean Batch Loss"
        listwriter_ts.writerow(header_ts.split('|'))
        listwriter_ep.writerow(header_ep.split('|'))

        ## Initialise weights randomly, replay memory (empty) and epsilon
        tf.global_variables_initializer().run() #init Q(s,a|w) with random weights
        D = deque() #initialize replay memory D
        epsilon = INITIAL_EPSILON
        t = 0
        starting_episode = 0

    else:
        timesteps_csvfile = open('train/timestep_data.csv','a')
        episodes_csvfile = open('train/episode_data.csv','a')
        listwriter_ts = csv.writer(timesteps_csvfile,delimiter = ',')
        listwriter_ep = csv.writer(episodes_csvfile,delimiter = ',')
        
        ## Load weights and replay memory (from last session)
        lastmodel = find_last_model()
        saver.restore(sess, "train/model_ts_"+str(lastmodel)+"/model.ckpt")
        print("Weights from the "+str(lastmodel)+"th model LOADED!")
        fload = open('train/memory_replay.pkl','rb')
        D,epsilon,starting_episode = pickle.load(fload)
        fload.close()
        t = lastmodel


    ##Initialisation variables outside from episodes
    env = gym.make('internet.SlitherIO-v0')
    env.configure(remotes=1)  # automatically creates a local docker container
    verbose = True
    debug = False;
    reward100 = np.zeros(100)


    ## Initialisation of target network (q'model) as a copy of the original one.
    ## Only if t == 0, as otherwise the weights are already initialised as we left
    ## them in the previous training session.
    if t == 0:
        print('UPDATING TARGET MODEL')
        sess.run(tf.assign(Wp_conv1,W_conv1)) #1st layer
        sess.run(tf.assign(bp_conv1,b_conv1))

        sess.run(tf.assign(Wp_conv2,W_conv2)) #2nd layer
        sess.run(tf.assign(bp_conv2,b_conv2))

        sess.run(tf.assign(Wp_conv3,W_conv3)) #3rd layer
        sess.run(tf.assign(bp_conv3,b_conv3))

        sess.run(tf.assign(Wp_fc1,W_fc1)) #4th layer
        sess.run(tf.assign(bp_fc1,b_fc1))

        sess.run(tf.assign(Wp_fc2,W_fc2)) #5th layer
        sess.run(tf.assign(bp_fc2,b_fc2))




    ## Starting the episodes pipeline
    for i_episode in range(starting_episode,2000):
        if t>10000000:
            break
        ## OUT EPISODE: break condition in case a targeted bug is found
        if debug:
            break

        ## OUT EPISODE: If it's the first episode initialise the environment
        if first_step_sess == True:
            observation_n = env.reset()

        ## OUT EPISODE: When the episode is initialising and no observation can be extracted
        while observation_n[0] == None:  # When Slither hasn't started
            actionidx = int(random.random()*ACTIONS)
            action = idx2act(actionidx)
            action_n = [action for ob in observation_n]
            observation_n, reward_n, done_n, info = env.step(action_n)
            env.render()

        ## OUT EPISODE: First state extracted + initialization of episode parameters
        state_p0 = preprocess_obs_ini(observation_n)
        acthist = np.zeros(ACTIONS); acthist[0] = 1;
        acthist = np.array([acthist,acthist,acthist,acthist])

        rr = 0
        done_n = [False]
        echo_ts = 0


        ## OUT EPISODE: Episode initialization of vars. action_array and current_loss
        action_array = -1
        current_loss = -1
        mean_batch_loss = -1

        ## OUT EPISODE: Episode start
        while not done_n[0]:

            ## IN EPISODE: Render the environment
            env.render()

            ## IN EPISODE: update timestep tracking var. and total_timestep trackin var.
            echo_ts += 1
            t+=1
            
            ## IN EPISODE:  If total_timesteps%k == 0: update action
            ##              If #total_timesteps < OBSERVATION --> random action 
            ##              Otherwise: choose action with e-greedy algorithm with Q_MODEL - this changes the action
            ## Included actions inside the state_representation
            if t%k == 0:
                if random.random() < epsilon or t<OBSERVATION:
                    actionidx = int(random.random()*ACTIONS)
                    action = idx2act(actionidx)
                    action_n = [action for ob in observation_n]
                    #print("------------- Random Action -------------")
                else:
                    feed = {image: state_p0, keep_prob: KEEP_PROB, actionhist: [acthist]}
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
            ##                                               - append s,a,r,s+1,terminal,action_history to the database                                                  
            if done_n[0]:# Punish hard when failing
                reward_n[0] = -50
                #print(state_p0)
                state_p1 = np.zeros_like(state_p0)
                aux = np.zeros(ACTIONS); aux[actionidx] = 1;
                acthist = np.append([aux],acthist[:3,:],axis = 0)
                #print('FINALLY OVER!!!')
                #print(state_p1)
                D.append((state_p0,actionidx,reward_n[0],state_p1,done_n[0],acthist))

            ## IN EPISODE: If state is not terminal: - add r to total_reward
            ##                                       - preprocess obs to state_p1
            ##                                       - append s,a,r,s+1,terminal,action history
            if not done_n[0]:
                rr += reward_n[0]
                state_p1 = preprocess_obs(observation_n)
                state_p1 = np.append(state_p1, state_p0[:, :, :, :3], axis=3)
                aux = np.zeros(ACTIONS); aux[actionidx] = 1;
                acthist = np.append([aux],acthist[:3,:],axis = 0)
                D.append((state_p0,actionidx,reward_n[0],state_p1,done_n[0],acthist))

            ## IN EPISODE: Check that replay memory buffer does not surpass max. cap.
            if len(D) > REPLAY_MEMORY:
                D.popleft()

            ## IN EPISODE: If we have filled enough the buffer we perform replay memory
            ##             START OF MEMORY REPLAY
            if t>OBSERVATION and t%UPDATE_FREQ == 0 and train:

                ## IN MEMORY REPLAY: - Sample memory from trajectories
                ##                   - Init: - neural net inputs (states)
                ##                           - targets (Q-values)
                minibatch = random.sample(D, BATCH)
                inputs = np.zeros((BATCH, state_p0.shape[1],state_p0.shape[2],state_p0.shape[3]))
                inpActHist = np.zeros((BATCH,4,ACTIONS))
                targets = np.zeros((BATCH,ACTIONS))

                ## IN MEMORY REPLAY: define inputs and labels and train Q-model using SGD from a minibatch
                mean_batch_loss = 0
                for i in range(0,len(minibatch)):

                    ## IN TRAINING: define st, at, rt, st+1, terminal of this sample
                    state_t = minibatch[i][0]            
                    action_t = minibatch[i][1]   #This is action index
                    reward_t = minibatch[i][2]
                    state_t1 = minibatch[i][3]
                    terminal = minibatch[i][4]
                    action_history = minibatch[i][5]
                    #print(action_history)
                    
                    ## IN TRAINING: define the input to the Q-model
                    inputs[i] = state_t[0]
                    inpActHist[i] = action_history

                    ## IN TRAINING: define feeds for the estimations of Qtable and Q'table
                    feed_state_t = {image: state_t, keep_prob: KEEP_PROB, actionhist: [action_history]}
                    feed_state_t1 = {image: state_t1, keep_prob: KEEP_PROB, actionhist: [action_history]}

                    ## IN TRAINING: Building the labels with Q_model we will variate just the value of the action taken
                    targets[i] = sess.run(q_values, feed_dict = feed_state_t)
                    
                    ## IN TRAINING: Building st+1 q-values with Q'_model
                    Q_sa = sess.run(qp_values, feed_dict = feed_state_t1)

                    ## IN TRAINING: Build the label
                    if terminal:
                        targets[i, action_t] = reward_t
                    else:
                        targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa)
                    
                    ## IN TRAINING: Training step
                    feed_train = {image: [inputs[i]], actions_ref: [targets[i]], keep_prob: KEEP_PROB, actionhist: [action_history]}
                    _, current_loss = sess.run([train_step, loss_value], feed_dict = feed_train)
                    mean_batch_loss += current_loss
                mean_batch_loss /= len(minibatch)
            ## IN MEMORY REPLAY: If timesteps % TARGET_NETWORK_UP_FREQ == 0:
            ##                                          then Q'model = Qmodel
            if t%TARGET_NETWORK_UP_FREQ == 0 and t > OBSERVATION and train:
                print('UPDATING TARGET MODEL')
                sess.run(tf.assign(Wp_conv1,W_conv1)) #1st layer 
                sess.run(tf.assign(bp_conv1,b_conv1))

                sess.run(tf.assign(Wp_conv2,W_conv2)) #2nd layer
                sess.run(tf.assign(bp_conv2,b_conv2))

                sess.run(tf.assign(Wp_conv3,W_conv3)) #3rd layer
                sess.run(tf.assign(bp_conv3,b_conv3))

                sess.run(tf.assign(Wp_fc1,W_fc1)) #4th layer
                sess.run(tf.assign(bp_fc1,b_fc1))

                sess.run(tf.assign(Wp_fc2,W_fc2)) #5th layer
                sess.run(tf.assign(bp_fc2,b_fc2))
                

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
                        "/EPSILON:",epsilon, "/ACTION: ",actionidx,"/REWARD: ",rr, \
                        "/Q_MAX:", action_array,"/MEAN REWARD:",np.sum(reward100)/100 , \
                        "/Last Mean Batch Loss: ", mean_batch_loss)
                
                ## IN EPISODE: Print variables to a CSV file
                if print_to_csv:
                    table = [i_episode,t,echo_ts,monitor,epsilon,
                            actionidx,rr,action_array,np.sum(reward100)/100,mean_batch_loss]
                    listwriter_ts.writerow(table)

            ## IN EPISODE: save the model's weights
            ##              save the model's memory replay
            if t % MODEL_SAVER == 0 and t>OBSERVATION:
                if not os.path.exists("train/model_ts_"+str(t)):
                    os.makedirs("train/model_ts_"+str(t))
                save_path = saver.save(sess, "train/model_ts_"+str(t)+"/model.ckpt")
                fsave = open('train/memory_replay.pkl','wb')
                pickle.dump([D,epsilon,i_episode+1],fsave)
                fsave.close()
                print("Model saved in path: %s, and memory replay saved" % save_path)
        
        ## OUT EPISODE: check performance through the episode and print to a CSV file
        print("Episode {} finished at frame {}, after {} timesteps with score {} and last mean batch loss{}".format(i_episode,t,echo_ts,rr,mean_batch_loss))    
        if print_to_csv:
            table = [i_episode,t,echo_ts,rr,mean_batch_loss]
            listwriter_ep.writerow(table)
        ## OUT EPISODE: update last 100 episode cummulative rewards array
        reward100[i_episode%100] = rr
        
         


