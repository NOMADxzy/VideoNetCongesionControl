import gym
import os
import time

import network_sim
from DDPG import *
from rl import *
from dt import  *
from log import *
from pong import *


def learn_dt():
    
    # Parameters
    log_fname = '../pong_dt.log'
    model_path = '../data/model-CartPole-1/saved'
    max_depth = 10
    n_batch_rollouts = 10#100
    max_samples = 20000#200000
    max_iters = 80#80
    train_frac = 0.8
    is_reweight = True
    n_test_rollouts = 50#50
    save_dirname = '../tmp/PccNs'
    save_fname = 'dt_policy.pk'
    save_viz_fname = 'd13_2.dot'
    is_train = True
    
    # Logging
    set_file(log_fname)
    
    # Data structures
    #env = get_pong_env()
    #env = gym.make('Pendulum-v0')
    env = gym.make('PccNs-v0')
    
    ######Train teacher first
    MAX_EPISODES = 5#200
    MAX_EP_STEPS = 400#200
    LR_A = 0.001    # learning rate for actor
    LR_C = 0.002    # learning rate for critic
    GAMMA = 0.9     # reward discount
    TAU = 0.01      # soft replacement
    MEMORY_CAPACITY = 10000
    BATCH_SIZE = 32
    RENDER = False
    env = env.unwrapped
    env.seed(1)

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high

    teacher = DQNPolicy(a_dim, s_dim, a_bound)

    var = 3  # control exploration
    t1 = time.time()
    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_reward = 0
        a_ep = []
        for j in range(MAX_EP_STEPS):
            if RENDER:
                env.render()

            # Add exploration noise
            a = teacher.choose_action(s)
            print(a)
            #a = np.clip(np.random.normal(a, var), -3, 3)    # add randomness to action selection for exploration
            a_ep.append(a[0])
            s_, r, done, info = env.step(a)
            s_temp = np.zeros(s_.size)
            for k in range(s_.size):
                if type(s_[k]) == type(s_):
                    s_temp[k] = s_[k][0]
                else:
                    s_temp[k] = s_[k]
            s_ = s_temp
                
            '''
            print('s')
            print(s)
            print(s.size)
            print('a')
            print(a)
            print('r')
            print(r)
            print('s_')
            print(s_)
            print(s_.size)
            '''
            teacher.store_transition(s, a, r / 10, s_)

            if teacher.pointer > MEMORY_CAPACITY:
                var *= .9995    # decay the action randomness
                teacher.learn()

            s = s_
            ep_reward += r
            if j == MAX_EP_STEPS-1:
                #print('Episode:', i, ' Reward: %i ' % int(ep_reward/MAX_EP_STEPS), 'Explore: %.2f' % var, )
                print('Episode:', i, ' Reward: %i ' % ep_reward, 'Explore: %.2f' % var, )
                 # if ep_reward > -300:RENDER = True
                break
        #print(a_ep)

    ##############


    student = DTPolicy(max_depth)
    state_transformer = get_pong_symbolic




    # Train student
    if is_train:
        student = train_dagger(env, teacher, student, state_transformer, max_iters, n_batch_rollouts, max_samples, train_frac, is_reweight, n_test_rollouts)
        save_dt_policy(student, save_dirname, save_fname)
        save_dt_policy_viz(student, save_dirname, save_viz_fname)
    else:
        student = load_dt_policy(save_dirname, save_fname)

    # Test student
    rew = test_policy(env, student, state_transformer, n_test_rollouts)
    log('Final reward: {}'.format(rew), INFO)
    log('Number of nodes: {}'.format(student.tree.tree_.node_count), INFO)

def bin_acts():
    # Parameters
    seq_len = 10
    n_rollouts = 10
    log_fname = 'pong_options.log'
    model_path = 'model-atari-pong-1/saved'
    
    # Logging
    set_file(log_fname)
    
    # Data structures
    env = get_pong_env()
    teacher = DQNPolicy(env, model_path)

    # Action sequences
    seqs = get_action_sequences(env, teacher, seq_len, n_rollouts)

    for seq, count in seqs:
        log('{}: {}'.format(seq, count), INFO)

def print_size():
    # Parameters
    dirname = 'results/run9'
    fname = 'dt_policy.pk'

    # Load decision tree
    dt = load_dt_policy(dirname, fname)

    # Size
    print(dt.tree.tree_.node_count)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    learn_dt()

