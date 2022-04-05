import numpy as np
import random
from collections import deque
import math
import numpy as np
import matplotlib.pyplot as plt

class Memory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size) # Some kind of "lists" optimized for fast fixed-length operations
    
    def push(self, state, action, new_state, reward, done):
        """Fill the buffer with status from enviroment"""
        
        experience = (state, action, new_state, np.array([reward]), done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        """Random samples with 'batch_size' elements"""
        state_batch = []
        action_batch = []
        new_state_batch = []
        reward_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, new_state, reward, done = experience
            state_batch.append(state)
            action_batch.append(action)
            new_state_batch.append(new_state)
            reward_batch.append(reward)
            done_batch.append(done)
        
        return state_batch, action_batch, new_state_batch, reward_batch, done_batch
    
    def reset(self):
        # Clear the buffer
        self.buffer.clear()
    
    def __len__(self):
        return len(self.buffer)
    
def calculate_epsilon(steps_done, egreedy_decay):
    """
    Probability to choose random action. It decrease exponentially from 90% chance to takeaction
    to 10% chance to take random action
    
    """
    egreedy_initial = 0.9
    egreedy_final = 0.1
    
    epsilon = egreedy_final + (egreedy_initial - egreedy_final) * \
              math.exp(-1. * steps_done / egreedy_decay )
    return epsilon
    

def fill_memory(env, number_of_elements, agent):
    """Fill the Memory with some random elements = 'number_of_elements' """
    time_steps = 0
    state = env.reset()
    done = False
    
    while time_steps < number_of_elements:
        action = env.action_space.sample()
        new_state, reward, done, info = env.step(action)
        
        agent.memory.push(state, action, new_state, reward, done)
        state = new_state 
        time_steps += 1
        if done:
            state = env.reset()
            done = False
    print("Added %i elements in Memory" % number_of_elements)

def evaluate_agent(agent, env, eval_episodes, render = False):
    
    """Evaluate performance of a policy after training"""
    avg_reward = 0
    for i in range(eval_episodes):
        obs = env.reset()
        done = False
        print ("Current episode: ", i+1)
        while not done:
            if render:
                env.render()
            action = agent.select_action(obs, -1, env)
            obs, reward, done, info = env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    
    print("\n---------------------------------------")
    print("Evaluation over {:d} episodes: {:f}" .format(eval_episodes, avg_reward))
    print("---------------------------------------")
    env.close()

def train(agent, env, num_episodes, batch_size, egreedy_decay, file2save, directory):
    """Train agents"""
    rewards = []
    avg_rewards = []
    best_avg = -5000
    avg_reward = 0
    steps_done = 0
    best_avg_reward = -5000
    for i_episode in range (num_episodes):
        
        episode_reward = 0
        state = env.reset()
        
        while True:
            steps_done += 1
            epsilon = calculate_epsilon(steps_done, egreedy_decay)
            action = agent.select_action(state, epsilon, env)
            new_state, reward, done, info = env.step(action)
            agent.memory.push(state, action, new_state, reward, done)
            agent.update(batch_size)
            state = new_state
            
            avg_reward = np.mean(rewards[-10:])
            episode_reward += reward
            if done:
                print ("Episode: ", i_episode+1, " Reward: ", np.round(episode_reward, decimals=2), \
                       " Average reward: ", np.round(avg_reward,decimals=2))
                    
                if best_avg <= avg_reward: # Save when we have best average reward
                    best_avg = avg_reward
                    agent.save(file2save, directory)
                break
   
        rewards.append(episode_reward)   
        avg_rewards.append(avg_reward)            
   
        if best_avg_reward < avg_reward:
            best_avg_reward = avg_reward
    print("\n\n\nBest average reward in {:d}, episodes is: {:f}".format((i_episode+1), np.round(best_avg_reward,decimals=2)))
    plt.plot(rewards, label = "Rewards")
    plt.plot(avg_rewards, label = "Avg Rewards")
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Average Reward')
    plt.legend(loc="upper left")
    plt.show()
    
    return avg_rewards
                
            
        
    
    
