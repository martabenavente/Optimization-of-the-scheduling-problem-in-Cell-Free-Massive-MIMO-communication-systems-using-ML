import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from itertools import product
import numpy as np
from generate_scenario import get_data
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import math
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import deque
import pandas as pd

# Paths for saving models and results
path = '/mnt/c/Users/marta/Desktop/TFG/Codes/Mod/actor.keras'
path1 = '/mnt/c/Users/marta/Desktop/TFG/Codes/Mod/critic.keras'
reward_csv_path = '/mnt/c/Users/marta/Desktop/TFG/Codes/Mod/training_rewards.csv'
reward_plot_path = '/mnt/c/Users/marta/Desktop/TFG/Codes/Mod/rewards_plot.png'

# Simulation parameters
N_max = 20
N = 1
tau_p = 4
ASD_varphi = 10 * (3.14159 / 180)

def generate_possible_actions(N_max=20, numActiveAPs=20):
    actions = list(product([0, 1], repeat=numActiveAPs))
    final_actions = [list(action) + [0] * (N_max - numActiveAPs) for action in actions]
    return final_actions

possible_actions = generate_possible_actions(N_max, numActiveAPs=10)

# Model parameters
state_dim = 3
action_dim = 20
N0 = 1e-9
num_episodes = 1000
batch_size = 32
epochs_per_episode = 10

# Reward tracking class
class RewardTracker:
    def __init__(self, window_size=100):
        self.episode_rewards = []
        self.moving_avg_rewards = []
        self.reward_window = deque(maxlen=window_size)
        self.episode_losses = {'actor': [], 'critic': []}
        
    def add_reward(self, reward, actor_loss=None, critic_loss=None):
        self.episode_rewards.append(reward)
        self.reward_window.append(reward)
        moving_avg = np.mean(self.reward_window) if len(self.reward_window) > 0 else 0
        self.moving_avg_rewards.append(moving_avg)
        if actor_loss is not None:
            self.episode_losses['actor'].append(actor_loss)
        if critic_loss is not None:
            self.episode_losses['critic'].append(critic_loss)
        
    def save_to_csv(self, filename=reward_csv_path):
        df = pd.DataFrame({
            'episode': range(len(self.episode_rewards)),
            'reward': self.episode_rewards,
            'moving_avg_reward': self.moving_avg_rewards,
            'actor_loss': self.episode_losses['actor'],
            'critic_loss': self.episode_losses['critic']
        })
        df.to_csv(filename, index=False)
        print(f"Rewards and losses saved to {filename}")
        
    def plot_results(self, filename=reward_plot_path):
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(self.episode_rewards, label='Episode Reward')
        plt.plot(self.moving_avg_rewards, label=f'Moving Avg ({len(self.reward_window)} episodes)')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Training Rewards')
        plt.legend()
        
        plt.subplot(1, 3, 2)
        plt.plot(self.episode_losses['actor'])
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.title('Actor Loss')
        
        plt.subplot(1, 3, 3)
        plt.plot(self.episode_losses['critic'])
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.title('Critic Loss')
        
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"Training plots saved to {filename}")

# Initialize reward tracker
reward_tracker = RewardTracker()

# Actor network
def build_actor(state_dim, action_dim):
    model = keras.Sequential([
        layers.Input(shape=(state_dim * action_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(action_dim, activation='sigmoid')
    ])
    return model

# Critic network
def build_critic(state_dim, action_dim):
    model = keras.Sequential([
        layers.Input(shape=(state_dim * action_dim + action_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    return model

# Create and compile networks
actor = build_actor(state_dim, action_dim)
critic = build_critic(state_dim, action_dim)
initial_actor_weights = actor.get_weights()

actor.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01))
critic.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001))

# Model saving/loading
def save_model(model, path):
    model.save(path)
    print(f"Model saved to {path}")

def load_model(path):
    if os.path.exists(path):
        return keras.models.load_model(path)
    raise FileNotFoundError(f"Model not found at {path}")

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
    
    def add(self, state, action, reward):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def size(self):
        return len(self.buffer)

buffer_capacity = 5000
replay_buffer = ReplayBuffer(buffer_capacity)

# Reward calculation
@tf.function
def calculate_sum_rate(state, action, max_gain=1):
    reshaped_state = tf.reshape(state, [-1, state_dim, action_dim])
    action = tf.cast(action, tf.float32)
    reshaped_state = tf.cast(reshaped_state, tf.float32)
    max_gain = tf.cast(max_gain, tf.float32)

    ap_gain = action * reshaped_state[:, 1, :]
    ap_interference = action * reshaped_state[:, 2, :]

    ap_gain *= max_gain
    ap_interference *= max_gain

    sum_rate_per_ap = tf.math.log1p(ap_gain / (ap_interference + N0)) / tf.math.log(2.0)
    return tf.reduce_sum(sum_rate_per_ap)

# Main training loop
if __name__ == '__main__':
    previous_actions = None
    
    for episode in tqdm(range(200), desc="Training Actor-Critic"):
        actor.set_weights(initial_actor_weights)
        K = random.randint(5, 45)
        L = random.randint(35, 75) if K < 31 else random.randint(K+1, 75)
        numActiveAPs = random.choices([5, 10, 15, 20], weights=(5, 10, 15, 20))[0]
        
        active_APs, df, gainOverNoisedB, powgain, max_gain = get_data(
            L=L, K=K, N=N, tau_p=tau_p, ASD_varphi=ASD_varphi, numActiveAPs=numActiveAPs, grid=False
        )

        agg_df = df.groupby('UE_id').agg({
            'AP_id': lambda x: list(x),
            'user_gain': lambda x: list(x),
            'interference_sum': lambda x: list(x)
        }).reset_index()

        agg_df['state'] = agg_df.apply(lambda x: np.concatenate((
            np.array(x['AP_id'], dtype=int), 
            x['user_gain'], 
            x['interference_sum']
        )), axis=1)

        episode_reward = 0
        episode_actor_loss = 0
        episode_critic_loss = 0
        update_count = 0

        for state in agg_df['state']:
            for i in range(numActiveAPs):
                action = actor.predict(np.array(state).reshape(1, -1), verbose=0)
                binary_action = (action > 0.01).astype(int)
                binary_action[:, numActiveAPs:] = 0

                reward = calculate_sum_rate(np.array(state).reshape(1, -1), binary_action)
                num_selected_aps = tf.reduce_sum(action)
                scaling_factor = num_selected_aps / tf.cast(numActiveAPs, tf.float32)
                penalty = tf.where(num_selected_aps < tf.cast(numActiveAPs * 0.5, tf.float32), -5.0, 0.0)
                reward += scaling_factor + penalty
                if previous_actions is not None and np.array_equal(previous_actions, action):
                    reward -= 1

                replay_buffer.add(state, binary_action, reward)
                episode_reward += reward.numpy()

            if replay_buffer.size() >= batch_size:
                batch = replay_buffer.sample(batch_size)
                states, actions, rewards = zip(*batch)
                states = np.array(states)

                # Critic update
                with tf.GradientTape() as tape:
                    actions_reshaped = np.array(actions).reshape(batch_size, -1)
                    q_values = critic(tf.concat([states, actions_reshaped], axis=1))
                    critic_loss = tf.reduce_mean(tf.square(np.array(rewards).reshape(-1, 1) - q_values))
                critic_grads = tape.gradient(critic_loss, critic.trainable_variables)
                critic.optimizer.apply_gradients(zip(critic_grads, critic.trainable_variables))

                # Actor update
                with tf.GradientTape() as tape:
                    predicted_actions = actor(states)
                    actor_loss = -tf.reduce_mean(critic(tf.concat([states, predicted_actions], axis=1)))
                    entropy = -tf.reduce_sum(predicted_actions * tf.math.log(predicted_actions + 1e-10))
                    actor_loss += 0.05 * entropy

                actor_grads = tape.gradient(actor_loss, actor.trainable_variables)
                actor.optimizer.apply_gradients(zip(actor_grads, actor.trainable_variables))

                episode_actor_loss += actor_loss.numpy()
                episode_critic_loss += critic_loss.numpy()
                update_count += 1

        # Calculate average losses
        avg_actor_loss = episode_actor_loss / update_count if update_count > 0 else 0
        avg_critic_loss = episode_critic_loss / update_count if update_count > 0 else 0

        # Track rewards and losses
        reward_tracker.add_reward(
            episode_reward,
            actor_loss=avg_actor_loss,
            critic_loss=avg_critic_loss
        )

        # Print progress
        print(f"\nEpisode {episode}:")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Moving Avg Reward: {reward_tracker.moving_avg_rewards[-1]:.2f}")
        print(f"  Actor Loss: {avg_actor_loss:.4f}")
        print(f"  Critic Loss: {avg_critic_loss:.4f}")

    # Save results and models
    reward_tracker.save_to_csv()
    reward_tracker.plot_results()
    save_model(actor, path)
    save_model(critic, path1)

    print("\nTraining complete")
