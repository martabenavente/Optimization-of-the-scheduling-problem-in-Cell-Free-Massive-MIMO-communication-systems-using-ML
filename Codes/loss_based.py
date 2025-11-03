import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
from generate_scenario import get_data
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import random
import matplotlib.pyplot as plt

# Paths and parameters
path = '/mnt/c/Users/marta/Desktop/TFG/Codes/Mod/loss_neg.keras'
path1 = '/mnt/c/Users/marta/Desktop/TFG/Codes/Mod/loss_inv.keras'
N_max = 20
N = 1
tau_p = 4
ASD_varphi = 10 * (3.14159 / 180)
epochs_per_episode = 10
num_validation_scenarios = 20

# Model definition
def build_model():
    inputs = layers.Input(shape=(20, 2))
    x = layers.Flatten()(inputs)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(N_max, activation='sigmoid')(x)
    return Model(inputs=inputs, outputs=outputs)

model = build_model()
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001))

def save_model(model, path):
    model.save(path)
    print(f"Modelo guardado en {path}")

# Data preparation
def prepare_data(df):
    return np.array([np.array([row['user_gain'], row['interference_sum']]).T for _, row in df.iterrows()])

# Training step
@tf.function
def train_step(model, x, oscar):
    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)
        gain = tf.cast(x[:, :, 0], tf.float32)
        interference = tf.cast(x[:, :, 1], tf.float32)
        selected_gain = tf.reduce_sum(y_pred * gain, axis=1)
        selected_interference = tf.reduce_sum(y_pred * interference, axis=1)
        noise = 1e-6
        sum_rates = tf.math.log(1 + (selected_gain / (selected_interference + noise))) / tf.math.log(2.0)
        num_selected_aps = tf.reduce_sum(y_pred)
        scaling_factor = num_selected_aps / tf.cast(numActiveAPs, tf.float32)
        sum_rates += 10 * scaling_factor
        
        if oscar:
            loss = tf.reduce_sum(1 / (sum_rates + noise))
        else:
            loss = tf.reduce_sum(-sum_rates)
        
        penalty = tf.nn.relu(tf.cast(numActiveAPs, tf.float32) * 0.5 - tf.reduce_sum(y_pred))
        loss += 0.5 * penalty
    
    gradients = tape.gradient(loss, model.trainable_variables)
    if all(g is not None for g in gradients):
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Validation utilities
def generate_validation_scenarios(num_scenarios=20, seed=42):
    random.seed(seed)
    return [
        (
            random.randint(35, 75) if (K := random.randint(5, 45)) < 31 else random.randint(K+1, 75),
            K,
            random.choice([5, 10, 15, 20])
        ) 
        for _ in range(num_scenarios)
    ]

def evaluate_validation(model, scenarios):
    total_loss = 0
    for L, K, numActiveAPs in scenarios:
        _, df, _, _, _ = get_data(L=L, K=K, N=N, tau_p=tau_p, 
                                 ASD_varphi=ASD_varphi, numActiveAPs=numActiveAPs, grid=False)
        X_val = prepare_data(df.groupby('UE_id').agg({'AP_id': list, 'user_gain': list, 'interference_sum': list}))
        total_loss += np.mean([train_step(model, ue.reshape(1, 20, 2), oscar=True).numpy() for ue in X_val])
    return total_loss / len(scenarios)

# Main training loop
if __name__ == '__main__':
    # Initialize logging
    val_scenarios = generate_validation_scenarios(num_validation_scenarios)
    train_log = []
    val_log = []

    numActiveAPs = random.choices([5, 10, 15, 20], weights=(5, 10, 15, 20))[0]
    initial_val_loss = evaluate_validation(model, val_scenarios)
    val_log.append((-1, initial_val_loss))
    print(f"Pre-training validation loss: {initial_val_loss:.4f}")
    
    for episode in range(100):
        # Generate training scenario
        K = random.randint(5, 45)
        L = random.randint(35, 75) if K < 31 else random.randint(K+1, 75)
        numActiveAPs = random.choices([5, 10, 15, 20], weights=(5, 10, 15, 20))[0]
        
        # Load and prepare data
        _, df, _, _, _ = get_data(L=20, K=2, N=N, tau_p=tau_p, 
                                ASD_varphi=ASD_varphi, numActiveAPs=numActiveAPs, grid=False)
        X_train = prepare_data(df.groupby('UE_id').agg({'AP_id': list, 'user_gain': list, 'interference_sum': list}))

        # Training epoch
        epoch_loss = np.mean([
            train_step(model, ue.reshape(1, 20, 2), oscar=True).numpy() 
            for ue in X_train
        ])
        train_log.append(epoch_loss)

        # Validation every 5 episodes
        if episode % 5 == 0:
            val_loss = evaluate_validation(model, val_scenarios)
            val_log.append((episode, val_loss))
            print(f"Episode {episode}: Train Loss = {epoch_loss:.4f}, Val Loss = {val_loss:.4f}")

    # Save validation results
    val_df = pd.DataFrame(val_log, columns=['epoch', 'val_loss'])
    val_df.to_csv('validation_history_inv.csv', index=False)

    # Plot validation only
    plt.figure(figsize=(10, 5))
    plt.plot(val_df['epoch'], val_df['val_loss'], 'r-', label='Validation Loss')
    plt.xlabel('Training Episode')
    plt.ylabel('Loss')
    plt.title('Validation Loss Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('validation_convergence1.png', bbox_inches='tight')
    plt.close()

    # Save model
    save_model(model, path)