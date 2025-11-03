import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
from generate_scenario import get_data
from tqdm import tqdm
from modelo_1 import load_model, calculate_sum_rate, generate_possible_actions
from fuzzy_clustering import fuzzy_clustering_aps
from loss_based import prepare_data
from memory_profiler import memory_usage
import time
import gc
import random
import tensorflow as tf

# Configure TensorFlow to reduce retracing warnings
tf.config.optimizer.set_jit(True)  # Enable XLA
tf.config.experimental.enable_op_determinism()

# Parameters
N_max = 20
N = 1
tau_p = 4
ASD_varphi = 10 * (3.14159 / 180)
N0 = 1e-9

# Model paths
path = '/mnt/c/Users/marta/Desktop/TFG/Codes/Mod/actor.keras'
path1 = '/mnt/c/Users/marta/Desktop/TFG/Codes/Mod/loss_neg.keras'
path2 = '/mnt/c/Users/marta/Desktop/TFG/Codes/Mod/loss_inv.keras'

def evaluate_configuration(num_episodes, Ls, Ks, Apss):
    """Evaluate models for a specific configuration"""
    # Load models
    actor = load_model(path=path)
    loss_neg = load_model(path=path1)
    loss_inv = load_model(path=path2)

    models = {
        "DCB": actor,
        "Clustering (Rate)": None,
        "Aleatorio": None,
        "Loss-driven (Inv)": loss_inv,
        "Loss-driven (Neg)": loss_neg
    }

    # Initialize results storage
    time_results = {"métrica": "tiempo"}
    memory_results = {"métrica": "memoria"}
    
    possible_actions = generate_possible_actions(N_max, Apss)

    for episode in tqdm(range(num_episodes), desc=f"Evaluating L={Ls}, K={Ks}, APs={Apss}"):
        # Get data for this episode
        active_APs, df, gainOverNoisedB, powgain, max_gain = get_data(
            L=Ls, K=Ks, N=N, tau_p=tau_p, ASD_varphi=ASD_varphi, 
            numActiveAPs=Apss, grid=True, semilla=episode
        )

        # Prepare data
        agg_df = df.groupby('UE_id').agg({
            'AP_id': lambda x: list(x),
            'user_gain': lambda x: list(x),
            'interference_sum': lambda x: list(x)
        }).reset_index()

        agg_df['state'] = agg_df.apply(
            lambda x: np.concatenate((np.array(x['AP_id'], dtype=int), x['user_gain'], x['interference_sum'])), 
            axis=1
        )

        X_episode = prepare_data(agg_df)
        states_batch = np.stack(agg_df['state'].values)

        if episode == 0:  # Warm-up run
            continue

        # Evaluate each model
        for model_name, model in models.items():
            gc.collect()
            
            # Time measurement
            start_time = time.perf_counter()
            
            if model_name == "Clustering (Rate)":
                clustering_sum_rate_rate, _ = fuzzy_clustering_aps(
                    df, max_gain=max_gain, n_clusters=Ks, n_aps=Ls, 
                    act_aps=Apss, noise=N0, m=2, alpha=0.25, itf=True
                )
            elif model_name == "Aleatorio":
                random_sum_rate = 0
                for state in agg_df['state']:
                    random_action = random.choice(possible_actions)
                    random_sum_rate += calculate_sum_rate(np.array(state).reshape(1, -1), random_action, max_gain=max_gain)
            elif model_name == "DCB":
                actions = actor.predict(states_batch, verbose=0)
                episode_sum_rate = 0
                for i, state in enumerate(agg_df['state']):
                    binary_action = (actions[i] > 0.01).astype(int)
                    binary_action[Apss:] = 0
                    episode_sum_rate += calculate_sum_rate(np.array(state).reshape(1, -1), binary_action, max_gain=max_gain)
            else:  # Loss-driven models
                preds = model.predict(X_episode, verbose=0)
                model_sum_rate = 0
                for i, state in enumerate(agg_df['state']):
                    binary_action = (preds[i] > 0.5).astype(int)
                    model_sum_rate += calculate_sum_rate(np.expand_dims(state.flatten(), axis=0), binary_action, max_gain=max_gain)
            
            elapsed_time = time.perf_counter() - start_time
            
            # Memory measurement - fixed lambda functions
            gc.collect()
            try:
                if model_name == "Clustering (Rate)":
                    mem_usage = memory_usage(
                        (fuzzy_clustering_aps, (df,), {
                            'max_gain': max_gain, 'n_clusters': Ks, 'n_aps': Ls, 
                            'act_aps': Apss, 'noise': N0, 'm': 2, 'alpha': 0.25, 'itf': True
                        }))
                elif model_name == "Aleatorio":
                    mem_usage = memory_usage(
                        (lambda: sum(
                            calculate_sum_rate(
                                np.array(state).reshape(1, -1), 
                                random.choice(possible_actions), 
                                max_gain
                            ) for state in agg_df['state']
                        ), )
                    )
                elif model_name == "DCB":
                    actions = actor.predict(states_batch, verbose=0)
                    mem_usage = memory_usage(
                        (lambda: sum(
                            calculate_sum_rate(
                                np.array(state).reshape(1, -1), 
                                (actions[i] > 0.01).astype(int), 
                                max_gain
                            ) for i, state in enumerate(agg_df['state'])
                        ), ())
                    )
                else:  # Loss-driven models
                    preds = model.predict(X_episode, verbose=0)
                    mem_usage = memory_usage(
                        (lambda: sum(
                            calculate_sum_rate(
                                np.expand_dims(state.flatten(), axis=0), 
                                (preds[i] > 0.5).astype(int), 
                                max_gain
                            ) for i, state in enumerate(agg_df['state'])
                        ), ())
                    )
                
                peak_memory = max(mem_usage) - min(mem_usage)
                
                # Store results
                if model_name not in time_results:
                    time_results[model_name] = []
                    memory_results[model_name] = []
                
                time_results[model_name].append(elapsed_time)
                memory_results[model_name].append(peak_memory)
            except Exception as e:
                print(f"Error measuring memory for {model_name}: {str(e)}")
                continue

    # Calculate averages and prepare final DataFrame
    results = []
    
    # For time metrics
    time_avg = {"métrica": "tiempo"}
    for model in models:
        if model in time_results:
            time_avg[model] = np.mean(time_results[model])
    results.append(time_avg)
    
    # For memory metrics
    memory_avg = {"métrica": "memoria"}
    for model in models:
        if model in memory_results:
            memory_avg[model] = np.mean(memory_results[model])
    results.append(memory_avg)
    
    return pd.DataFrame(results)

def main():
    # Initialize results DataFrame
    final_results = pd.DataFrame()

    # Evaluate different configurations
    # for apss in [5, 10, 15, 20]:
    #     result = evaluate_configuration(num_episodes=30, Ls=50, Ks=30, Apss=apss)
    #     result.index = [f"{apss}", f"{apss}"]
    #     final_results = pd.concat([final_results, result])
    
    # final_results.to_csv('e2_apps.csv')

    # for kss in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
    #     result = evaluate_configuration(num_episodes=30, Ls=50, Ks=kss, Apss=10)
    #     result.index = [f"{kss}", f"{kss}"]
    #     final_results = pd.concat([final_results, result])
    
    # final_results.to_csv('e2_kss.csv')

    # for lss in [35, 40, 45, 55, 60, 65, 70, 75, 80, 85, 90, 300, 330, 360, 390, 400, 430, 460, 490, 500]:
    #     result = evaluate_configuration(num_episodes=30, Ls=lss, Ks=30, Apss=10)
    #     result.index = [f"{lss}", f"{lss}"]
    #     final_results = pd.concat([final_results, result])
    
    # final_results.to_csv('e_lss.csv')


    print(final_results)
