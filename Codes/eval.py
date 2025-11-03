import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
from generate_scenario import get_data
import matplotlib.pyplot as plt
from tqdm import tqdm
from modelo_1 import load_model, calculate_sum_rate, generate_possible_actions
from fuzzy_clustering import fuzzy_clustering_aps, buscar_threshold_optimo
from loss_based import prepare_data
import random

# Definimos los parámetros iniciales para la simulación
N_max = 20
N = 1   # Número de antenas por AP
tau_p = 4  # Número de pilotos ortogonales
ASD_varphi = 10 * (3.14159 / 180)  # Desviación estándar angular para azimuth (en radianes)


# Parámetros del modelo
state_dim = 3  # Dimensión del estado (UE_id, AP_id, user_gain, interference_sum)
N0 = 1e-9  # Ruido térmico (valor arbitrario, ajustable)
num_episodes = 500  # Número de simulaciones
num_iterations = 1000  # Iteraciones de entrenamiento

path = '/mnt/c/Users/marta/Desktop/TFG/Codes/Mod/actor.keras'
path1 = '/mnt/c/Users/marta/Desktop/TFG/Codes/Mod/loss_neg.keras'
path3 = '/mnt/c/Users/marta/Desktop/TFG/Codes/Mod/loss_inv.keras'

seeds = [42, 77, 101, 123, 294, 326, 400, 456, 505, 678, 733, 789, 826, 914, 960]


# Evaluación del modelo después del entrenamiento
def evaluate_model(num_episodes=10, Ls=50, Ks=30, Apss=10, possible_actions=False):
    total_sum_rates = []
    random_sum_rates = []  
    clustering_sum_rates_rate = []
    ap_selection_sum_rates = []
    minus_ap_selection_sum_rates = []

    actor = load_model(path=path)  # Cargar el modelo guardado
    dariel = load_model(path=path1)
    oscar = load_model(path=path3)

    for episode in tqdm(range(num_episodes), desc="Evaluando modelo"):
        # semilla = random.choice(seeds)
        active_APs, df, gainOverNoisedB, powgain, max_gain = get_data(L=Ls, K=Ks, N=N, tau_p=tau_p, ASD_varphi=ASD_varphi, numActiveAPs=Apss,
                                                            grid=True, semilla=episode)

        agg_df = df.groupby('UE_id').agg({
            'AP_id': lambda x: list(x),
            'user_gain': lambda x: list(x),
            'interference_sum': lambda x: list(x)
        }).reset_index()

        # Crear vectores de estado combinando las columnas
        agg_df['state'] = agg_df.apply(lambda x: np.concatenate((np.array(x['AP_id'], dtype=int), x['user_gain'], x['interference_sum'])), axis=1)

        X_episode = prepare_data(agg_df)

        episode_sum_rate = 0
        random_sum_rate = 0
        ap_selection_sum_rate = 0
        minus_ap_selection_sum_rate = 0
        threshold, sum_rate = buscar_threshold_optimo(df, max_gain, Ks, Ls, Apss, noise=N0, itf=True)
        clustering_sum_rate_rate, _ = fuzzy_clustering_aps(df, max_gain=max_gain, n_clusters=Ks, n_aps=Ls, act_aps=Apss, noise=N0, m=2, alpha=0.25, itf=True, threshold=threshold)

        states_batch = np.stack(agg_df['state'].values)
        actor_preds = actor.predict(states_batch, verbose=0)
        oscar_preds = oscar.predict(X_episode, verbose=0)
        dariel_preds = dariel.predict(X_episode, verbose=0)

        for i, state in enumerate(agg_df['state']):
            random_action = random.choice(possible_actions)
            random_reward = calculate_sum_rate(np.array(state).reshape(1, -1), random_action, max_gain=max_gain)
            random_sum_rate += random_reward

            action = actor_preds[i]
            binary_action = (action > 0.01).astype(int)
            binary_action[Apss:] = 0
            reward = calculate_sum_rate(np.array(state).reshape(1, -1), binary_action, max_gain=max_gain)
            episode_sum_rate += reward

            action_ap_selection = oscar_preds[i]
            binary_action_ap = (action_ap_selection > 0.5).astype(int)
            reward_ap = calculate_sum_rate(np.expand_dims(state.flatten(), axis=0), binary_action_ap, max_gain=max_gain)
            ap_selection_sum_rate += reward_ap

            minus_action_ap_selection = dariel_preds[i]
            minus_binary_action_ap = (minus_action_ap_selection > 0.5).astype(int)
            minus_reward_ap = calculate_sum_rate(np.expand_dims(state.flatten(), axis=0), minus_binary_action_ap, max_gain=max_gain)
            minus_ap_selection_sum_rate += minus_reward_ap

        random_sum_rates.append(random_sum_rate)
        total_sum_rates.append(episode_sum_rate)
        clustering_sum_rates_rate.append(clustering_sum_rate_rate)
        ap_selection_sum_rates.append(ap_selection_sum_rate)
        minus_ap_selection_sum_rates.append(minus_ap_selection_sum_rate)

    # Almacenar métricas
    results = {
        "Aleatorio": [np.mean(random_sum_rates)],
        "DCB": [np.mean(total_sum_rates)],
        "Clustering (Rate)": [np.mean(clustering_sum_rates_rate)],
        "Loss-driven (Inv)": [np.mean(ap_selection_sum_rates)],
        "Loss-driven (Neg)": [np.mean(minus_ap_selection_sum_rates)]
    }
    
    return results

df_results = pd.DataFrame()

# for apss in [5, 10, 15, 20]:
#     pos_act = generate_possible_actions(20, apss)
#     result = evaluate_model(num_episodes=100, Ls=50, Ks=30, Apss=apss, possible_actions=pos_act)
#     result_df = pd.DataFrame(result, index=[f"{apss}"])
#     df_results = pd.concat([df_results, result_df])
#     print(df_results)

# df_results.to_csv('apps-1500.csv')

# for kss in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
#     pos_act = generate_possible_actions(20, 10)
#     result = evaluate_model(num_episodes=100, Ls=50, Ks=kss, Apss=10, possible_actions=pos_act)
#     result_df = pd.DataFrame(result, index=[f"50_{kss}_10"])
#     df_results = pd.concat([df_results, result_df])
#     print(df_results)

# df_results.to_csv('kss-1500.csv') 

# 35, 40, 45, 55, 60, 65, 70, 75, 80, 85, 90, 300, 330, 360, 390, 400, 430, 460, 490, 500
# for lss in [35, 40, 45, 55, 60, 65, 70, 75, 80, 85, 90]:
#     pos_act = generate_possible_actions(20, 10)
#     result = evaluate_model(num_episodes=100, Ls=lss, Ks=30, Apss=10, possible_actions=pos_act)
#     result_df = pd.DataFrame(result, index=[f"{lss}"])
#     df_results = pd.concat([df_results, result_df])
#     print(df_results)

# df_results.to_csv('lss-1500.csv')

print("Guardado")