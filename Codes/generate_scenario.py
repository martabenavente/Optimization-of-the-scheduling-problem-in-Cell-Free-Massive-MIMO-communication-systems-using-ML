from functionsSetup import generateSetup
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_data(L, K, N, tau_p, ASD_varphi, numActiveAPs, grid, semilla = False, N_max=20):

    # Llamamos a la función para generar los parámetros del entorno
    gainOverNoisedB, powgain, active_APs, distances = generateSetup(
        L=L, K=K, N=N, tau_p=tau_p, ASD_varphi=ASD_varphi, numActiveAPs=numActiveAPs, grid=grid, semilla=semilla
    )
    max_gain = np.max(powgain)
    # ----------------------------------------------------------------------------------------------------------------

    # print(f'gainOverNoisedB shape: {gainOverNoisedB.shape}')
    # print(f'Sumas ap 0: {np.sum(gainOverNoisedB[0, np.arange(K) != 1])}')
    # print(f'gainOverNoisedB: {gainOverNoisedB}')
    # print(f'Active APs: {active_APs}')
    # print(f'powgain: {powgain}')
    # print(f'AP positions: {APpositions}')
    # print(f'UE positions: {UEpositions}')

    # Visualizamos las posiciones
    # plt.figure(figsize=(8, 8))
    # plt.plot(APpositions.real, APpositions.imag, 'g*', label='APs')
    # plt.plot(UEpositions.real, UEpositions.imag, 'r*', label='UEs')
    # plt.xlabel('Coordenada X (metros)')
    # plt.ylabel('Coordenada Y (metros)')
    # plt.title('Posiciones iniciales de APs y UEs')
    # plt.legend()
    # plt.grid()
    # plt.savefig("ordenadas.png")

    # Inicializar la lista para almacenar los datos
    data = []

    # Construir cada fila del DataFrame
    for ue_id in range(K):  # Recorrer cada usuario
        max_active = max(active_APs[ue_id])
        for ap_id in active_APs[ue_id]:
            # Crear un diccionario para cada combinación AP-UE
            data.append({
                "UE_id": ue_id,  # Identificador del usuario
                "AP_id": ap_id,  # Identificador de la AP
                "user_gain": (powgain[ap_id, ue_id])/(max_active),
                "interference_sum": (np.sum(powgain[ap_id, :]) - powgain[ap_id, ue_id])/(max_active),
                # "interference_sum": np.log1p(np.sum(powgain[:, np.arange(K) != ue_id], axis=1)[ap_id]),  # Interferencia as Suma de la ganancia del resto de usuarios relevantes para la msima AP
                "distances": distances[ap_id, ue_id]
            })

        for ap_id in range(numActiveAPs+1, N_max+1):
            data.append({
                "UE_id": ue_id,
                "AP_id": ap_id + L,  # Indicar que es una AP "falsa" o de relleno
                "user_gain": 0.0,
                "interference_sum": 0.0,
                "distances": 0.0
            })

    # Convertir la lista a un DataFrame de pandas
    df_simulation = pd.DataFrame(data)
    # print(df_simulation.head(21))

    return active_APs, df_simulation, gainOverNoisedB, powgain, max_gain

# get_data(50, 30, 1, 4, 10 * (3.14159 / 180), 10, grid=True, semilla=1)