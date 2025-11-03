import numpy as np
import skfuzzy as fuzz

def fuzzy_clustering_aps(data, max_gain, n_clusters, n_aps, act_aps, noise, m=2, alpha=1, itf=False, threshold=None):
    if threshold is None:
        threshold = 1 * np.exp(-0.02 * act_aps * n_clusters)

    total_sum_rate = 0
    sum_rate_per_ue = np.zeros(n_clusters)
    aps_selected_list = []

    # Extract metrics
    user_gain = np.zeros((n_clusters, n_aps), dtype=float)
    interference_sum = np.zeros((n_clusters, n_aps), dtype=float)
    distancias = np.zeros((n_clusters, n_aps), dtype=float)
    
    for _, row in data.iterrows():
        if row['AP_id'] < n_aps:
            ue_id = int(row['UE_id'])
            ap_id = int(row['AP_id'])
            user_gain[ue_id, ap_id] = row['user_gain']
            interference_sum[ue_id, ap_id] = row['interference_sum']
            distancias[ue_id, ap_id] = row['distances']

    # Perform fuzzy clustering
    if itf:
        sum_rates = np.log2(1 + (user_gain / (interference_sum + noise)))
        masked_sum_rates = np.where(sum_rates > 0, sum_rates, np.nan)
        inverted = 1 / masked_sum_rates
        clustering_data = np.nan_to_num(inverted, nan=np.nanmax(inverted) * 2)
    else:
        max_dist = np.max(distancias[distancias > 0])
        distancias[distancias == 0] = 2 * max_dist
        clustering_data = distancias

    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        clustering_data,
        c=n_clusters,
        m=m,
        error=0.005,
        maxiter=1000
    )

    # Process each UE's AP selection
    grouped = data.groupby('UE_id')
    
    for ue_id, ue_data in grouped:
        ap_indices = ue_data.index.to_numpy() - (20 * ue_id)
        ue_cluster = ue_id % n_clusters
        
        # Get metrics for this UE's APs
        memberships = u[ue_cluster, ap_indices]
        gains = ue_data['user_gain'].to_numpy()
        interference = ue_data['interference_sum'].to_numpy()
        
        # Calculate potential rates for all APs
        sinr = (gains * max_gain) / (interference * max_gain + noise)
        potential_rates = np.log2(1 + sinr)
        
        # Dynamic threshold based on act_aps availability
        dynamic_threshold = threshold * (1 - 0.5*(act_aps/n_aps))  # More APs → lower threshold
        
        # Combine membership with rate potential
        combined_scores = memberships * potential_rates
        
        # Sort APs by their combined score
        sorted_aps = np.argsort(combined_scores)[::-1]  # Best first
        
        # Select APs: must have positive rate contribution
        selected_aps = []
        for ap_idx in sorted_aps[:act_aps]:
            if potential_rates[ap_idx] > 0.15:  # Only select if positive contribution
                selected_aps.append(ap_idx)
            if len(selected_aps) >= act_aps:
                break
        
        # Ensure at least one AP is selected
        if not selected_aps:
            selected_aps = [np.argmax(potential_rates)]  # Select best available
            
        # Create final AP selection vector
        final_aps = np.zeros_like(memberships)
        final_aps[selected_aps] = 1
        
        # Calculate actual sum-rate
        active_sinr = (final_aps * (gains * max_gain)) / ((interference * max_gain) + noise)
        sum_rate = np.sum(np.log2(1 + active_sinr))
        
        # Update metrics
        num_aps_selected = len(selected_aps)
        aps_selected_list.append(num_aps_selected)
        total_sum_rate += sum_rate
        sum_rate_per_ue[ue_id] = sum_rate

    # print(f"Media de APs seleccionadas por UE: {np.mean(aps_selected_list):.2f}")
    # print(f"Distribución: {np.bincount(np.array(aps_selected_list, dtype=int))}")
    return total_sum_rate, sum_rate_per_ue

def buscar_threshold_optimo(data, max_gain, n_clusters, n_aps, act_aps, noise, m=2, alpha=1.0, itf=True, low=0.01, high=0.5, tol=1e-3, max_iter=20):
    best_threshold = low
    best_sum_rate = -np.inf

    for _ in range(max_iter):
        mid = (low + high) / 2
        # print(f"Probando threshold={mid:.4f}")

        sum_rate, _ = fuzzy_clustering_aps(
            data, max_gain, n_clusters, n_aps, act_aps, noise,
            m=m, alpha=alpha, itf=itf, threshold=mid
        )

        sum_rate_low, _ = fuzzy_clustering_aps(
            data, max_gain, n_clusters, n_aps, act_aps, noise,
            m=m, alpha=alpha, itf=itf, threshold=low
        )
        sum_rate_high, _ = fuzzy_clustering_aps(
            data, max_gain, n_clusters, n_aps, act_aps, noise,
            m=m, alpha=alpha, itf=itf, threshold=high
        )

        if sum_rate > best_sum_rate:
            best_sum_rate = sum_rate
            best_threshold = mid

        if sum_rate_low > sum_rate_high:
            high = mid
        else:
            low = mid

        if high - low < tol:
            break

    # print(f"Threshold óptimo = {best_threshold:.4f}, Sum-rate = {best_sum_rate:.2f}")
    sum_rate_final, _ = fuzzy_clustering_aps(
        data, max_gain, n_clusters, n_aps, act_aps, noise,
        m=m, alpha=alpha, itf=itf, threshold=best_threshold
    )

    # print(f"Threshold óptimo = {best_threshold:.4f}, Sum-rate = {best_sum_rate:.2f}")
    return best_threshold, best_sum_rate
