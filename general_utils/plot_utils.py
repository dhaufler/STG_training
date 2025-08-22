import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as ssd
import scipy.cluster.hierarchy as sch
import tensorflow as tf
import os



def compute_reordering(
    target_rates,
    clustering_method="ward",  # Options: "ward", "average", "single", "complete"
    metric="cosine",           # Options: "cosine", "euclidean", "cityblock", "correlation"
    baseline_subtract=False,
    log_transform=False,
    z_scored=False,
    clustering_interval=None  # Specify time interval for clustering relative to stimulus onset
):
    """
    Compute the reordering of neurons based on hierarchical clustering.

    Parameters:
    - target_rates: np.ndarray or tf.Tensor of shape [1, seq_len, num_target_cells].
    - clustering_method (str): Clustering method for hierarchical clustering.
                               Options: "ward", "average", "single", "complete".
    - metric (str): Distance metric for clustering.
                    Options: "cosine", "euclidean", "cityblock", "correlation".
    - baseline_subtract (bool): If True, subtract baseline (first 500 ms) before clustering.
    - log_transform (bool): If True, apply log-transform (log(target_rates + 1)) to the data.
    - z_scored (bool): If True, z-score the data for clustering.
    - clustering_interval (list or tuple, optional): Specify [start, end] time in ms for clustering 
                                                     relative to stimulus onset (500 ms). 
                                                     If None, use the full time range.

    Returns:
    - full_ordering: np.ndarray. Indices representing the reordering of neurons.
    """
    # Ensure input is a numpy array and remove batch dimension
    if isinstance(target_rates, tf.Tensor):
        target_rates = target_rates.numpy()
    target_rates = target_rates[0]  # Remove batch dimension: [seq_len, num_target_cells]

    # Log-transform the rates
    if log_transform:
        target_rates = np.log(target_rates + 1)  # Avoid log(0) by adding 1
        
    # Baseline subtraction
    if baseline_subtract:
        baseline_period = np.arange(0, 500)  # Assuming time starts at 0 ms
        baseline = np.mean(target_rates[baseline_period, :], axis=0)
        target_rates = target_rates - baseline

    # Z-score the rates
    if z_scored:
        target_rates = (target_rates - np.mean(target_rates, axis=0)) / (
            np.std(target_rates, axis=0) + 1e-8
        )
        # Should refactor this, and perform z-scoring using only non-sentence periods
        if baseline_subtract:
            baseline_period = np.arange(0, 500)  # Assuming time starts at 0 ms
            baseline = np.mean(target_rates[baseline_period, :], axis=0)
            target_rates = target_rates - baseline
            
    # Adjust clustering interval relative to stimulus onset at 500 ms
    if clustering_interval is not None:
        start_time, end_time = clustering_interval
        start_idx = max(0, start_time + 500)  # Adjust relative to the time axis
        end_idx = min(target_rates.shape[0], end_time + 500)
        target_rates_for_clustering = target_rates[start_idx:end_idx, :]
    else:
        target_rates_for_clustering = target_rates  # Use full range

    # Identify and exclude neurons with all-zero or constant firing rates
    variability_mask = np.std(target_rates_for_clustering, axis=0) > 0  # Neurons with non-zero variance

    # Ensure no NaN or infinite values in the selected data
    target_rates_for_clustering = np.nan_to_num(target_rates_for_clustering, nan=0.0, posinf=0.0, neginf=0.0)

    # Select only neurons with variability for clustering
    target_rates_variable = target_rates_for_clustering[:, variability_mask]  # [seq_len, num_variable_neurons]

    if target_rates_variable.shape[1] == 0:
        raise ValueError("No neurons with sufficient variability for clustering.")

    # Compute pairwise distances
    if metric == "correlation":
        # Use correlation as the distance metric
        pairwise_corr = np.corrcoef(target_rates_variable.T)
        pairwise_corr = np.nan_to_num(pairwise_corr, nan=1.0)  # Replace NaNs with 1 (perfect correlation)
        distance_matrix = 1 - pairwise_corr
    else:
        # Use specified metric
        distance_matrix = ssd.pdist(target_rates_variable.T, metric=metric)
        distance_matrix = ssd.squareform(distance_matrix)  # Convert to square distance matrix

    # Ensure distance matrix is symmetric
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    np.fill_diagonal(distance_matrix, 0)

    # Perform hierarchical clustering
    linkage_matrix = sch.linkage(distance_matrix, method=clustering_method)

    # Compute optimal leaf ordering
    optimal_order_variable = sch.leaves_list(linkage_matrix)

    # Map back to the full set of neurons
    full_ordering = np.arange(target_rates.shape[1])
    full_ordering[variability_mask] = optimal_order_variable

    return full_ordering

def compute_reordering_with_time_preference(
    target_rates,
    clustering_method="ward",  # Options: "ward", "average", "single", "complete"
    metric="cosine",           # Options: "cosine", "euclidean", "cityblock", "correlation"
    baseline_subtract=False,
    log_transform=False,
    z_scored=False,
    clustering_interval=None,  # Specify time interval for clustering relative to stimulus onset
    time_preference=True       # If True, sort neurons by time of max response before clustering
):
    """
    Compute the reordering of neurons based on hierarchical clustering,
    optionally incorporating time of maximal response for pre-ordering.

    Parameters:
    - target_rates: np.ndarray or tf.Tensor of shape [1, seq_len, num_target_cells].
    - clustering_method (str): Clustering method for hierarchical clustering.
                               Options: "ward", "average", "single", "complete".
    - metric (str): Distance metric for clustering.
                    Options: "cosine", "euclidean", "cityblock", "correlation".
    - baseline_subtract (bool): If True, subtract baseline (first 500 ms) before clustering.
    - log_transform (bool): If True, apply log-transform (log(target_rates + 1)) to the data.
    - z_scored (bool): If True, z-score the data for clustering.
    - clustering_interval (list or tuple, optional): Specify [start, end] time in ms for clustering 
                                                     relative to stimulus onset (500 ms). 
                                                     If None, use the full time range.
    - time_preference (bool): If True, pre-sort neurons by time of maximal response before clustering.

    Returns:
    - full_ordering: np.ndarray. Indices representing the reordering of neurons.
    """
    # Ensure input is a numpy array and remove batch dimension
    if isinstance(target_rates, tf.Tensor):
        target_rates = target_rates.numpy()
    target_rates = target_rates[0]  # Remove batch dimension: [seq_len, num_target_cells]

    # Log-transform the rates
    if log_transform:
        target_rates = np.log(target_rates + 1)  # Avoid log(0) by adding 1

    # Baseline subtraction
    if baseline_subtract:
        baseline_period = np.arange(0, 500)  # Assuming time starts at 0 ms
        baseline = np.mean(target_rates[baseline_period, :], axis=0)
        target_rates = target_rates - baseline

    # Z-score the rates
    if z_scored:
        target_rates = (target_rates - np.mean(target_rates, axis=0)) / (
            np.std(target_rates, axis=0) + 1e-8
        )
        # Baseline subtraction
        if baseline_subtract:
            baseline_period = np.arange(0, 500)  # Assuming time starts at 0 ms
            baseline = np.mean(target_rates[baseline_period, :], axis=0)
            target_rates = target_rates - baseline


    # Adjust clustering interval relative to stimulus onset at 500 ms
    if clustering_interval is not None:
        start_time, end_time = clustering_interval
        start_idx = max(0, start_time + 500)  # Adjust relative to the time axis
        end_idx = min(target_rates.shape[0], end_time + 500)
        target_rates_for_clustering = target_rates[start_idx:end_idx, :]
    else:
        target_rates_for_clustering = target_rates  # Use full range

    # Identify and exclude neurons with all-zero or constant firing rates
    variability_mask = np.std(target_rates_for_clustering, axis=0) > 0  # Neurons with non-zero variance

    # Select only neurons with variability for clustering
    target_rates_variable = target_rates_for_clustering[:, variability_mask]  # [seq_len, num_variable_neurons]

    # Compute time of maximal response
    if time_preference:
        max_response_times = np.argmax(target_rates_variable, axis=0)
        time_ordering = np.argsort(max_response_times)  # Order by earliest max response
        target_rates_variable = target_rates_variable[:, time_ordering]
    else:
        time_ordering = np.arange(target_rates_variable.shape[1])  # Default ordering

    # Compute pairwise distances
    if metric == "correlation":
        # Use correlation as the distance metric
        pairwise_corr = np.corrcoef(target_rates_variable.T)
        pairwise_corr = (pairwise_corr + pairwise_corr.T) / 2  # Ensure symmetry
        np.fill_diagonal(pairwise_corr, 1)
        distance_matrix = 1 - pairwise_corr
        linkage_matrix = sch.linkage(distance_matrix, method=clustering_method)  # Pass directly
    else:
        # Use specified metric
        distance_matrix = ssd.pdist(target_rates_variable.T, metric=metric)
        distance_matrix = ssd.squareform(distance_matrix)  # Convert to square form
        linkage_matrix = sch.linkage(distance_matrix, method=clustering_method)

    # Compute optimal leaf ordering
    optimal_order_variable = sch.leaves_list(linkage_matrix)

    # Combine time-based ordering with clustering
    combined_ordering = time_ordering[optimal_order_variable]

    # Map back to the full set of neurons
    full_ordering = np.arange(target_rates.shape[1])
    full_ordering[variability_mask] = combined_ordering

    return full_ordering

def plot_target_rates_and_model_spikes(target_rates, rollout_results, flags, epoch, step, reordering=None, z_scored=True, save = False):
    """
    Plots target firing rates, smoothed model spike rates, and model spikes raster,
    with optional reordering of the neurons and z-scored plotting.

    Parameters:
    - target_rates: np.ndarray or tf.Tensor of shape [1, seq_len, num_target_cells].
    - rollout_results: Dictionary containing 'spikes' as a tensor of shape [1, seq_len, n_neurons].
    - reordering: Optional np.ndarray. If provided, reorders the target rates and model spikes.
    - z_scored (bool): If True, z-score the data and use a blue-white-red diverging colormap.
    """
    # Ensure inputs are numpy arrays for compatibility
    if isinstance(target_rates, tf.Tensor):
        target_rates = target_rates.numpy()
    if isinstance(rollout_results['spikes'], tf.Tensor):
        model_spikes = rollout_results['spikes'].numpy()
    else:
        model_spikes = rollout_results['spikes']

    # Remove batch dimension
    target_rates = target_rates[0]  # Shape: [seq_len, num_target_cells]
    model_spikes = model_spikes[0]  # Shape: [seq_len, n_neurons]

    # Apply optional reordering
    if reordering is not None:
        target_rates = target_rates[:, reordering]
        model_spikes = model_spikes[:, reordering]

    # Smooth model spikes to compute smoothed spike rates
    model_spikes_3d = np.expand_dims(model_spikes, axis=0)  # Add batch dimension
    smoothed_model_rates = gaussian_smoothing(tf.convert_to_tensor(model_spikes_3d, dtype=tf.float32), sigma=50)
    smoothed_model_rates = smoothed_model_rates.numpy() * 1000  # Convert to Hz
    smoothed_model_rates = smoothed_model_rates[0]  # Remove batch dimension

    # Z-scoring option
    if z_scored:
        # target_rates_median = np.median(target_rates, axis=0)
        # target_rates_mad = np.median(np.abs(target_rates - target_rates_median), axis=0) + 1e-8
        # target_rates = (target_rates - target_rates_median) / target_rates_mad
        
        target_rates_mean = np.mean(target_rates, axis=0)
        target_rates_std = np.std(target_rates, axis=0) + 1e-8
        target_rates = (target_rates - target_rates_mean) / target_rates_std

        # smoothed_model_rates_median = np.median(smoothed_model_rates, axis=0)
        # smoothed_model_rates_mad = np.median(np.abs(smoothed_model_rates - smoothed_model_rates_median), axis=0) + 1e-8
        # smoothed_model_rates = (smoothed_model_rates - smoothed_model_rates_median) / smoothed_model_rates_mad
        
        smoothed_model_rates_mean = np.mean(smoothed_model_rates, axis=0)
        smoothed_model_rates_std = np.std(smoothed_model_rates, axis=0) + 1e-8
        smoothed_model_rates = (smoothed_model_rates - smoothed_model_rates_mean) / smoothed_model_rates_std

        cmap = "bwr"  # Diverging colormap for z-scored data
        vmin, vmax = -5, 5  # Z-score range for diverging colormap
       # vmin, vmax = -2, -5  # Auto-scale for z-scored data
    else:
        cmap = "Greys"
        vmin, vmax = None, None  # Auto-scale for raw data

    # Transpose for plotting (neurons on Y-axis, time on X-axis)
    target_rates = np.transpose(target_rates)  # Shape: [num_target_cells, seq_len]
    smoothed_model_rates = np.transpose(smoothed_model_rates)  # Shape: [n_neurons, seq_len]
    model_spikes = np.transpose(model_spikes)  # Shape: [n_neurons, seq_len]

    # Extract spike times and neurons for raster plot
    spike_neurons, spike_times = np.nonzero(model_spikes)

    # Create the figure and subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 10), constrained_layout=False)

    # Plot target rates heatmap
    im1 = axes[0].imshow(target_rates, aspect="auto", cmap=cmap, origin="lower",
                          vmin=vmin, vmax=vmax,  # Scaling based on z-scored or raw data
                          extent=[0, target_rates.shape[1], 0, target_rates.shape[0]])
    axes[0].set_title("Target Firing Rates Heatmap")
    axes[0].set_xlabel("Time (ms)")
    axes[0].set_ylabel("Target Cell Index")
    fig.colorbar(im1, ax=axes[0], label="Z-Score" if z_scored else "Firing Rate (Hz)")

    # Plot smoothed model spike rates heatmap
    im2 = axes[1].imshow(smoothed_model_rates, aspect="auto", cmap=cmap, origin="lower",
                          vmin=vmin, vmax=vmax,  # Scaling based on z-scored or raw data
                          extent=[0, smoothed_model_rates.shape[1], 0, smoothed_model_rates.shape[0]])
    axes[1].set_title("Smoothed Model Spike Rates Heatmap")
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("Neuron Index")
    fig.colorbar(im2, ax=axes[1], label="Z-Score" if z_scored else "Firing Rate (Hz)")

    # Plot model spikes raster
    axes[2].scatter(spike_times, spike_neurons, c="black", s=0.35, alpha=0.4, label="Spikes")
    axes[2].set_title("Model Spikes Raster")
    axes[2].set_xlim(0, model_spikes.shape[1])  # Match x-axis to target rates
    axes[2].set_ylim(0, model_spikes.shape[0])  # Match y-axis to number of neurons
    axes[2].set_xlabel("Time (ms)")
    axes[2].set_ylabel("Neuron Index")
    
    figName = 'target_rates_and_model_spikes_Epoch_' + str(epoch) + '_Step_' + str(step) + '.png'
    figDir = os.path.join(flags.checkpoint_savedir, 'figures')
    if not os.path.exists(figDir):
        os.makedirs(figDir)
    
    if save:
        plt.savefig(os.path.join(figDir, figName))
    # Show the plot
    plt.show()
    
def gaussian_smoothing(spikes, sigma=100):
    """
    Smooth binary spike trains using a Gaussian kernel along the time dimension.
    Args:
        spikes (tf.Tensor): [batch_size, seq_length, n_neurons], binary spikes.
        sigma (int): Standard deviation of the Gaussian kernel.

    Returns:
        tf.Tensor: Smoothed spike trains with the same shape as the input.
    """
    # Create Gaussian kernel
    kernel_size = int(6 * sigma + 1)  # Kernel spans 3 sigmas on each side
    x = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=tf.float32)
    gauss_kernel = tf.exp(-0.5 * (x / sigma) ** 2)
    gauss_kernel /= tf.reduce_sum(gauss_kernel)  # Normalize the kernel

    # Ensure spikes and kernel compatibility
    spikes_shape = tf.shape(spikes)
    gauss_kernel = tf.reshape(gauss_kernel, [-1, 1])  # Shape: [kernel_size, 1]

    # Pad the spikes to handle edges
    pad_size = kernel_size // 2
    spikes_padded = tf.pad(spikes, [[0, 0], [pad_size, pad_size], [0, 0]], mode="CONSTANT")

    # Smooth by sliding the kernel along the time dimension
    smoothed_spikes = tf.nn.depthwise_conv2d(
        input=tf.expand_dims(spikes_padded, axis=-1),  # Add channel dim: [batch, seq, n_neurons, 1]
        filter=tf.reshape(gauss_kernel, [kernel_size, 1, 1, 1]),  # 4D kernel for depthwise conv
        strides=[1, 1, 1, 1],
        padding="VALID"
    )
    smoothed_spikes = tf.squeeze(smoothed_spikes, axis=-1)  # Remove channel dim: [batch, seq, n_neurons]

    return smoothed_spikes

def plot_voltage_and_spikes_from_rollout(rollout_results):
    """
    Plots a 2D heatmap of the voltage outputs and a raster plot of spikes from rollout results.

    Parameters:
    - rollout_results: Dictionary containing 'voltage' and 'spikes' with shapes 
      [batch_size, seq_len, n_neurons].
    """
    # Extract voltage and spikes
    voltage = rollout_results['voltage']
    spikes = rollout_results['spikes']

    # Ensure inputs are numpy arrays
    voltage = voltage.numpy() if hasattr(voltage, "numpy") else voltage
    spikes = spikes.numpy() if hasattr(spikes, "numpy") else spikes

    # Ensure batch size is 1
    if voltage.ndim != 3 or voltage.shape[0] != 1:
        raise ValueError("Voltage and spikes must have shape [1, seq_len, n_neurons].")

    # Remove batch dimension
    voltage = voltage[0]  # Shape: [seq_len, n_neurons]
    spikes = spikes[0]    # Shape: [seq_len, n_neurons]

    # Debugging shapes
    print("Voltage shape after batch removal:", voltage.shape)
    print("Spikes shape after batch removal:", spikes.shape)

    # Transpose for plotting (neurons on Y-axis, time on X-axis)
    voltage = voltage.T  # Shape: [n_neurons, seq_len]
    spikes = spikes.T    # Shape: [n_neurons, seq_len]

    # Debugging shapes after transpose
    print("Voltage shape after transpose:", voltage.shape)
    print("Spikes shape after transpose:", spikes.shape)

    # Check spike distribution
    spike_neurons, spike_times = np.where(spikes > 0)  # Corrected order
    print("Total spikes:", len(spike_times))
    print("Spike times snippet:", spike_times[:10])
    print("Spike neurons snippet:", spike_neurons[:10])

    # Create the figure and subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 10), constrained_layout=False)

    # Plot voltage heatmap
    im1 = axes[0].imshow(voltage, aspect="auto", cmap="viridis", origin="lower",
                          extent=[0, voltage.shape[1], 0, voltage.shape[0]])
    axes[0].set_title("Voltage Heatmap")
    axes[0].set_xlabel("Time (ms)")
    axes[0].set_ylabel("Neuron Index")
    #fig.colorbar(im1, ax=axes[0], label="Voltage (a.u.)")

    # Plot spikes raster
    axes[1].scatter(spike_times, spike_neurons, c="black", s=1, label="Spikes")  # Corrected order
    axes[1].set_title("Spike Raster")
    axes[1].set_xlim(0, voltage.shape[1])  # Match x-axis to voltage heatmap
    axes[1].set_ylim(0, voltage.shape[0])  # Match y-axis to voltage heatmap
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("Neuron Index")

    #axes[0].set_xlim([0, 500])
    #axes[1].set_xlim([0, 500])
    #axes[0].set_ylim([0, 100])
    #axes[1].set_ylim([0, 100])
    # Show the plot
    plt.show()