# STG Training functions

import os
import shutil
import h5py
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
from IPython.display import clear_output
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.special import gammaln

# Adaptive Learning Rate calculation--just for reference
# Usage
# adjusted_lr = calculate_adaptive_lr(0.00025, len(StimID_training_original))
def calculate_adaptive_lr(n_stimuli, base_lr = 0.000035, avg_stimulus_duration_ms=2000, chunk_ms=500):
    """
    Calculate learning rate based on number of stimuli and chunking structure.
    """
    # Estimate chunks per stimulus
    chunks_per_stimulus = avg_stimulus_duration_ms / chunk_ms
    
    # Total gradient updates per epoch
    total_updates_per_epoch = n_stimuli * chunks_per_stimulus
    
    # Scale learning rate inversely with sqrt of total updates
    base_updates = 1 * (2500 / 500)  # 2 stimuli baseline
    scale_factor = np.sqrt(base_updates / total_updates_per_epoch)
    
    # Apply scaling with reasonable bounds
    adjusted_lr = base_lr * scale_factor
    adjusted_lr = np.clip(adjusted_lr, base_lr * 0.05, base_lr * 2.0)
    
    return adjusted_lr


def optimizers_match(current_optimizer, checkpoint_directory):
    current_optimizer_vars = {v.name: v.shape.as_list() for v in current_optimizer.variables()}
    checkpoint_vars = tf.train.list_variables(checkpoint_directory)
    checkpoint_optimizer_vars = {name.split('/.ATTRIBUTES')[0]: value for name, value in checkpoint_vars if 'optimizer' in name}
    if len(current_optimizer_vars) != len(checkpoint_optimizer_vars)-1:
        print('Checkpoint optimizer variables do not match the current optimizer variables.. Renewing optimizer...')
        return False
    else:
        for name, desired_shape in current_optimizer_vars.items():
            var_not_matched = True
            for opt_var, opt_var_shape in checkpoint_optimizer_vars.items():
                if opt_var_shape == desired_shape: 
                    var_not_matched = False
                    del checkpoint_optimizer_vars[opt_var]
                    break
            if var_not_matched:
                print(f'{name} does not have a match')
                return False
        return True


def check_model_dirs(network_path, net_files_path):
    
    # Check to ensure network directory has necessary files

    # check if subdirectory components exists
    if os.path.isdir(os.path.join(network_path, "components")):
            print(f"components dir exists.")
    else:
        shutil.copytree(os.path.join(net_files_path, "components"), os.path.join(network_path, "components"))
        print(f"copied over components dir.")
        
    # check if target_rates sym link exists
    if os.path.isdir(os.path.join(network_path, "target_rates")):
            print(f"target_rates dir exists.")
    else:
        shutil.copy(os.path.join(net_files_path, "target_rates"), os.path.join(network_path, "target_rates"), follow_symlinks=False)
        print(f"copied over target_rates dir.")
        
    # check if aud_in_spikes exists
    if os.path.isdir(os.path.join(network_path, "aud_in_spikes")):
            print(f"input spikes dir exists.")
    elif os.path.isdir(os.path.join(network_path, "filter_spikes")):
            os.rename(os.path.join(network_path, "filter_spikes"), os.path.join(network_path, "aud_in_spikes"))
            print(f"input spikes dir exists.")
    else:
            print(f"input spikes dir does not exist. Using default.")
            shutil.copytree(os.path.join(net_files_path, "aud_in_spikes"), os.path.join(network_path, "aud_in_spikes"))
        

def count_nodes(file_path, population_name=None):
    """
    Count the number of nodes in a specific population of a Sonata file.
    
    Parameters:
        file_path (str): Path to the Sonata file.
        population_name (str, optional): Specific population name to search for.
    Returns:
        int: Number of nodes in the specified population or total nodes if None.
    """
    try:
        with h5py.File(file_path, 'r') as h5file:
            if 'nodes' in h5file:
                nodes_group = h5file['nodes']
                if population_name:
                    if population_name in nodes_group:
                        population_group = nodes_group[population_name]
                        if 'node_id' in population_group:
                            return len(population_group['node_id'])
                        else:
                            print(f"'node_id' not found in population '{population_name}'.")
                            return 0
                    else:
                        print(f"Population '{population_name}' not found in the file.")
                        return 0
                else:
                    total_nodes = 0
                    for pop_name in nodes_group.keys():
                        population_group = nodes_group[pop_name]
                        if 'node_id' in population_group:
                            total_nodes += len(population_group['node_id'])
                    return total_nodes
            else:
                print("No 'nodes' group found in the file.")
                return 0
    except Exception as e:
        print(f"An error occurred while reading the file '{file_path}': {e}")
        return 0
    
#@tf.function  
def roll_out(_x, flags, extractor_model, state_variables):
    """
    Perform a forward pass through the extractor model.
    simulates one forward pass through the neural network model using the 
    provided inputs and initial state.


    Args:
        _x: LGN input tensor of shape (batch_size, seq_len, n_input).
        seq_len: Length of the sequence to process.
        extractor_model: Extractor model to use for forward pass.
        state_variables: State variables to update during the forward pass.

    Returns:
        Dictionary with spikes and voltage outputs.
    """
    # Read initial state
    #_initial_state = tf.nest.map_structure(lambda a: a.read_value(), state_variables)

    # Create dummy zeros for background inputs
    dummy_zeros = tf.zeros((flags.batch_size, flags.seq_len, flags.neurons), dtype=tf.float32)

    # Debugging shapes
    #print(f"LGN Input (_x) shape: {_x.shape}")
    #print(f"Background Input (dummy_zeros) shape: {dummy_zeros.shape}")

    # Forward pass through extractor model
    _out, _p = extractor_model([_x, dummy_zeros])  # Pass inputs as a list

    # Extract outputs
    _z, _v, _ = _out[0]  # Spikes (_z), Voltage (_v)

    # Update state variables
    new_state = tuple(_out[1:])
    if state_variables is not None:
        tf.nest.map_structure(lambda a, b: a.assign(b), state_variables, new_state)


    return {
        "spikes": _z,
        "voltage": _v,
    }

# @tf.function
# def roll_out_chunk(inputs, init_state, *, training=True):
#     """
#     inputs     : (k, B, T, n_input)   –  k noisy repeats already stacked
#     init_state : nested tuple matching rsnn_layer.cell.get_initial_state()
#     returns    : spikes (k,B,T,N) , final_state
#     """
#     k, B, T, _ = tf.shape(inputs)
#     dummy_bkg = tf.zeros((k, B, T, flags.neurons), dtype=inputs.dtype)

#     # Flatten k & B for a single call into the model
#     flat_in  = tf.reshape(inputs,  (k*B, T, flags.n_input))
#     flat_bkg = tf.reshape(dummy_bkg,(k*B, T, flags.neurons))
#     flat_ini = tf.nest.map_structure(
#         lambda x: tf.repeat(x, repeats=k, axis=0),  # tile state k times
#         init_state)

#     rsnn_out, final_state = model([flat_in, flat_bkg, flat_ini],
#                                   training=training)

#     spikes, *_ = rsnn_out[0]                    # (k*B,T,N)
#     spikes = tf.reshape(spikes, (k, B, T, flags.neurons))

#     # Collapse k back to 1 when returning state: choose k=0 arbitrarily
#     # (all repeats use identical initial state so final_state[0] suffices)
#     final_state = tf.nest.map_structure(lambda x: x[0], final_state)

#     return spikes, final_state


def load_chunk_batch(
        stim_batch,               # list of StimIDs, length = B
        sampler_dict,             # dict : StimID → iterator of (start,end)
        k_repeats,                # noisy repeats per chunk (k)
        *,                        # --- keyword-only below ------------------
        uniq_target_ids,          # np.ndarray / list, length = n_targets
        n_targets,                # int  (len(uniq_target_ids))
        flags,                    # absl FLAGS object (n_input, batch_size, …)
        StimID_to_StimName,       # global mapping loaded in main
        sentence_state_dict       # StimID → RNN-state tuple
):
    """
    Slice a chunk from each stimulus in `stim_batch`, replicate it `k_repeats`
    times with tiny uniform noise, and return tensors ready for training.

    Shapes
    -------
        inputs_tensor   : (k, B, T, n_input)
        targets_tensor  : (B, T, n_targets)     ← **pooled dimension**
        init_states     : list[tuple], len = B
    """
    import tensorflow as tf          # local import keeps utils self-contained

    inputs_list, targets_list, init_states = [], [], []

    for StimID in stim_batch:
        # ─── choose window -------------------------------------------------
        start_idx, end_idx = next(sampler_dict[StimID])
        chunk_len          = end_idx - start_idx

        # ─── load data for this window ------------------------------------
        aud_in_spikes, target_rates, _ = tu.load_aud_input_and_target_rates(
            StimID             = StimID,
            input_dir          = flags.input_dir,
            target_rates_dir   = flags.target_rates_dir,
            StimID_to_StimName = StimID_to_StimName,
            seq_len            = chunk_len,
            n_input            = flags.n_input,
            batch_size         = flags.batch_size,
            t_start            = start_idx,
            N_neurons          = n_targets,          # ← pooled size
            target_cell_ids    = uniq_target_ids,    # ← unique IDs
            repeat_number      = None
        )

        # ─── k noisy repeats ----------------------------------------------
        repeats = [
            aud_in_spikes + tf.random.uniform(
                shape=tf.shape(aud_in_spikes),
                minval=-1e-6, maxval=1e-6, dtype=aud_in_spikes.dtype
            )
            for _ in range(k_repeats)
        ]
        inputs_list .append(tf.stack(repeats, axis=0))   # (k, T, n_input)
        targets_list.append(target_rates)                # (T, n_targets)
        init_states .append(sentence_state_dict[StimID])

    # ─── stack across batch B ---------------------------------------------
    inputs_tensor  = tf.stack(inputs_list,  axis=1)      # (k, B, T, n_input)
    targets_tensor = tf.stack(targets_list, axis=0)      # (B, T, n_targets)

    return inputs_tensor, targets_tensor, init_states



# def randomize_weights(model, input_type, shape, scale, zero_percentage=0.0):
#     if input_type == "aud_in":
#         layer = model.get_layer("input_layer")
#         layer_name = "aud_input"
#     elif input_type == "bkg":
#         layer = model.get_layer("noise_layer")
#         layer_name = "bkg_input"
#     else:
#         raise ValueError("Invalid input_type. Choose 'aud_in' or 'bkg'.")

#     # Get current weights
#     weights = layer.get_weights()
#     print("Initial weights median:", np.median(weights[0][weights[0] > 0]))

#     # Randomize weights
#     randomized_weights = np.random.lognormal(mean=np.log(scale), sigma=shape, size=weights[0].shape)

#     # Zero out weights if needed
#     flat_weights = randomized_weights.flatten()
#     if zero_percentage > 0.0:
#         num_zero_weights = int(len(flat_weights) * (zero_percentage / 100))
#         zero_indices = np.random.choice(len(flat_weights), num_zero_weights, replace=False)
#         flat_weights[zero_indices] = 0
#     randomized_weights = flat_weights.reshape(weights[0].shape)

#     # Update weights
#     weights[0] = randomized_weights
#     layer.set_weights(weights)

#     # Debugging: Check updated weights
#     updated_weights = layer.get_weights()[0]
#     print(f"Updated {layer_name} weights with lognormal distribution (shape={shape}, scale={scale}, zero_percentage={zero_percentage}).")
#     print("Updated weights median (excluding zeros):", np.median(updated_weights[updated_weights > 0]))

#     return np.median(updated_weights[updated_weights > 0])

def randomize_weights(
        model, *, input_type,
        shape,              # lognormal σ
        scale,              # lognormal median for aud_in
        zero_percentage=0.0,
        neg_fraction=0.5,   # frac. to flip sign for aud_in
        bkg_multiplier=1.5  # >1 → bkg weights larger
):
    if input_type == "aud_in":
        layer       = model.get_layer("input_layer")
        layer_name  = "aud_input"
        eff_scale   = scale                          # keep original median
    elif input_type == "bkg":
        layer       = model.get_layer("noise_layer")
        layer_name  = "bkg_input"
        eff_scale   = scale * bkg_multiplier         # 50 % larger median
    else:
        raise ValueError("input_type must be 'aud_in' or 'bkg'.")

    # — current weights just for logging —
    w0 = layer.get_weights()
    print(f"{layer_name} initial median(|w|) :", np.median(np.abs(w0[0][w0[0] != 0])))

    # — draw positive magnitudes from log-normal —
    magnitudes = np.random.lognormal(mean=np.log(eff_scale),
                                     sigma=shape,
                                     size=w0[0].shape)

    # — optionally flip signs (aud_in only) —
    if input_type == "aud_in" and neg_fraction > 0:
        mask = np.random.rand(*magnitudes.shape) < neg_fraction
        magnitudes[mask] *= -1.0                    # half become negative

    # — optional sparsity —
    if zero_percentage > 0.0:
        flat = magnitudes.flatten()
        k    = int(len(flat) * zero_percentage / 100)
        flat[np.random.choice(len(flat), k, replace=False)] = 0.0
        magnitudes = flat.reshape(magnitudes.shape)

    # — write back to the layer —
    w0[0] = magnitudes.astype(w0[0].dtype)
    layer.set_weights(w0)

    med = np.median(np.abs(magnitudes[magnitudes != 0]))
    print(f"{layer_name} updated median(|w|) :", med, "\n")
    return med


def load_aud_input_and_target_rates(
        StimID: int,
        input_dir: str,
        target_rates_dir: str,
        StimID_to_StimName: dict,
        seq_len: int,
        n_input: int,
        batch_size: int,
        t_start: int | float,
        N_neurons: int,
        target_cell_ids,
        repeat_number: int | None = None,
):
    """
    Load a slice of auditory–input spikes together with the *pooled* target
    firing-rate traces for a single stimulus.

    Parameters
    ----------
    StimID : int
        Numerical ID used in the folder / file names (e.g. `stim_101_repeat_0`).
    input_dir : str
        Directory that contains subfolders `stim_<ID>_repeat_<k>/spikes.csv`.
    target_rates_dir : str
        Folder that contains one CSV per dataset cell:
        ``<cell_id>__<StimName>.csv`` with a column “Rate”.
    StimID_to_StimName : dict[int,str]
        Maps *StimID* to the textual stimulus identifier used in those CSV
        file names.
    seq_len : int
        Desired window length in milliseconds.
    n_input : int
        Number of *auditory input* neurons in the network.
    batch_size : int
        How many independent batch copies to create (usually 1).
    t_start : int | float
        Start time (in ms) of the window to slice from the full stimulus.
    N_neurons : int
        **Pooled target dimension** — typically *n_targets* (≈ 1090), *not*
        the total number of model neurons.
    target_cell_ids : Sequence[str | int]
        *Unique* dataset-cell IDs whose rate traces should be loaded and
        ordered; length must equal `N_neurons`.
    repeat_number : int | None, optional
        If given, load that exact repeat; otherwise choose one at random.

    Returns
    -------
    aud_in_spikes : np.ndarray, shape = (batch_size, seq_len, n_input)
        Binary spike tensor for the auditory input layer.
    target_rates  : np.ndarray, shape = (batch_size, seq_len, N_neurons)
        Smoothed-rate targets aligned with the order of `target_cell_ids`.
    chosen_dir    : str
        The directory from which the spikes were loaded (useful for debugging).
    """
    import os, random
    import numpy as np
    import pandas as pd

    # ------------------------------------------------------------
    # 0. Resolve stimulus name
    # ------------------------------------------------------------
    if StimID not in StimID_to_StimName:
        raise KeyError(f"StimID {StimID} missing from StimID_to_StimName.")
    StimName = StimID_to_StimName[StimID]

    # ------------------------------------------------------------
    # 1. Locate repeat directory and spikes.csv
    # ------------------------------------------------------------
    sub_dirs = [
        d for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d)) and d.startswith(f"stim_{StimID}_repeat_")
    ]
    if not sub_dirs:
        raise FileNotFoundError(f"No sub-directories for StimID {StimID} in {input_dir}")

    if repeat_number is not None:
        chosen_dir = os.path.join(input_dir, f"stim_{StimID}_repeat_{repeat_number}")
        if not os.path.exists(chosen_dir):
            raise FileNotFoundError(f"Requested repeat {repeat_number} not found for StimID {StimID}")
    else:
        chosen_dir = os.path.join(input_dir, random.choice(sub_dirs))

    spike_file = os.path.join(chosen_dir, "spikes.csv")
    if not os.path.exists(spike_file):
        raise FileNotFoundError(f"'spikes.csv' not found in {chosen_dir}")

    # ------------------------------------------------------------
    # 2. Build auditory-input spike tensor
    # ------------------------------------------------------------
    spike_df = pd.read_csv(spike_file, sep=r"\s+")

    aud_in_spikes = np.zeros((batch_size, seq_len, n_input), dtype=np.int8)

    interval_start = t_start
    interval_end   = t_start + seq_len

    window_df = spike_df.loc[
        (spike_df["timestamps"] >= interval_start) &
        (spike_df["timestamps"] <  interval_end)
    ]

    for _, row in window_df.iterrows():
        t = int(row["timestamps"] - interval_start)
        n = int(row["node_ids"])
        if 0 <= n < n_input and 0 <= t < seq_len:
            aud_in_spikes[:, t, n] = 1      # broadcast to every batch element

    # ------------------------------------------------------------
    # 3. Build target-rate matrix for the *pooled* IDs
    # ------------------------------------------------------------
    target_rates_matrix = np.zeros((seq_len, N_neurons), dtype=np.float32)

    for col, cell_id in enumerate(target_cell_ids):
        rate_path = os.path.join(target_rates_dir, f"{cell_id}__{StimName}.csv")
        if not os.path.exists(rate_path):
            raise FileNotFoundError(f"Missing rate file: {rate_path}")

        rate_full = pd.read_csv(rate_path)["Rate"].values
        window    = rate_full[int(t_start): int(t_start) + seq_len]

        if len(window) < seq_len:           # pad if shorter
            window = np.pad(window, (0, seq_len - len(window)), constant_values=0)

        target_rates_matrix[:, col] = window

    # add batch axis
    target_rates = np.expand_dims(target_rates_matrix, axis=0)   # (1, T, N)
    if batch_size > 1:
        target_rates = np.repeat(target_rates, batch_size, axis=0)

    return aud_in_spikes, target_rates, chosen_dir



# modified above to use 'target_cell_id' in STG network to match STG nodes and target rates
def load_aud_input_and_target_rates_ORIG(
    StimID, input_dir, target_rates_dir, StimID_to_StimName, seq_len, n_input, batch_size, t_start, N_neurons, repeat_number=None
):
    """
    Loads auditory input spikes and corresponding target firing rates for a given StimID.

    Parameters:
    - StimID (int): The stimulus ID used to identify the spikes file and target rates.
    - input_dir (str): Base directory containing the auditory input spikes.
    - target_rates_dir (str): Directory containing the target rate CSV files.
    - StimID_to_StimName (dict): Mapping of stimulus numbers to target identifiers.
    - seq_len (int): Sequence length (in ms).
    - n_input (int): Number of auditory input neurons.
    - batch_size (int): Batch size for data loading.
    - t_start (float): Start time for the sequence (in ms).
    - N_neurons (int): Number of model neurons to align target rates with.
    - repeat_number (int, optional): Specific repeat number to select. If None, a random repeat is chosen.

    Returns:
    - aud_in_spikes (np.ndarray): Auditory input spikes [batch_size, seq_len, n_input].
    - target_rates (np.ndarray): Target firing rates sorted and aligned with N_neurons [batch_size, seq_len, N_neurons].
    - chosen_dir (str): Directory of the selected stimulus for debugging purposes.
    """
    # Map StimID to StimName
    if StimID not in StimID_to_StimName:
        raise ValueError(f"StimID {StimID} not found in StimID_to_StimName mapping.")
    StimName = StimID_to_StimName[StimID]

    # Find matching directories for the given StimID
    sub_dirs = [
        d for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d)) and d.startswith(f"stim_{StimID}_repeat_")
    ]
    if not sub_dirs:
        raise ValueError(f"No directories found for StimID {StimID} in {input_dir}.")

    # Select a directory based on the repeat_number
    if repeat_number is not None:
        chosen_dir = os.path.join(input_dir, f"stim_{StimID}_repeat_{repeat_number}")
        if not os.path.exists(chosen_dir):
            raise ValueError(f"Directory for StimID {StimID} with repeat {repeat_number} not found.")
    else:
        chosen_dir = os.path.join(input_dir, random.choice(sub_dirs))

    spike_file = os.path.join(chosen_dir, 'spikes.csv')
    if not os.path.exists(spike_file):
        raise FileNotFoundError(f"'spikes.csv' not found in folder: {chosen_dir}")

    # Load auditory input spikes
    spike_data = pd.read_csv(spike_file, sep='\s+')

    aud_in_spikes = np.zeros((batch_size, seq_len, n_input), dtype=int)
    #t_start is 1 
    for batch_idx in range(batch_size):
        interval_start = t_start
        interval_end = t_start + seq_len
        #selected spikes go from 1ms to 3300ms
        # assumption is that sentence stim starts at 500ms
        selected_spikes = spike_data[(spike_data['timestamps'] >= interval_start) & (spike_data['timestamps'] < interval_end)]

        for _, row in selected_spikes.iterrows():
            time_idx = int(row['timestamps'] - interval_start)
            neuron_id = int(row['node_ids'])
            if 0 <= neuron_id < n_input and 0 <= time_idx < seq_len:
                aud_in_spikes[batch_idx, time_idx, neuron_id] = 1

    # Load and process target rates
    target_rate_files = [
        f for f in os.listdir(target_rates_dir)
        if f.endswith(".csv") and f.split("__")[1].startswith(StimName)
    ]

    if not target_rate_files:
        raise ValueError(f"No target rate files found for StimName {StimName} in {target_rates_dir}.")

    # Parse subject, neuron, and session numbers and sort files
    parsed_files = []
    for file in target_rate_files:
        try:
            subject, neuron_session = file.split("_", 1)
            neuron, stim = neuron_session.split("__", 1)
            if stim.rstrip(".csv") == StimName:
                parsed_files.append((int(subject), int(neuron), file))
        except ValueError:
            print(f"Skipping file with invalid format: {file}")

    # Sort by subject# (ascending) and then by neuron# (ascending)
    parsed_files.sort(key=lambda x: (x[0], x[1]))

    # Extract session and neuron numbers
    sessions = [entry[0] for entry in parsed_files]
    neurons = [entry[1] for entry in parsed_files]

    # Load and concatenate target rates
    target_rate_arrays = []
    for _, _, file in parsed_files:
        rate_data = pd.read_csv(os.path.join(target_rates_dir, file))
        rate_data = rate_data["Rate"].values
        start_idx = int(t_start)
        end_idx = start_idx + seq_len
        rate_data = rate_data[start_idx:end_idx]
        
        if len(rate_data) < seq_len:
            rate_data = np.pad(rate_data, (0, seq_len - len(rate_data)), constant_values=0)

        target_rate_arrays.append(rate_data)

    # Stack rates into a single array (seq_len, num_neurons)
    target_rates_full = np.column_stack(target_rate_arrays)

    # Expand to include batch_size as the first dimension
    target_rates = np.expand_dims(target_rates_full, axis=0)  # [1, seq_len, num_neurons]

    # Repeat target rates along batch dimension if batch_size > 1
    if batch_size > 1:
        target_rates = np.repeat(target_rates, batch_size, axis=0)

    # Ensure the final number of neurons matches N_neurons
    if target_rates.shape[2] > N_neurons:
        target_rates = target_rates[:, :, :N_neurons]

    # Expand target rates in the neuron dimension by repeating each neuron 10 times
    target_rates = np.repeat(target_rates, repeats=10, axis=2)  # Expands last axis (N_neurons)

    return aud_in_spikes, target_rates, chosen_dir
# def gaussian_smoothing(spikes, sigma=100):
#     """
#     Smooth binary spike trains using a Gaussian kernel along the time dimension.
#     Args:
#         spikes (tf.Tensor): [batch_size, seq_length, n_neurons], binary spikes.
#         sigma (int): Standard deviation of the Gaussian kernel.

#     Returns:
#         tf.Tensor: Smoothed spike trains with the same shape as the input.
#     """
#     # Create Gaussian kernel
#     kernel_size = int(6 * sigma + 1)  # Kernel spans 3 sigmas on each side
#     x = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=tf.float32)
#     gauss_kernel = tf.exp(-0.5 * (x / sigma) ** 2)
#     gauss_kernel /= tf.reduce_sum(gauss_kernel)  # Normalize the kernel

#     # Ensure spikes and kernel compatibility
#     spikes_shape = tf.shape(spikes)
#     gauss_kernel = tf.reshape(gauss_kernel, [-1, 1])  # Shape: [kernel_size, 1]

#     # Pad the spikes to handle edges
#     pad_size = kernel_size // 2
#     spikes_padded = tf.pad(spikes, [[0, 0], [pad_size, pad_size], [0, 0]], mode="CONSTANT")

#     # Smooth by sliding the kernel along the time dimension
#     smoothed_spikes = tf.nn.depthwise_conv2d(
#         input=tf.expand_dims(spikes_padded, axis=-1),  # Add channel dim: [batch, seq, n_neurons, 1]
#         filter=tf.reshape(gauss_kernel, [kernel_size, 1, 1, 1]),  # 4D kernel for depthwise conv
#         strides=[1, 1, 1, 1],
#         padding="VALID"
#     )
#     smoothed_spikes = tf.squeeze(smoothed_spikes, axis=-1)  # Remove channel dim: [batch, seq, n_neurons]

#     return smoothed_spikes

def gaussian_smoothing(spikes, sigma=100.0):
    """
    Apply 1-D Gaussian smoothing along time.
    Works for very short chunks by shrinking the kernel or skipping padding.

    spikes : tensor  [batch, T, N]  (float32/float16/bfloat16)
    sigma  : float   standard deviation in *samples* (not ms)

    Returns      : tensor  [batch, T, N]   same shape as input
    """
    # ── 0. Early-exit: no smoothing requested ────────────────────────────────
    if sigma <= 0:
        return spikes

    # ── 1. Build a kernel whose length never exceeds the signal ─────────────
    T = tf.shape(spikes)[1]                        # dynamic length of chunk
    kern_len = tf.minimum(T, tf.cast(6 * sigma + 1, tf.int32))
    # keep it odd
    kern_len = kern_len + 1 - (kern_len & 1)

    x = tf.cast(tf.range(kern_len) - kern_len // 2, tf.float32)
    kernel = tf.exp(-0.5 * (x / sigma) ** 2)
    kernel /= tf.reduce_sum(kernel)
    kernel = tf.reshape(kernel, [kern_len, 1, 1, 1])

    # ── 2. Decide how much, if any, padding we are allowed ───────────────────
    pad = (kern_len // 2)
    max_pad = tf.maximum(0, T - 1)                 # MirrorPad needs pad < T
    pad = tf.minimum(pad, max_pad)

    # Graph-mode compatible conditional convolution using tf.cond
    def conv_with_padding():
        padded_spikes = tf.pad(
            spikes,
            paddings=[[0, 0], [pad, pad], [0, 0]],
            mode="REFLECT"
        )
        return tf.nn.depthwise_conv2d(
            input=tf.expand_dims(padded_spikes, axis=-1),     # [B, T, N, 1]
            filter=kernel,                             # [K, 1, 1, 1]
            strides=[1, 1, 1, 1],
            padding="VALID"
        )
    
    def conv_no_padding():
        return tf.nn.depthwise_conv2d(
            input=tf.expand_dims(spikes, axis=-1),     # [B, T, N, 1]
            filter=kernel,                             # [K, 1, 1, 1]
            strides=[1, 1, 1, 1],
            padding="SAME"
        )
    
    # ── 3. Conditional convolution execution ────────────────────────────────
    out = tf.cond(
        pad > 0,
        true_fn=conv_with_padding,
        false_fn=conv_no_padding
    )
    return tf.squeeze(out, axis=-1)                # back to [B, T, N]

# def gaussian_smoothing(spikes, sigma=100):
#     kernel_size = int(6 * sigma + 1)
#     x = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=tf.float32)
#     gauss_kernel = tf.exp(-0.5 * (x / sigma) ** 2)
#     gauss_kernel /= tf.reduce_sum(gauss_kernel)

#     gauss_kernel = tf.reshape(gauss_kernel, [kernel_size, 1])
#     pad_size = kernel_size // 2

#     # Use REFLECT or SYMMETRIC instead of CONSTANT=0
#     spikes_padded = tf.pad(
#         spikes,
#         [[0, 0], [pad_size, pad_size], [0, 0]],
#         mode="REFLECT"
#     )

#     smoothed_spikes = tf.nn.depthwise_conv2d(
#         input=tf.expand_dims(spikes_padded, axis=-1),
#         filter=tf.reshape(gauss_kernel, [kernel_size, 1, 1, 1]),
#         strides=[1, 1, 1, 1],
#         padding="VALID"
#     )
#     smoothed_spikes = tf.squeeze(smoothed_spikes, axis=-1)

#     return smoothed_spikes

# def gaussian_smoothing(spikes, sigma=100):
#     """
#     Applies Gaussian smoothing using depthwise convolution, ensuring proper shape handling.

#     Parameters
#     ----------
#     spikes : tf.Tensor
#         Input spike train tensor of shape [batch, time, neurons].
#     sigma : int
#         Standard deviation of the Gaussian kernel.

#     Returns
#     -------
#     smoothed_spikes : tf.Tensor
#         Smoothed spike train tensor of shape [batch, time, neurons].
#     """

#     kernel_size = int(6 * sigma + 1)
#     x = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=tf.float32)
#     gauss_kernel = tf.exp(-0.5 * (x / sigma) ** 2)
#     gauss_kernel /= tf.reduce_sum(gauss_kernel)

#     gauss_kernel = tf.reshape(gauss_kernel, [kernel_size, 1])  # Ensure shape is correct
#     pad_size = kernel_size // 2

#     # Ensure spikes are at least rank 3 [batch, time, neurons]
#     if len(spikes.shape) == 2:  
#         spikes = tf.expand_dims(spikes, axis=0)  # Add batch dimension if missing

#     spikes = tf.ensure_shape(spikes, [None, None, None])

#     # Compute dynamic padding safely in graph mode
#     pad_size_tensor = tf.minimum(tf.shape(spikes)[1], pad_size)  # Ensure we don’t pad more than the time dimension
#     spikes_padded = tf.pad(
#         spikes,
#         [[0, 0], [pad_size_tensor, pad_size_tensor], [0, 0]],
#         mode="REFLECT"
#     )

#     # Depthwise convolution expects NHWC format, ensure correct shape
#     smoothed_spikes = tf.nn.depthwise_conv2d(
#         input=tf.expand_dims(spikes_padded, axis=-1),  # Add channel dimension
#         filter=tf.reshape(gauss_kernel, [kernel_size, 1, 1, 1]),
#         strides=[1, 1, 1, 1],
#         padding="VALID"
#     )

#     smoothed_spikes = tf.squeeze(smoothed_spikes, axis=-1)  # Remove extra channel dimension

#     return smoothed_spikes


def hierarchical_cluster_target_rates(target_rates_2d, zero_handling='put_zero_last'):
    """
    Performs hierarchical clustering on the columns of target_rates_2d
    (i.e., 'neurons'), based on their time-series profiles.
    
    Parameters
    ----------
    target_rates_2d : np.ndarray
        Shape (time, num_neurons). This should be your target rates
        for a single batch element (already stripped of batch dimension).
    zero_handling : str
        If 'put_zero_last', we move columns whose sum is 0 to the end.
        If None, we simply cluster all columns as is.

    Returns
    -------
    ordering : np.ndarray
        1D array of column indices in the desired clustered order.
    """
    # Shape check
    time_dim, num_neurons = target_rates_2d.shape

    # Sum across time to find "non-zero" columns
    col_sums = np.sum(target_rates_2d, axis=0)
    if zero_handling == 'put_zero_last':
        non_zero_cols = np.where(col_sums != 0)[0]
        zero_cols = np.where(col_sums == 0)[0]
        # Subset for clustering
        responses_to_cluster = target_rates_2d[:, non_zero_cols]

        # Perform hierarchical clustering using correlation or cosine, etc.
        # e.g. metric='cosine', method='average'
        Z = linkage(responses_to_cluster.T, method='average', metric='cosine')
        
        # Leaves_list gives the order of columns in the subset
        clustered_subset_order = leaves_list(Z)
        # Convert from subset to full column indices
        ordered_non_zero_cols = non_zero_cols[clustered_subset_order]
        # Put zero columns at the end
        ordering = np.concatenate([ordered_non_zero_cols, zero_cols])
    else:
        # Just cluster all columns
        Z = linkage(target_rates_2d.T, method='average', metric='cosine')
        ordering = leaves_list(Z)
    
    return ordering

# def plot_target_rates_and_model_spikes(
#     target_rates,
#     rollout_results,
#     epoch,
#     target_scaling=1.0,
#     sigma=150.0,
#     perform_clustering=True,  
#     neuron_order=None,        
#     z_scored=False,
#     display=False,
#     save=True,
#     save_dir='model_checkpoints',
#     file_string='target_rates_and_model_spikes'
# ):
#     """
#     Plots:
#       1) Target firing rates (NO baseline subtraction, log-transformed).
#       2) Target firing rates (log-transformed & baseline-corrected).
#       3) Smoothed model spike rates (log-transformed & baseline-corrected).
#       4) Model spike raster.

#     Parameters
#     ----------
#     target_rates : np.ndarray or tf.Tensor
#         Shape [1, seq_len, num_target_cells]. (Converted to Hz internally.)
#     rollout_results : dict
#         Must contain 'spikes' with shape [1, seq_len, n_neurons].
#     epoch : int
#         Current epoch number (for file naming).
#     perform_clustering : bool
#         If True, do hierarchical clustering on target rates across neurons.
#         If neuron_order is also provided, that order takes precedence.
#     neuron_order : np.ndarray or None
#         If given, reorder target_rates and model_spikes columns. Must match #neurons.
#     z_scored : bool
#         If True, z-score the data and use a diverging colormap.
#     display : bool
#         If True, display the figure.
#     save : bool
#         If True, saves the plot in save_dir.
#     save_dir : str
#         Directory to save the figure (if save=True).
#     file_string : str
#         Base filename for saving.

#     Returns
#     -------
#     None
#     """

#     # 1) Convert target rates to Hz
#     target_rates = target_scaling*target_rates * 1000.0
#     target_rates =gaussian_smoothing(target_rates, sigma=sigma)

#     # 2) Ensure inputs are numpy arrays
#     if isinstance(target_rates, tf.Tensor):
#         target_rates = target_rates.numpy()
#     if isinstance(rollout_results['spikes'], tf.Tensor):
#         model_spikes = rollout_results['spikes'].numpy()
#     else:
#         model_spikes = rollout_results['spikes']

#     # 3) Remove batch dimension
#     # target_rates => [seq_len, num_target_cells]
#     # model_spikes => [seq_len, n_neurons]
#     target_rates = target_rates[0]  # shape (seq_len, num_target_cells)
#     model_spikes = model_spikes[0]  # shape (seq_len, n_neurons)

#     # 4) Optionally perform hierarchical clustering (unless neuron_order is given)
#     if perform_clustering and (neuron_order is None):
#         from scipy.cluster.hierarchy import leaves_list, linkage
#         # External helper to get ordering
#         ordering = hierarchical_cluster_target_rates(target_rates)
#     else:
#         ordering = neuron_order

#     # 5) Apply ordering if we have one
#     if ordering is not None:
#         target_rates = target_rates[:, ordering]
#         model_spikes = model_spikes[:, ordering]

#     # 6) We'll keep a copy of target_rates BEFORE baseline subtraction
#     small_constant = 0.1
#     # Log transform only (no baseline)
#     target_rates_no_sub = np.log(target_rates + small_constant)

#     # 7) Smooth model spikes => smoothed_model_rates
#     model_spikes_3d = np.expand_dims(model_spikes, axis=0)
#     smoothed_model_rates = gaussian_smoothing(
#         tf.convert_to_tensor(model_spikes_3d, dtype=tf.float32), sigma=sigma
#     )
#     smoothed_model_rates = smoothed_model_rates.numpy()[0] * 1000.0  # shape => (seq_len, n_neurons)
#     smoothed_model_rates = np.log(smoothed_model_rates + small_constant)

#     # 8) Baseline subtraction for the "target_rates" (the usual approach)
#     target_rates = np.log(target_rates + small_constant)
#     mean_500ms_target = np.mean(target_rates[:500], axis=0, keepdims=True)
#     mean_500ms_smoothed = np.mean(smoothed_model_rates[:500], axis=0, keepdims=True)
#     target_rates -= mean_500ms_target
#     smoothed_model_rates -= mean_500ms_smoothed

#     # 9) Optional z-scoring
#     if z_scored:
#         target_rates_mean = np.mean(target_rates, axis=0)
#         target_rates_std = np.std(target_rates, axis=0) + 1e-8
#         target_rates = (target_rates - target_rates_mean) / target_rates_std

#         smoothed_model_rates_mean = np.mean(smoothed_model_rates, axis=0)
#         smoothed_model_rates_std = np.std(smoothed_model_rates, axis=0) + 1e-8
#         smoothed_model_rates = ((smoothed_model_rates - smoothed_model_rates_mean)
#                                 / smoothed_model_rates_std)

#         cmap = "bwr"
#         vmin, vmax = -5, 5
#     else:
#         cmap = "Greys"
#         vmin, vmax = -1, 5

#     # 10) Prepare for plotting
#     # Transpose for plotting (neurons on Y-axis, time on X-axis)
#     # shapes => [num_neurons, seq_len]
#     target_rates = target_rates.T
#     target_rates_no_sub = target_rates_no_sub.T  # <--- no baseline sub
#     smoothed_model_rates = smoothed_model_rates.T
#     model_spikes = model_spikes.T
#     spike_neurons, spike_times = np.nonzero(model_spikes)

#     # 11) Create the figure and subplots
#     fig, axes = plt.subplots(1, 4, figsize=(26, 10), constrained_layout=False)

#     # -- (a) Subplot 1: Original target rates (log + no baseline)
#     # no explicit vmin/vmax => auto-scale
#     im0 = axes[0].imshow(
#         target_rates_no_sub,
#         aspect="auto",
#         cmap=cmap,
#         origin="lower",
#         extent=[0, target_rates_no_sub.shape[1], 0, target_rates_no_sub.shape[0]],
#     )
#     axes[0].set_title("Target Rates (Log Only, No Baseline)")
#     axes[0].set_xlabel("Time (ms)")
#     axes[0].set_ylabel("Neuron Index")
#     fig.colorbar(im0, ax=axes[0], label="Log Rate (Hz)")

#     # -- (b) Subplot 2: Baseline-subtracted target rates
#     im1 = axes[1].imshow(
#         target_rates,
#         aspect="auto",
#         cmap=cmap,
#         origin="lower",
#         vmin=vmin,
#         vmax=vmax,
#         extent=[0, target_rates.shape[1], 0, target_rates.shape[0]],
#     )
#     axes[1].set_title("Target Rates (Log + Baseline Sub)")
#     axes[1].set_xlabel("Time (ms)")
#     axes[1].set_ylabel("Neuron Index")
#     fig.colorbar(im1, ax=axes[1], label="Z-Score" if z_scored else "Log Rate (Hz)")

#     # -- (c) Subplot 3: Smoothed model rates
#     im2 = axes[2].imshow(
#         smoothed_model_rates,
#         aspect="auto",
#         cmap=cmap,
#         origin="lower",
#         vmin=vmin,
#         vmax=vmax,
#         extent=[0, smoothed_model_rates.shape[1], 0, smoothed_model_rates.shape[0]],
#     )
#     axes[2].set_title("Smoothed Model Rates\n(Log + Baseline Sub)")
#     axes[2].set_xlabel("Time (ms)")
#     axes[2].set_ylabel("Neuron Index")
#     fig.colorbar(im2, ax=axes[2], label="Z-Score" if z_scored else "Log Rate (Hz)")

#     # -- (d) Subplot 4: Model spikes raster
#     axes[3].scatter(spike_times, spike_neurons, c="black", s=1.5, alpha=0.5)
#     axes[3].set_title("Model Spikes Raster")
#     axes[3].set_xlim(0, model_spikes.shape[1])
#     axes[3].set_ylim(0, model_spikes.shape[0])
#     axes[3].set_xlabel("Time (ms)")
#     axes[3].set_ylabel("Neuron Index")

#     # 12) Optionally save
#     if save:
#         os.makedirs(save_dir, exist_ok=True)
#         save_path = os.path.join(save_dir, f"{file_string}_{epoch}.png")
#         plt.savefig(save_path)
#         print(f"Plot saved to: {save_path}")

#     # 13) Optionally display
#     if display:
#         plt.show()
#     else:
#         plt.close(fig)

# def plot_target_rates_and_model_spikes(
#     target_rates,
#     rollout_results,
#     epoch,
#     target_scaling=1.0,
#     sigma=150.0,
#     perform_clustering=True,
#     neuron_order=None,
#     z_scored=False,
#     display=False,
#     save=True,
#     save_dir='model_checkpoints',
#     file_string='target_rates_and_model_spikes'
# ):
#     """
#     Plots (in 3 subplots):
#       1) Target firing rates, log-transformed (no baseline subtraction).
#       2) Smoothed model spike rates, log-transformed (no baseline subtraction).
#       3) Model spike raster.

#     Parameters
#     ----------
#     target_rates : np.ndarray or tf.Tensor
#         Shape [1, seq_len, num_target_cells]. (Converted to Hz internally.)
#     rollout_results : dict
#         Must contain 'spikes' with shape [1, seq_len, n_neurons].
#     epoch : int
#         Current epoch number (for naming).
#     target_scaling : float
#         A scaling factor to multiply the target rates before converting to Hz.
#     sigma : float
#         Gaussian smoothing window for both target and model spikes.
#     perform_clustering : bool
#         If True, do hierarchical clustering on target rates across neurons.
#         If neuron_order is also provided, that order takes precedence.
#     neuron_order : np.ndarray or None
#         If given, reorder columns (neurons) for target_rates & model_spikes.
#     z_scored : bool
#         If True, z-score the data and use a diverging colormap.
#     display : bool
#         If True, display the figure.
#     save : bool
#         If True, save the figure to 'save_dir'.
#     save_dir : str
#         Directory to save the figure (if save=True).
#     file_string : str
#         Base filename for saving.
#     """

#     # 1) Convert target rates to Hz and smooth
#     target_rates = target_scaling * target_rates * 1000.0
#     target_rates = gaussian_smoothing(target_rates, sigma=sigma)

#     # 2) Ensure inputs are numpy arrays
#     if isinstance(target_rates, tf.Tensor):
#         target_rates = target_rates.numpy()
#     if isinstance(rollout_results['spikes'], tf.Tensor):
#         model_spikes = rollout_results['spikes'].numpy()
#     else:
#         model_spikes = rollout_results['spikes']

#     # 3) Remove batch dimension
#     target_rates = target_rates[0]  # shape => [seq_len, num_target_cells]
#     model_spikes = model_spikes[0]  # shape => [seq_len, n_neurons]
    
#     # ✅ Add small random noise to `target_rates` to prevent clustering issues
#     np.random.seed(42)  # Fixed seed for reproducibility
#     noise = np.random.uniform(0.0, 0.1, size=target_rates.shape)  # Small positive noise
#     target_rates += noise  # Only applied to target_rates (not model spikes)
#         # 4) Optionally perform hierarchical clustering unless we have neuron_order
#     if perform_clustering and (neuron_order is None):
#         from scipy.cluster.hierarchy import leaves_list, linkage

#         ordering = hierarchical_cluster_target_rates(target_rates)
#     else:
#         ordering = neuron_order

#     # 5) Apply ordering if we have one
#     if ordering is not None:
#         target_rates = target_rates[:, ordering]
#         model_spikes = model_spikes[:, ordering]

#     # 6) Create a log-transformed version of the target rates (no baseline sub)
#     small_constant = 0.01
#     target_rates_no_sub = np.log(target_rates + small_constant)  # shape => [seq_len, num_neurons]

#     # 7) Smooth model spikes => smoothed_model_rates (log only, no baseline sub)
#     model_spikes_3d = np.expand_dims(model_spikes, axis=0)  # [1, seq_len, n_neurons]
#     smoothed_model_rates = gaussian_smoothing(
#         tf.convert_to_tensor(model_spikes_3d, dtype=tf.float32), sigma=sigma
#     ).numpy()[0] * 1000.0  # convert to Hz
#     smoothed_model_rates_no_sub = np.log(smoothed_model_rates + small_constant)

#     # 8) If z_scored, do it here
#     if z_scored:
#         # z-score target
#         t_mean = np.mean(target_rates_no_sub, axis=0)
#         t_std = np.std(target_rates_no_sub, axis=0) + 1e-8
#         target_rates_no_sub = (target_rates_no_sub - t_mean) / t_std

#         # z-score model
#         m_mean = np.mean(smoothed_model_rates_no_sub, axis=0)
#         m_std = np.std(smoothed_model_rates_no_sub, axis=0) + 1e-8
#         smoothed_model_rates_no_sub = (smoothed_model_rates_no_sub - m_mean) / m_std

#         cmap = "bwr"
#         vmin, vmax = -5, 5
#     else:
#         cmap = "Greys"
#         vmin, vmax = -2, 5

#     # 9) Transpose data for plotting (neurons on Y-axis, time on X-axis)
#     # shapes => [num_neurons, seq_len]
#     target_rates_no_sub = target_rates_no_sub.T
#     smoothed_model_rates_no_sub = smoothed_model_rates_no_sub.T
#     model_spikes = model_spikes.T  # [n_neurons, seq_len]
#     spike_neurons, spike_times = np.nonzero(model_spikes)

#     # 10) Create figure with three subplots
#     fig, axes = plt.subplots(1, 3, figsize=(22, 10))

#     # (a) Subplot 1: Target rates (log only, no baseline)
#     im0 = axes[0].imshow(
#         target_rates_no_sub,
#         aspect="auto",
#         cmap=cmap,
#         origin="lower",
#         vmin=vmin,  # auto-scale
#         vmax=vmax,  # auto-scale
#         extent=[0, target_rates_no_sub.shape[1], 0, target_rates_no_sub.shape[0]],
#     )
#     axes[0].set_title("Target Rates\n(Log Only, No Baseline)")
#     axes[0].set_xlabel("Time (ms)")
#     axes[0].set_ylabel("Neuron Index")
#     fig.colorbar(im0, ax=axes[0], label="Log(Hz)")

#     # (b) Subplot 2: Smoothed model rates (log only, no baseline)
#     im1 = axes[1].imshow(
#         smoothed_model_rates_no_sub,
#         aspect="auto",
#         cmap=cmap,
#         origin="lower",
#         vmin=vmin,
#         vmax=vmax,
#         extent=[0, smoothed_model_rates_no_sub.shape[1], 0, smoothed_model_rates_no_sub.shape[0]],
#     )
#     axes[1].set_title("Smoothed Model Rates\n(Log Only, No Baseline)")
#     axes[1].set_xlabel("Time (ms)")
#     axes[1].set_ylabel("Neuron Index")
#     fig.colorbar(im1, ax=axes[1], label="Z-Score" if z_scored else "Log(Hz)")

#     # (c) Subplot 3: Model spikes raster
#     axes[2].scatter(spike_times, spike_neurons, c="black", s=0.35, alpha=0.25)
#     axes[2].set_title("Model Spikes Raster")
#     axes[2].set_xlim(0, model_spikes.shape[1])
#     axes[2].set_ylim(0, model_spikes.shape[0])
#     axes[2].set_xlabel("Time (ms)")
#     axes[2].set_ylabel("Neuron Index")

#     # Optionally save
#     if save:
#         os.makedirs(save_dir, exist_ok=True)
#         save_path = os.path.join(save_dir, f"{file_string}_{epoch}.png")
#         plt.savefig(save_path)
#         print(f"Plot saved to: {save_path}")

#     # Optionally display
#     if display:
#         plt.show()
#     else:
#         plt.close(fig)


def plot_target_rates_and_model_spikes(
    target_rates,
    rollout_results,
    epoch,
    *,
    target_scaling=1.0,
    sigma=100.0,
    perform_clustering=True,
    neuron_order=None,
    z_scored=False,
    display=False,
    save=True,
    save_dir='model_checkpoints',
    file_string='target_rates_and_model_spikes',
    stim_onset=500,
    x_min=250,
    x_max=2500,
    POOL_MAT=None,
    UNPOOL_MAP=None,
):
    """
    Draw three aligned panels
        • left   – smoothed target rates  (log-scale)
        • middle – smoothed model rates  (ensemble–averaged, then broadcast)
        • right  – spike raster (time on x-axis, neuron index on y-axis)
    """
    import numpy as np, os, matplotlib.pyplot as plt, tensorflow as tf

    # ───── 1. target rates: smooth & scale ────────────────────────────
    target_rates = tf.convert_to_tensor(target_rates, tf.float32)
    if target_rates.ndim == 2:
        target_rates = target_rates[None, ...]                     # (1,T,N)
    target_rates = gaussian_smoothing(
        target_rates * target_scaling * 1e3, sigma=sigma).numpy()[0]   # (T,N)

    # tiny noise for visual separation
    rng = np.random.default_rng(42)
    target_rates += rng.uniform(0.0, 0.1, target_rates.shape)

    # ───── 2. model spikes  (T,N)  ────────────────────────────────────
    model_spikes = rollout_results["spikes"]
    if isinstance(model_spikes, tf.Tensor):
        model_spikes = model_spikes.numpy()
    model_spikes = model_spikes[0]                                 # (T,N)

    # ───── 3. smoothed model rates ───────────────────────────────────
    smoothed_model = gaussian_smoothing(
        model_spikes[None, ...], sigma=sigma).numpy()[0] * 1e3      # (T,N)

    # Apply pooling only if both matrices are provided
    if POOL_MAT is not None and UNPOOL_MAP is not None:
        pooled = np.matmul(smoothed_model, POOL_MAT.numpy())        # (T,n_targets)
        smoothed_model = pooled[:, UNPOOL_MAP.numpy()]              # (T,N)
    # If pooling is disabled, smoothed_model remains as-is for direct comparison

    # ───── 4. optional clustering / fixed order ──────────────────────
    if perform_clustering and neuron_order is None:
        order = hierarchical_cluster_target_rates(target_rates)
    else:
        order = neuron_order

    if order is not None:
        idx = tf.convert_to_tensor(order, tf.int32)
        target_rates   = tf.gather(target_rates,   idx, axis=1).numpy()
        smoothed_model = tf.gather(smoothed_model, idx, axis=1).numpy()
        model_spikes   = tf.gather(model_spikes,   idx, axis=1).numpy()

    # ───── 5. log / z-score transforms ───────────────────────────────
    eps = 1e-2
    tgt_log = np.log(target_rates + eps)
    mdl_log = np.log(smoothed_model + eps)

    if z_scored:
        tgt_log = (tgt_log - tgt_log.mean(0)) / (tgt_log.std(0) + 1e-8)
        mdl_log = (mdl_log - mdl_log.mean(0)) / (mdl_log.std(0) + 1e-8)
        cmap, vmin, vmax = "bwr", -5, 5
    else:
        cmap, vmin, vmax = "Greys", -3, 5

    # ───── 6. raster coordinates  (correct time on x!) ───────────────
    spike_times, spike_neurons = np.nonzero(model_spikes)           # swap order
    shifted = spike_times - stim_onset                              # time axis
    time_axis = np.arange(model_spikes.shape[0]) - stim_onset
    extent = [time_axis[0], time_axis[-1], 0, model_spikes.shape[1]]

    # ───── 7. plotting ───────────────────────────────────────────────
    fig, ax = plt.subplots(1, 3, figsize=(22, 10))

    ax[0].imshow(tgt_log.T, aspect="auto", origin="lower",
                 cmap=cmap, vmin=vmin, vmax=vmax, extent=extent)
    ax[0].set_title("Target Rates (log)")

    ax[1].imshow(mdl_log.T, aspect="auto", origin="lower",
                 cmap=cmap, vmin=vmin, vmax=vmax, extent=extent)
    ax[1].set_title("Smoothed Model Rates (log)")

    ax[2].scatter(shifted, spike_neurons, s=0.45, c="black", alpha=0.25)
    ax[2].set_title("Model Spikes Raster")
    ax[2].set_ylim(0, model_spikes.shape[1])

    for a in ax:
        a.set_xlabel("Time (ms, rel. to stim onset)")
        if x_min is not None and x_max is not None:
            a.set_xlim(x_min - stim_onset, x_max - stim_onset)
    ax[0].set_ylabel("Neuron index")

    # ───── 8. save / show ────────────────────────────────────────────
    if save:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{file_string}_{epoch}.png")
        plt.savefig(path)
        print(f"Plot saved → {path}")
    if display:
        plt.show()
    else:
        plt.close(fig)




def smart_cluster(target_rates_TN, unpool_map):
    """
    Cluster only the 685 unique target traces, then expand the order so
    every duplicate model neuron follows its prototype.

    Parameters
    ----------
    target_rates_TN : (T,N) np.ndarray
        Full matrix after un-pooling (many duplicate columns).
    unpool_map : (N,) 1-D array
        Maps every model neuron index → target-ID (0 … 684).

    Returns
    -------
    order_full : list[int]
        Permutation of length N for plotting.
    """
    # 1. cluster the 685 prototypes
    uniq_ids = np.unique(unpool_map)
    prototypes = target_rates_TN[:, (unpool_map == np.arange(len(uniq_ids))).argmax(0)]
    proto_order = hierarchical_cluster_target_rates(prototypes)  # len = 685

    # 2. expand: for each target-ID in that order, append *all* its duplicates
    buckets = {tid: np.where(unpool_map == tid)[0] for tid in uniq_ids}
    order_full = np.concatenate([buckets[tid] for tid in proto_order]).tolist()
    return order_full

# ─── smart_cluster helper ──────────────────────────────────────────────
def build_full_cluster_order(target_rates_pool, unpool_map):
    """
    Parameters
    ----------
    target_rates_pool : (T, n_targets) np.ndarray
        One trace per *unique* target cell (685 in your data).
    unpool_map        : (N,) np.ndarray
        For every model neuron (N = 40 052) tells which target-ID it copies.

    Returns
    -------
    order_full : list[int]   length = N
        The y-axis permutation for plotting.  All duplicates of a target
        follow their prototype, preserving the dendrogram structure.
    """
    # 1. cluster only the 685 prototypes  →  permutation of target IDs
    proto_order = hierarchical_cluster_target_rates(target_rates_pool)

    # 2. expand that order to all duplicates
    buckets = {tid: np.where(unpool_map == tid)[0] for tid in range(len(proto_order))}
    order_full = np.concatenate([buckets[tid] for tid in proto_order]).tolist()
    return order_full


def plot_target_rates_and_model_spikes_notebook(
    target_rates,
    rollout_results,
    epoch,
    target_scaling=1.0,
    sigma=100.0,
    perform_clustering=True,
    neuron_order=None,
    z_scored=False,
    display=False,
    save=True,
    save_dir='model_checkpoints',
    file_string='target_rates_and_model_spikes',
    stim_onset=500,
    x_min=250,
    x_max=2500,
):
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    import tensorflow as tf

    # print("🔍 Starting plot_target_rates_and_model_spikes")
    # print("target_rates initial shape:", target_rates.shape)
    # print("rollout_results['spikes'] type:", type(rollout_results['spikes']), 
    #       "shape:", rollout_results['spikes'].shape)

    # 1. Convert target rates to Hz and smooth
    target_rates = target_scaling * target_rates * 1000.0
    if len(target_rates.shape) == 2:
        target_rates = tf.expand_dims(target_rates, axis=0)

    #print("target_rates shape before smoothing:", target_rates.shape)
    target_rates = gaussian_smoothing(target_rates, sigma=sigma)
    #print("target_rates shape after smoothing:", target_rates.shape)

    # 2. Ensure inputs are numpy arrays
    if isinstance(target_rates, tf.Tensor):
        target_rates = target_rates.numpy()
    if isinstance(rollout_results['spikes'], tf.Tensor):
        model_spikes = rollout_results['spikes'].numpy()
    else:
        model_spikes = rollout_results['spikes']
    #print("model_spikes shape (after conversion):", model_spikes.shape)

    # 3. Remove batch dimension
    target_rates = target_rates[0]
    model_spikes = model_spikes[0]
    #print("target_rates shape after squeeze:", target_rates.shape)
    #print("model_spikes shape after squeeze:", model_spikes.shape)

    # 4. Add small noise to target rates
    np.random.seed(42)
    target_rates += np.random.uniform(0.0, 0.1, size=target_rates.shape)

    # 5. Determine neuron ordering
    if perform_clustering and (neuron_order is None):
        ordering = hierarchical_cluster_target_rates(target_rates)
    else:
        ordering = neuron_order

    # 6. Apply neuron ordering if available
    if ordering is not None:
        print("Applying neuron ordering")
        ordering_tf = tf.convert_to_tensor(ordering, dtype=tf.int32)
        target_rates = tf.gather(target_rates, ordering_tf, axis=1)
        model_spikes = tf.gather(model_spikes, ordering_tf, axis=1)
        #print("Shapes after ordering:", "target_rates", target_rates.shape, 
        #      "model_spikes", model_spikes.shape)

    # 7. Log transform
    small_constant = 0.01
    target_rates_no_sub = np.log(target_rates + small_constant)

    # 8. Smooth model spikes and log transform
    #print("model_spikes shape before expand_dims:", model_spikes.shape)
    #model_spikes_3d = np.expand_dims(model_spikes, axis=0)
    #print("model_spikes_3d shape:", model_spikes_3d.shape)
    
    # Ensure model_spikes has shape [1, T, N]
    model_spikes_tensor = tf.convert_to_tensor(model_spikes, dtype=tf.float32)
    if len(model_spikes_tensor.shape) == 2:
        model_spikes_tensor = tf.expand_dims(model_spikes_tensor, axis=0)
    
    smoothed_model_rates = gaussian_smoothing(
        model_spikes_tensor, sigma=sigma
    ).numpy()[0] * 1000.0

    # ─── NEW: ensemble-average, then broadcast back ───────────────
    smoothed_pool = np.matmul(smoothed_model_rates, POOL_MAT.numpy())     # (T, n_targets)
    smoothed_model_rates = smoothed_pool[:, UNPOOL_MAP.numpy()]           # (T, N)
    # ───────────────────────────────────────────────────────────────

    smoothed_model_rates_no_sub = np.log(smoothed_model_rates + small_constant)
    #print("smoothed_model_rates shape:", smoothed_model_rates.shape)

    # 9. Optionally z-score
    if z_scored:
        target_rates_no_sub = (target_rates_no_sub - np.mean(target_rates_no_sub, axis=0)) / (
            np.std(target_rates_no_sub, axis=0) + 1e-8
        )
        smoothed_model_rates_no_sub = (smoothed_model_rates_no_sub - np.mean(smoothed_model_rates_no_sub, axis=0)) / (
            np.std(smoothed_model_rates_no_sub, axis=0) + 1e-8
        )
        cmap = "bwr"
        vmin, vmax = -5, 5
    else:
        cmap = "Greys"
        vmin, vmax = -2, 5

    # 10. Transpose for plotting
    target_rates_no_sub = tf.transpose(target_rates_no_sub)
    smoothed_model_rates_no_sub = tf.transpose(smoothed_model_rates_no_sub)
    model_spikes = tf.transpose(model_spikes)
    #spike_neurons, spike_times = np.nonzero(model_spikes)

    # 11. Remove batch dimension if necessary
    if model_spikes.ndim == 3:
        model_spikes = model_spikes[0]  # or tf.squeeze(model_spikes, axis=0)

    # 12. Find nonzero spikes
    spike_neurons, spike_times = np.nonzero(model_spikes)

    # Time axis
    time_axis = np.arange(model_spikes.shape[1]) - stim_onset
    extent = [time_axis[0], time_axis[-1], 0, model_spikes.shape[0]]

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(22, 10))

    axes[0].imshow(target_rates_no_sub, aspect="auto", cmap=cmap, origin="lower", vmin=vmin, vmax=vmax, extent=extent)
    axes[0].set_title("Target Rates\n(Log Only, No Baseline)")
    axes[0].set_xlabel("Time (ms, rel to stim onset)")
    axes[0].set_ylabel("Neuron Index")

    axes[1].imshow(smoothed_model_rates_no_sub, aspect="auto", cmap=cmap, origin="lower", vmin=vmin, vmax=vmax, extent=extent)
    axes[1].set_title("Smoothed Model Rates\n(Log Only, No Baseline)")
    axes[1].set_xlabel("Time (ms, rel to stim onset)")
    axes[1].set_ylabel("Neuron Index")

    shifted_spike_times = spike_times - stim_onset
    axes[2].scatter(shifted_spike_times, spike_neurons, c="black", s=2.65, alpha=0.45)
    axes[2].set_title("Model Spikes Raster")
    axes[2].set_xlabel("Time (ms, rel to stim onset)")
    axes[2].set_ylabel("Neuron Index")
    axes[2].set_ylim(0, model_spikes.shape[0])

    if x_min is not None and x_max is not None:
        for ax in axes:
            ax.set_xlim(x_min - stim_onset, x_max - stim_onset)

    if save:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{file_string}_{epoch}.png")
        plt.savefig(save_path)
        print(f"Plot saved to: {save_path}")

    if display:
        plt.show()
    else:
        plt.close(fig)


def plot_target_rates_and_model_spikes_old(
    target_rates,
    rollout_results,
    epoch,
    reordering=None,
    z_scored=False,
    display=True,
    save=False,
    save_dir='model_checkpoints',
    file_string='target_rates_and_model_spikes'
):
    """
    Plots target firing rates, smoothed model spike rates, and model spikes raster,
    with optional reordering of the neurons and z-scored plotting. Allows optional
    saving and/or display of the plot, using an epoch-based naming scheme.

    Parameters:
    - target_rates: np.ndarray or tf.Tensor of shape [1, seq_len, num_target_cells].
    - rollout_results: Dictionary containing 'spikes' as a tensor of shape [1, seq_len, n_neurons].
    - epoch: Current epoch number (for naming and reference).
    - reordering: Optional np.ndarray. If provided, reorders the target rates and model spikes.
    - z_scored (bool): If True, z-score the data and use a blue-white-red diverging colormap.
    - display (bool): If True, displays the plot.
    - save (bool): If True, saves the plot using an epoch-based filename in save_dir.
    - save_dir (str): Directory to save the figure (if save=True).
    """

    # Convert target rates to Hz
    target_rates = target_rates * 1000

    # Ensure inputs are numpy arrays
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
    if ordering is not None:
        ordering_tf = tf.convert_to_tensor(ordering, dtype=tf.int32)
        target_rates = tf.gather(target_rates, ordering_tf, axis=1)
        model_spikes = tf.gather(model_spikes, ordering_tf, axis=1)


    # Smooth model spikes to compute smoothed spike rates
    model_spikes_3d = np.expand_dims(model_spikes, axis=0)  # Add batch dimension
    smoothed_model_rates = gaussian_smoothing(
        tf.convert_to_tensor(model_spikes_3d, dtype=tf.float32), sigma=50
    )
    smoothed_model_rates = smoothed_model_rates.numpy() * 1000  # Convert to Hz
    smoothed_model_rates = smoothed_model_rates[0]  # Remove batch dimension

    # Add small constant to avoid log(0) and apply log transform
    small_constant = 0.1
    target_rates = np.log(target_rates + small_constant)
    smoothed_model_rates = np.log(smoothed_model_rates + small_constant)

    # Subtract the mean of the first 500 ms for each neuron
    mean_500ms_target = np.mean(target_rates[:500], axis=0, keepdims=True)
    mean_500ms_smoothed = np.mean(smoothed_model_rates[:500], axis=0, keepdims=True)
    target_rates -= mean_500ms_target
    smoothed_model_rates -= mean_500ms_smoothed

    # Z-scoring option
    if z_scored:
        target_rates_mean = np.mean(target_rates, axis=0)
        target_rates_std = np.std(target_rates, axis=0) + 1e-8
        target_rates = (target_rates - target_rates_mean) / target_rates_std

        smoothed_model_rates_mean = np.mean(smoothed_model_rates, axis=0)
        smoothed_model_rates_std = np.std(smoothed_model_rates, axis=0) + 1e-8
        smoothed_model_rates = (smoothed_model_rates - smoothed_model_rates_mean) / smoothed_model_rates_std

        cmap = "bwr"  # Diverging colormap for z-scored data
        vmin, vmax = -5, 5  # Z-score range for diverging colormap
    else:
        cmap = "Greys"
        vmin, vmax = -1, 5  # Auto-scale for raw data

    # Transpose for plotting (neurons on Y-axis, time on X-axis)
    target_rates = np.transpose(target_rates)  # Shape: [num_target_cells, seq_len]
    smoothed_model_rates = np.transpose(smoothed_model_rates)  # Shape: [n_neurons, seq_len]
    model_spikes = np.transpose(model_spikes)  # Shape: [n_neurons, seq_len]

    # Extract spike times and neurons for raster plot
    spike_neurons, spike_times = np.nonzero(model_spikes)

    # Create the figure and subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 10), constrained_layout=False)

    # Plot target rates heatmap
    im1 = axes[0].imshow(
        target_rates,
        aspect="auto",
        cmap=cmap,
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        extent=[0, target_rates.shape[1], 0, target_rates.shape[0]],
    )
    axes[0].set_title("Target Firing Rates Heatmap\n(Log Transformed & Baseline Corrected)")
    axes[0].set_xlabel("Time (ms)")
    axes[0].set_ylabel("Target Cell Index")
    fig.colorbar(im1, ax=axes[0], label="Z-Score" if z_scored else "Log Firing Rate (Hz)")

    # Plot smoothed model spike rates heatmap
    im2 = axes[1].imshow(
        smoothed_model_rates,
        aspect="auto",
        cmap=cmap,
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        extent=[0, smoothed_model_rates.shape[1], 0, smoothed_model_rates.shape[0]],
    )
    axes[1].set_title("Smoothed Model Spike Rates\n(Log Transformed & Baseline Corrected)")
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("Neuron Index")
    fig.colorbar(im2, ax=axes[1], label="Z-Score" if z_scored else "Log Firing Rate (Hz)")

    # Plot model spikes raster
    axes[2].scatter(spike_times, spike_neurons, c="black", s=1.5, alpha=0.5, label="Spikes")
    axes[2].set_title("Model Spikes Raster")
    axes[2].set_xlim(0, model_spikes.shape[1])  # Match x-axis to target rates
    axes[2].set_ylim(0, model_spikes.shape[0])  # Match y-axis to number of neurons
    axes[2].set_xlabel("Time (ms)")
    axes[2].set_ylabel("Neuron Index")

    # Optionally save the figure (mimics naming style from the reference function)
    if save:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{file_string}_{epoch}.png")
        plt.savefig(save_path)
        print(f"Plot saved to: {save_path}")

    # Optionally display
    if display:
        plt.show()
    else:
        plt.close(fig)  # Prevents inline display in some environments

### LOSS & RELATED FUNCTIONS ###

def compute_combined_loss(
    model,
    model_spikes, 
    target_rates, 
    sigma=50, 
    exclude_start_ms=0, 
    exclude_end_ms=0, 
    rate_weight=0.00002, 
    log_sum_weight=1e-10, # controls the strength of the log-sum regularization for controlling weight sparsity, how many go to 0. Default to 0, turn on later
    zscore_mse_weight=0.0, # good things to test with current/new versions of code
    zscore_var_weight=0.0, # ^
    min_activity_weight=1,  # 2,
    max_activity_weight=0,  # 0.2,
    time_point_min_activity_penalty_weight=0,  # 1,
    time_point_max_activity_penalty_weight=0  # 0.1
):
    """
    Compute a combined loss that includes:
    - Smoothed rate loss
    - Log-Sum regularization
    - Z-score MSE and variance losses
    - Penalties for activity below/above thresholds (both across neurons and time points).
    """
    epsilon = 1e-8  # Small constant to prevent numerical instability

    target_rates = target_rates * 1000

    # Smooth the model spikes using Gaussian smoothing
    smoothed_model_rates = 1000 * gaussian_smoothing(model_spikes, sigma=sigma)

    # Adjust start and end indices for exclusion
    seq_length = tf.shape(model_spikes)[1]
    start_idx = exclude_start_ms
    end_idx = seq_length - exclude_end_ms

    smoothed_model_rates = smoothed_model_rates[:, start_idx:end_idx, :]
    target_rates = target_rates[:, start_idx:end_idx, :]

    # Compute the mean target rate per neuron across the entire time interval
    mean_target_rates_per_neuron = tf.reduce_mean(target_rates, axis=1, keepdims=True)

    # Penalize activity below half the mean target firing rate (per neuron across time)
    min_target_activity = 0.5 * mean_target_rates_per_neuron
    neuron_activity_loss = tf.reduce_mean(tf.maximum(min_target_activity - smoothed_model_rates, 0.0))

    # Penalize activity exceeding twice the mean target firing rate (per neuron across time)
    max_target_activity = 2.0 * mean_target_rates_per_neuron # was 5
    neuron_activity_cap_loss = tf.reduce_mean(tf.maximum(smoothed_model_rates - max_target_activity, 0.0))

    # Compute the mean activity across neurons for each time point
    mean_model_activity_per_time = tf.reduce_mean(smoothed_model_rates, axis=2, keepdims=True)
    mean_target_activity_per_time = tf.reduce_mean(target_rates, axis=2, keepdims=True)

    # Penalize time points where mean activity drops below half the mean target activity (time-based loss)
    min_target_time_activity = 0.5 * mean_target_activity_per_time
    time_point_activity_loss = tf.reduce_mean(tf.maximum(min_target_time_activity - mean_model_activity_per_time, 0.0))

    # Penalize time points where mean activity exceeds twice the mean target activity (time-based cap)
    max_target_time_activity = 2.0 * mean_target_activity_per_time # was 10
    time_point_activity_cap_loss = tf.reduce_mean(tf.maximum(mean_model_activity_per_time - max_target_time_activity, 0.0))

    # Compute smoothed rate loss
    #delta = 1.0  # Tune as needed
    rate_loss_contributions = (smoothed_model_rates - target_rates) ** 2
    rate_loss_contributions = rate_loss_contributions / (mean_target_rates_per_neuron + 0.001)  # Normalize by target rates
    smoothed_rate_loss = tf.reduce_mean(rate_loss_contributions)

    # Compute Log-Sum regularization loss
    weights = model.trainable_variables[0]  # Assuming the sparse input weights are variable 0
    log_sum_loss = tf.reduce_sum(tf.math.log(weights + epsilon))

    # Compute z-score MSE and variance losses
    model_zscores = (smoothed_model_rates - tf.reduce_mean(smoothed_model_rates, axis=1, keepdims=True)) / (
        tf.math.reduce_std(smoothed_model_rates, axis=1, keepdims=True) + epsilon
    )
    target_zscores = (target_rates - tf.reduce_mean(target_rates, axis=1, keepdims=True)) / (
        tf.math.reduce_std(target_rates, axis=1, keepdims=True) + epsilon
    )

    # Z-score MSE loss
    zscore_mse_loss = tf.reduce_mean((model_zscores - target_zscores) ** 2)

    # Z-score variance loss
    model_zscores_var = tf.math.reduce_variance(model_zscores, axis=2)
    target_zscores_var = tf.math.reduce_variance(target_zscores, axis=2)
    zscore_var_loss = tf.reduce_mean(tf.abs(model_zscores_var - target_zscores_var))

    # Combine all losses
    combined_loss = (
        rate_weight * smoothed_rate_loss +
        log_sum_weight * log_sum_loss +
        zscore_mse_weight * zscore_mse_loss +
        zscore_var_weight * zscore_var_loss +
        min_activity_weight * neuron_activity_loss +
        max_activity_weight * neuron_activity_cap_loss +
        time_point_min_activity_penalty_weight * time_point_activity_loss +
        time_point_max_activity_penalty_weight * time_point_activity_cap_loss
    )
    
    return combined_loss                                            

def sigmoid_with_range(proportions, target=0.5, k=10.0, scale=1.0):
    """
    Sigmoid-like function that starts at +1 (far left),
    crosses 0 at proportion = target, and goes to -1 (far right).

    f(x) = scale * [ 1 - 2 / (1 + exp(-k * (x - target))) ]

    Parameters
    ----------
    proportions : array-like
        Array of proportion values (0 to 1).
    target : float
        Where the function crosses zero.
    k : float
        Controls slope (steepness) of transition around target.
    scale : float
        Overall scaling factor (e.g., flags.log_sum_weight).

    Returns
    -------
    np.ndarray
        The computed function values for each proportion.
    """
    return scale * (1.0 - 2.0 / (1.0 + np.exp(-k * (proportions - target))))






def small_gaussian_kernel(sigma=2.0, max_extent=3):
    """
    Creates a small 1D Gaussian kernel centered at 0,
    with integer width ~ (2*max_extent + 1).
    Ensures it sums to 1.
    """
    half_size = max_extent
    x = tf.range(-half_size, half_size+1, dtype=tf.float32)
    g = tf.exp(-0.5 * (x / sigma)**2)
    g /= tf.reduce_sum(g)  # normalize to sum=1
    return g  # shape: [2*max_extent + 1, ]

def spike_spreading_conv(spikes, kernel):
    """
    1D convolution of 'spikes' along the time dimension with a small Gaussian kernel.
    spikes: [batch, time, neurons]
    kernel: 1D kernel of shape [k_size]
    Returns 'spike_density' of same shape, 
    where each spike is spread over ~k_size bins.
    """
    k_size = tf.shape(kernel)[0]
    # reshape kernel to [k_size, 1, 1, 1] for depthwise conv2d
    kernel_4d = tf.reshape(kernel, [k_size, 1, 1, 1])

    pad_size = k_size // 2
    # pad spikes along time dimension
    spikes_padded = tf.pad(spikes, [[0,0],[pad_size,pad_size],[0,0]], mode='CONSTANT')
    # expand dims => [batch, time, neurons, 1]
    spikes_4d = tf.expand_dims(spikes_padded, axis=-1)

    # Depthwise conv => convolving across time dimension only
    spike_density_4d = tf.nn.depthwise_conv2d(
        input=spikes_4d,
        filter=kernel_4d,
        strides=[1,1,1,1],
        padding='VALID'
    )  # shape => [batch, time, neurons, 1]

    # remove last dim
    spike_density = tf.squeeze(spike_density_4d, axis=-1)
    # shape => [batch, time, neurons]

    return spike_density


# def compute_combined_loss2(
#     model,
#     model_spikes_list,
#     target_rates,
#     target_scaling=1.0,
#     sigma=150,
#     start_idx=0,
#     end_idx=-1,
#     rate_weight=1.0,
#     log_sum_weight=1e-7,
#     plot_neurons=False,  # New argument to enable plotting
#     neurons_to_plot=None,  # List of neuron indices to plot
#     epoch=0,  # Epoch number for saving the figure
#     save_dir='model_checkpoints'  # Directory for saving plots
# ):
#     """
#     Compute a combined loss that includes:
#     - Log of Mean Squared Error (MSE) between smoothed model rates and smoothed target rates.
#     - Log-sum regularization on the first trainable variable.
#     - Averages the loss across multiple input spike sets if provided.
#     - Optionally plots and saves neuron-specific rate comparisons.

#     Parameters
#     ----------
#     model : tf.keras.Model
#         The model whose trainable variables are accessed for regularization.
#     model_spikes_list : list of tf.Tensor or single tf.Tensor
#         Spike trains from the model; each tensor shape is [batch, time, neurons].
#         Each entry is expected to be 0 or 1 in each 1 ms bin.
#     target_rates : tf.Tensor
#         Target firing rates, shape [batch, time, neurons]. 
#         (Internally scaled to Hz & smoothed.)
#     sigma : float
#         Gaussian smoothing window for the spike trains.
#     start_idx : int
#         Starting index (time) to include in the loss.
#     end_idx : int
#         Ending index (time) to include in the loss.
#     rate_weight : float
#         Weight for the rate alignment loss.
#     log_sum_weight : float
#         Weight for the log-sum regularization term.
#     plot_neurons : bool, optional
#         If True, calls the plotting function to visualize selected neuron rates.
#     neurons_to_plot : list, optional
#         List of neuron indices to plot.
#     epoch : int, optional
#         Current epoch number, used for saving plots.
#     save_dir : str, optional
#         Directory for saving neuron rate plots.

#     Returns
#     -------
#     combined_loss : tf.Tensor
#         The total loss (log(MSE) + log-sum reg).
#     rate_loss : tf.Tensor
#         The portion of the loss from the rate alignment (log MSE).
#     log_sum_regularization : tf.Tensor
#         The log-sum regularization component.
#     """


#     # Ensure model_spikes_list is a list
#     if not isinstance(model_spikes_list, list):
#         model_spikes_list = [model_spikes_list]
#     num_sets = len(model_spikes_list)

#     # Scale and smooth target rates
#     scaled_target_rates = target_scaling*(target_rates * 1000.0) 
#     smoothed_target_rates = gaussian_smoothing(scaled_target_rates, sigma=sigma)
#     smoothed_target_rates = smoothed_target_rates[:, start_idx:end_idx, :]  # [batch, time, neurons]

#     # Accumulate loss across multiple model spike sets
#     total_rate_loss = 0.0
#     selected_model_traces = []
#     selected_target_traces = []

#     for model_spikes in model_spikes_list:
#         # Smooth model spikes to get model rates
#         smoothed_model_rates = 1000.0 * gaussian_smoothing(model_spikes, sigma=sigma)
#         smoothed_model_rates = smoothed_model_rates[:, start_idx:end_idx, :]  # [batch, time, neurons]

#         # --- Compute MSE ---
#         mse_per_neuron = tf.reduce_mean(tf.square(smoothed_model_rates - smoothed_target_rates), axis=[1, 2])
#         mse_loss = tf.reduce_mean(mse_per_neuron)  # Average over batch

#         # Apply log(MSE + 1.0) transformation
#         #log_mse_loss = tf.math.log(mse_loss + 1.0)
#         log_mse_loss = mse_loss

#         # Multiply by rate_weight and accumulate
#         total_rate_loss += rate_weight * log_mse_loss

#         # If plotting is enabled, extract neuron traces
#         if plot_neurons and neurons_to_plot is not None:
#             # Use tf.gather to correctly index multiple neurons
#             selected_model_traces.append(tf.gather(smoothed_model_rates[0], neurons_to_plot, axis=-1))  # Extract time traces
#             selected_target_traces.append(tf.gather(smoothed_target_rates[0], neurons_to_plot, axis=-1))

#     # Average across multiple spike sets if provided
#     final_rate_loss = total_rate_loss / (target_scaling*target_scaling*num_sets)

#     # --- Log-Sum Regularization ---
#     if log_sum_weight>0:
#         log_sum_regularization = log_sum_weight * tf.reduce_sum(
#             tf.math.log(tf.abs(model.trainable_variables[0]) + 1.0))
#     else:
#         log_sum_regularization = tf.constant(0.0, dtype=tf.float32)

#     # Final combined loss
#     final_combined_loss = final_rate_loss + log_sum_regularization

#     # Call the plotting function if needed
#     if plot_neurons and neurons_to_plot is not None:
#         plot_selected_neuron_rates(
#             selected_model_traces,
#             selected_target_traces,
#             neurons_to_plot,
#             epoch,
#             save_dir
#         )

#     return final_combined_loss, final_rate_loss, log_sum_regularization

# def compute_combined_loss2(
#     model,
#     model_spikes_list,
#     target_rates,
#     target_scaling=1.0,
#     sigma=150,
#     start_idx=0,
#     end_idx=-1,
#     rate_weight=1.0,
#     log_sum_weight=1e-7,
#     plot_neurons=False,  
#     neurons_to_plot=None,  
#     epoch=0,  
#     save_dir='model_checkpoints'  
# ):
#     """
#     Compute a combined loss that includes:
#     - Log-transformed MSE per timestep before reduction.
#     - Log-sum regularization on the first trainable variable.
#     - Averages the loss across multiple input spike sets if provided.
#     - Optionally plots and saves neuron-specific rate comparisons.

#     Parameters
#     ----------
#     model : tf.keras.Model
#         The model whose trainable variables are accessed for regularization.
#     model_spikes_list : list of tf.Tensor or single tf.Tensor
#         Spike trains from the model; each tensor shape is [batch, time, neurons].
#     target_rates : tf.Tensor
#         Target firing rates, shape [batch, time, neurons].
#     sigma : float
#         Gaussian smoothing window for the spike trains.
#     start_idx : int
#         Starting index (time) to include in the loss.
#     end_idx : int
#         Ending index (time) to include in the loss.
#     rate_weight : float
#         Weight for the rate alignment loss.
#     log_sum_weight : float
#         Weight for the log-sum regularization term.
#     plot_neurons : bool, optional
#         If True, calls the plotting function to visualize selected neuron rates.
#     neurons_to_plot : list, optional
#         List of neuron indices to plot.
#     epoch : int, optional
#         Current epoch number, used for saving plots.
#     save_dir : str, optional
#         Directory for saving neuron rate plots.

#     Returns
#     -------
#     combined_loss : tf.Tensor
#         The total loss (log-transformed MSE + log-sum reg).
#     rate_loss : tf.Tensor
#         The portion of the loss from the rate alignment.
#     log_sum_regularization : tf.Tensor
#         The log-sum regularization component.
#     """

#     # Ensure model_spikes_list is a list
#     if not isinstance(model_spikes_list, list):
#         model_spikes_list = [model_spikes_list]
#     num_sets = len(model_spikes_list)

#     # Scale and smooth target rates
#     scaled_target_rates = target_scaling * (target_rates * 1000.0) 
#     smoothed_target_rates = gaussian_smoothing(scaled_target_rates, sigma=sigma)
#     smoothed_target_rates = smoothed_target_rates[:, start_idx:end_idx, :]  # [batch, time, neurons]

#     shift = 5.0  # Shift to prevent instability in log transform

#     # Accumulate loss across multiple model spike sets
#     total_rate_loss = 0.0
#     selected_model_traces = []
#     selected_target_traces = []

#     for model_spikes in model_spikes_list:
#         # Smooth model spikes to get model rates
#         smoothed_model_rates = 1000.0 * gaussian_smoothing(model_spikes, sigma=sigma)
#         smoothed_model_rates = smoothed_model_rates[:, start_idx:end_idx, :]  # [batch, time, neurons]

#         # Compute per-time-step squared error
#         squared_error = tf.square(smoothed_model_rates - smoothed_target_rates)  # [batch, time, neurons]

#         # Apply log transformation with shift
#         log_mse_per_timestep = tf.math.log(squared_error + shift)  # Element-wise log transform

#         # Reduce over time and neurons
#         neuronwise_loss = tf.reduce_mean(log_mse_per_timestep, axis=[1, 2])  # [batch]
#         log_mse_loss = tf.reduce_mean(neuronwise_loss)  # Final batch-averaged loss

#         # Multiply by rate_weight and accumulate
#         total_rate_loss += rate_weight * log_mse_loss

#         # If plotting is enabled, extract neuron traces
#         if plot_neurons and neurons_to_plot is not None:
#             selected_model_traces.append(tf.gather(smoothed_model_rates[0], neurons_to_plot, axis=-1))  
#             selected_target_traces.append(tf.gather(smoothed_target_rates[0], neurons_to_plot, axis=-1))

#     # Average across multiple spike sets if provided
#     final_rate_loss = total_rate_loss / (target_scaling * target_scaling * num_sets)

#     # --- Log-Sum Regularization ---
#     if log_sum_weight > 0:
#         log_sum_regularization = log_sum_weight * tf.reduce_sum(
#             tf.math.log(tf.abs(model.trainable_variables[0]) + 1.0)
#         )
#     else:
#         log_sum_regularization = tf.constant(0.0, dtype=tf.float32)

#     # Final combined loss
#     final_combined_loss = final_rate_loss + log_sum_regularization

#     # Call the plotting function if needed
#     if plot_neurons and neurons_to_plot is not None:
#         plot_selected_neuron_rates(
#             selected_model_traces,
#             selected_target_traces,
#             neurons_to_plot,
#             epoch,
#             save_dir
#         )

#     return final_combined_loss, final_rate_loss, log_sum_regularization

# def compute_combined_loss2(
#     model,
#     model_spikes_list,
#     target_rates,
#     target_scaling=1.0,
#     sigma=150,
#     start_idx=0,
#     end_idx=-1,
#     rate_weight=1.0,
#     log_sum_weight=1e-7,
#     plot_neurons=False,  
#     neurons_to_plot=None,  
#     epoch=0,  
#     save_dir='model_checkpoints'  
# ):
#     """
#     Compute a combined loss that includes:
#     - Mean Squared Error (MSE) between smoothed model rates and smoothed target rates.
#     - Log-sum regularization on the first trainable variable.
#     - Iterates through batch instances to apply Gaussian smoothing individually.
#     - Optionally plots and saves neuron-specific rate comparisons.

#     Parameters
#     ----------
#     model : tf.keras.Model
#         The model whose trainable variables are accessed for regularization.
#     model_spikes_list : list of tf.Tensor or single tf.Tensor
#         This argument is ignored in loss computation but remains for compatibility.
#     target_rates : tf.Tensor
#         Target firing rates, shape [batch, time, neurons].
#     sigma : float
#         Gaussian smoothing window for the spike trains.
#     start_idx : int
#         Starting index (time) to include in the loss.
#     end_idx : int
#         Ending index (time) to include in the loss.
#     rate_weight : float
#         Weight for the rate alignment loss.
#     log_sum_weight : float
#         Weight for the log-sum regularization term.
#     plot_neurons : bool, optional
#         If True, calls the plotting function to visualize selected neuron rates.
#     neurons_to_plot : list, optional
#         List of neuron indices to plot.
#     epoch : int, optional
#         Current epoch number, used for saving plots.
#     save_dir : str, optional
#         Directory for saving neuron rate plots.

#     Returns
#     -------
#     combined_loss : tf.Tensor
#         The total loss (MSE + log-sum reg).
#     rate_loss : tf.Tensor
#         The portion of the loss from the rate alignment.
#     log_sum_regularization : tf.Tensor
#         The log-sum regularization component.
#     """

#     #tf.print("DEBUG: target_rates shape:", tf.shape(target_rates))
#     #tf.print("DEBUG: model.output shape:", tf.shape(model.output))
#     tf.py_function(func=lambda: tf.print("DEBUG: model.output shape:", tf.shape(model.output)), inp=[], Tout=[])
#     tf.py_function(func=lambda: tf.print("DEBUG: target_rates shape:", tf.shape(target_rates)), inp=[], Tout=[])

#     batch_size = tf.shape(target_rates)[0]  # Get the batch size dynamically

#     total_mse_loss = 0.0

#     for i in range(batch_size):
#         # Extract single batch instance and ensure shape compatibility
#         model_spikes_instance = tf.expand_dims(model.output[i], axis=0)  # Add batch dimension
#         target_rates_instance = tf.expand_dims(target_rates[i], axis=0)  # Add batch dimension

#         # Scale and smooth target rates
#         scaled_target_rates = target_scaling * (target_rates_instance * 1000.0)
#         smoothed_target_rates = gaussian_smoothing(scaled_target_rates, sigma=sigma)
#         smoothed_target_rates = smoothed_target_rates[start_idx:end_idx, :]  # [time, neurons]

#         # Smooth model spikes to get model rates
#         smoothed_model_rates = 1000.0 * gaussian_smoothing(model_spikes_instance, sigma=sigma)
#         smoothed_model_rates = smoothed_model_rates[start_idx:end_idx, :]  # [time, neurons]

#         # Compute Mean Squared Error (MSE)
#         mse_loss = tf.reduce_mean(tf.square(smoothed_model_rates - smoothed_target_rates))

#         # Accumulate loss
#         total_mse_loss += mse_loss

#     # Average over batch
#     final_rate_loss = rate_weight * (total_mse_loss / tf.cast(batch_size, tf.float32))

#     # --- Log-Sum Regularization ---
#     if log_sum_weight > 0:
#         log_sum_regularization = log_sum_weight * tf.reduce_sum(
#             tf.math.log(tf.abs(model.trainable_variables[0]) + 1.0)
#         )
#     else:
#         log_sum_regularization = tf.constant(0.0, dtype=tf.float32)

#     # Final combined loss
#     final_combined_loss = final_rate_loss + log_sum_regularization

#     return final_combined_loss, final_rate_loss, log_sum_regularization


# def compute_combined_loss2(
#     model,
#     model_spikes,  # Now expects [seq_len, n_neurons]
#     target_rates,  # Now expects [seq_len, n_neurons]
#     target_scaling=1.0,
#     sigma=150,
#     start_idx=None,
#     end_idx=None,
#     rate_weight=1.0,
#     log_sum_weight=1e-7,
# ):
#     """
#     Computes loss using Gaussian-smoothed rates for a **single instance**.
#     """
#     # ✅ Scale and smooth rates
#     scaled_target_rates = target_scaling * (target_rates * 1000.0)
#     smoothed_target_rates = gaussian_smoothing(scaled_target_rates, sigma=sigma)
#     smoothed_model_rates = 1000.0 * gaussian_smoothing(model_spikes, sigma=sigma)

#     # ✅ Extract time window using start_idx & end_idx
#     smoothed_target_rates = smoothed_target_rates[start_idx:end_idx, :]
#     smoothed_model_rates = smoothed_model_rates[start_idx:end_idx, :]

#     # ✅ Compute Mean Squared Error loss
#     mse_loss = tf.reduce_mean(tf.square(smoothed_model_rates - smoothed_target_rates))
    
#     # ✅ Compute element-wise squared error
#     #squared_error = tf.square(smoothed_model_rates - smoothed_target_rates)

#     # ✅ Take log of (squared_error + 5) **before** reducing
#     #log_squared_error = tf.math.log(squared_error + 5.0)

#     # ✅ Compute mean log MSE
#     #mse_loss = tf.reduce_mean(log_squared_error)

#     # ✅ Log-Sum Regularization
#     log_sum_regularization = tf.constant(0.0, dtype=tf.float32)
#     if log_sum_weight > 0:
#         try:
#             # Get the layer by name
#             layer = model.get_layer("input_layer")
#             # Get the trainable kernel (weights) tensor
#             weights_tensor = layer.trainable_weights[0]  # assuming the kernel is first
#             log_sum_regularization = log_sum_weight * tf.reduce_sum(tf.math.log(tf.abs(weights_tensor) + 1e-8))
#         except Exception as e:
#             tf.print("Error accessing input_layer weights:", e)

#         #log_sum_regularization = log_sum_weight * tf.reduce_sum(
#         #    tf.math.log(tf.abs(model.trainable_variables[0]) + 0.00000001)
#         #)

#     final_combined_loss = (rate_weight * mse_loss) + log_sum_regularization

#     return final_combined_loss, (rate_weight * mse_loss), log_sum_regularization

# def compute_combined_loss2(
#     model,
#     model_spikes,                 # [seq_len, n_neurons]
#     target_rates,                 # [seq_len, n_neurons]
#     *,
#     target_scaling: float = 1.0,
#     sigma: int = 150,
#     start_idx: int | None = None,
#     end_idx:   int | None = None,
#     rate_weight: float = 0.00001,
#     log_sum_weight: float = 20.0,   # tune via flag
# ):
#     """
#     Loss =  rate_weight · ⟨MSE⟩  +  log_sum_weight · ⟨ -log(|w|/scale + 1e-8) ⟩
#     (scale = mean |w|, no TFP dependency)
#     ----------------------------------------------------------------------
#     Returns: total_loss,  rate_term,  log_term
#     """

#     # ------------------------------------------------------------------
#     # 1.  Rate‑matching term
#     # ------------------------------------------------------------------
#     def safe_expand_for_gaussian(tensor):
#         if len(tensor.shape) == 2:
#             return tf.expand_dims(tensor, axis=0)  # add batch dimension if missing
#         else:
#             return tensor  # already correct shape

#     # Then use this in your compute_combined_loss2 call like this:

#     # Safely expand if needed
#     tar_input = safe_expand_for_gaussian(target_scaling * target_rates * 1000.0)
#     mod_input = safe_expand_for_gaussian(model_spikes * 1000.0)

#     # Apply gaussian smoothing
#     tar = gaussian_smoothing(tar_input, sigma=sigma)
#     mod = gaussian_smoothing(mod_input, sigma=sigma)

#     # If we expanded earlier, remove the batch dimension
#     if len(tar.shape) == 3:
#         tar = tf.squeeze(tar, axis=0)
#     if len(mod.shape) == 3:
#         mod = tf.squeeze(mod, axis=0)

#     tar = tar[start_idx:end_idx, :]
#     mod = mod[start_idx:end_idx, :]

#     mse = tf.reduce_mean(tf.square(mod - tar))
#     rate_term = rate_weight * mse

#     #tf.print("tar shape after smoothing:", tf.shape(tar))
#     #tf.print("mod shape after smoothing:", tf.shape(mod))
#     #tf.print("start_idx:", start_idx, "end_idx:", end_idx)


#     # ------------------------------------------------------------------
#     # 2.  Constant L1  +  log-variance regulariser (no warm-up ramp)
#     # ------------------------------------------------------------------
#     log_term = tf.constant(0.0, dtype=mse.dtype)

#     if log_sum_weight > 0.0:
#         # ------------- locate the sparse-input weight matrix ----------
#         lay = None
#         for name in ("input_layer", "sparse_input_layer"):
#             try:
#                 lay = model.get_layer(name)
#                 # --- numeric guard: overwrite the variable itself if needed -------------
#                 var = lay.trainable_weights[0]          # keep a handle to the real variable
#                 safe_val = tf.where(tf.math.is_finite(var), var, tf.zeros_like(var))
#                 var.assign(safe_val)                    # replace bad entries in-place
#                 w = tf.convert_to_tensor(safe_val)      # continue as before

#                 break
#             except ValueError:
#                 continue

#         if lay is None:
#             tf.print("⚠ no input layer found → reg=0")
#         else:
#             # dense view of the weights
#             w = tf.convert_to_tensor(lay.trainable_weights[0])

#             # ----------------------------------------------------------
#             # numeric guard: replace any NaN/Inf with 0
#             # ----------------------------------------------------------
#             w = tf.where(tf.math.is_finite(w), w, tf.zeros_like(w))

#             # ----------------------------------------------------------
#             # absolute value with ε so log and gradients are finite
#             # ----------------------------------------------------------
#             eps    = 1.0e-3                       # > smallest plausible weight
#             w_abs  = tf.abs(w) + eps

#             # ----------------------------------------------------------
#             # hyper-parameters (tweak only these three)
#             # ----------------------------------------------------------
#             SPARSE_CUT  = 5e-5                  # defines “zero”
#             l1_coeff    = log_sum_weight        # λ₁  (sparsity strength)
#             lvar_coeff  = 1e-6 * l1_coeff       # λ₂  (tail shaping)

#             # ----------------------------------------------------------
#             # mask for active weights (> SPARSE_CUT)
#             # ----------------------------------------------------------
#             mask       = tf.cast(w_abs > SPARSE_CUT, w_abs.dtype)
#             active_log = tf.boolean_mask(tf.math.log(w_abs), mask)   # ignores zeros

#             # ----------------------------------------------------------
#             # penalties
#             # ----------------------------------------------------------
#             l1_term   = l1_coeff * tf.reduce_mean(w_abs)
#             lvar_term = tf.where(
#                 tf.size(active_log) > 0,
#                 lvar_coeff * tf.math.reduce_variance(active_log),
#                 tf.constant(0.0, dtype=w_abs.dtype)
#             )

#             log_term = l1_term + lvar_term

#             # ----------------------------------------------------------
#             # diagnostics (always finite)
#             # ----------------------------------------------------------
#             frac_zero = tf.reduce_mean(tf.cast(w_abs < SPARSE_CUT, w_abs.dtype))
#             safe_mean = tf.where(tf.math.is_finite(tf.reduce_mean(w_abs)),
#                                 tf.reduce_mean(w_abs),
#                                 tf.constant(0.0, w_abs.dtype))

#             safe_var  = tf.where(tf.math.is_finite(tf.math.reduce_variance(active_log)),
#                                 tf.math.reduce_variance(active_log),
#                                 tf.constant(0.0, w_abs.dtype))

#             tf.print("〈|w|〉:", safe_mean,
#                     "Var[log|w|]:", safe_var,
#                     "zero-frac:", frac_zero,
#                     "λ₁:", l1_coeff,
#                     "λ·reg:", log_term)


#     # ------------------------------------------------------------------
#     # 3.  Combined loss
#     # ------------------------------------------------------------------
#     total_loss = rate_term + log_term

#     tf.print("rate_term:", rate_term,
#              "log_term:", log_term,
#              "total:", total_loss)

#     return total_loss, rate_term, log_term


# def compute_combined_loss2(
#     model,
#     model_spikes,                 # [seq_len, n_neurons]
#     target_rates,                 # [seq_len, n_neurons]
#     *,
#     target_scaling: float = 1.0,
#     sigma: int = 150,
#     start_idx: int | None = None,
#     end_idx:   int | None = None,
#     rate_weight: float = 1e-5,
#     log_sum_weight: float = 1.0,
# ):
#     """
#     total_loss =  rate_weight · MSE(rate)
#                 + log_sum_weight · ( L1  +  α·Σ log  +  β·Var[log] )

#     α = 0.10,  β = 1e-4 by default.
#     All regulariser terms use |w|+ε with ε = 2e-5 to avoid log(0).
#     """
#     # for v in model.trainable_variables:
#     #     v.assign( tf.where( tf.math.is_finite(v), v, tf.zeros_like(v) ) )

#     # --------------------------------------------------------------
#     # 1.  rate-matching term (smoothed MSE)
#     # --------------------------------------------------------------
#     def _batch(x):
#         return x if x.ndim == 3 else tf.expand_dims(x, 0)

#     tar = gaussian_smoothing(_batch(target_scaling * target_rates * 1e3), sigma)
#     mod = gaussian_smoothing(_batch(model_spikes                * 1e3), sigma)

#     tar = tf.squeeze(tar, 0)[start_idx:end_idx]
#     mod = tf.squeeze(mod, 0)[start_idx:end_idx]

#     rate_term = rate_weight * tf.reduce_mean(tf.square(mod - tar))

#     # --------------------------------------------------------------
#     # 2.  weight regulariser (no masking, no reassignment)
#     # --------------------------------------------------------------
#     log_term = tf.constant(0.0, dtype=rate_term.dtype)

#     if log_sum_weight > 0.0:
#         # locate sparse input layer
#         try:
#             layer = model.get_layer("input_layer")
#         except ValueError:
#             try:
#                 layer = model.get_layer("sparse_input_layer")
#             except ValueError:
#                 layer = None

#         if layer is None:
#             tf.print("⚠  no input layer found — reg = 0")
#         else:
#             w = tf.convert_to_tensor(layer.trainable_weights[0])
#             w = tf.where(tf.math.is_finite(w), w, tf.zeros_like(w))   # scrub NaN/Inf

#             eps      = 0.01
#             w_abs    = tf.abs(w) + eps        # never hits zero
#             log_wabs = tf.math.log(w_abs)

#             # coefficients
#             λ1 = log_sum_weight
#             α  = 0.01 * λ1                    # weight for log-sum term
#             β  = 1e-4 * λ1                    # weight for log-variance term

#             #l1_term      = 0.0#λ1 * tf.reduce_mean(w_abs)             # sparsity
#             logsum_term  = α  * tf.reduce_mean(log_wabs) - α * tf.math.log(eps) 
#             #logvar_term  = 0.0#β  * tf.math.reduce_variance(log_wabs) # spreads the tail
#             #log_term     = l1_term + logsum_term + logvar_term
#             log_term = logsum_term

#             tf.print("〈|w|〉:", tf.reduce_mean(w_abs),
#                      "〈log|w|〉:", tf.reduce_mean(log_wabs),
#                      "Var[log|w|]:", tf.math.reduce_variance(log_wabs),
#                      "λ₁:", λ1, "α:", α, "β:", β,
#                      "log_term:", log_term)

#     # --------------------------------------------------------------
#     # 3.  combined loss
#     # --------------------------------------------------------------
#     total_loss = rate_term + log_term
#     tf.print("rate_term:", rate_term, "total_loss:", total_loss)

#     return total_loss, rate_term, log_term


# def compute_combined_loss2(
#     model,
#     model_spikes,                 # [seq_len, n_neurons]
#     target_rates,                 # [seq_len, n_neurons]
#     *,
#     target_scaling: float = 1.0,
#     sigma: int = 100.0,
#     start_idx: int | None = None,
#     end_idx:   int | None = None,
#     rate_weight: float = 0.000075,
#     log_sum_weight: float = 1.0,
#     bkg_l2_weight: float = 0.0,    # NEW  strength for bkg weights
#     rec_l2_weight: float = 0.0    # NEW  strength for recurrent weights
# ):
#     """
#     total_loss =
#           rate_weight · MSE(rate)
#         + log_sum_weight · ( α·Σ log  + … )
#         + bkg_l2_weight · ||W_bkg||²
#         + rec_l2_weight · ||W_rec||²
#     """
#     def _var_by_name(model, key):
#         """Return the first trainable variable whose .name contains `key`."""
#         for v in model.trainable_variables:
#             if key in v.name:
#                 return v
#         return None

#     def clipped_l2(w, tau):
#         """
#         Dead-zone L2 with NaN/Inf safety.

#         R_tau(w) = Σ_i  max(|w_i| - tau, 0)^2

#         • Any NaN/Inf element is treated as 0 (does not corrupt the sum).
#         • No penalty for |w| ≤ tau.
#         • Quadratic growth beyond tau.
#         """
#         # 1.  scrub non-finite values
#         w_clean = tf.where(tf.math.is_finite(w), w, tf.zeros_like(w))

#         # 2.  apply dead-zone
#         excess = tf.nn.relu(tf.abs(w_clean) - tau)   # max(|w|-tau, 0)

#         # 3.  full squared-norm (tf.nn.l2_loss returns ½‖·‖²)
#         return tf.nn.l2_loss(excess) * 2.0


#     # --------------------------------------------------------------
#     # 1.  rate-matching term
#     # --------------------------------------------------------------
#     def _batch(x): return x if x.ndim == 3 else tf.expand_dims(x, 0)

#     tar = gaussian_smoothing(_batch(target_scaling * target_rates * 1e3), sigma)
#     mod = gaussian_smoothing(_batch(model_spikes                * 1e3), sigma)

#     tar = tf.squeeze(tar, 0)[start_idx:end_idx]
#     mod = tf.squeeze(mod, 0)[start_idx:end_idx]

#     rate_term = rate_weight * tf.reduce_mean(tf.square(mod - tar)) / (target_scaling*target_scaling)

#     # --------------------------------------------------------------
#     # 2.  log-sum regulariser (unchanged)
#     # --------------------------------------------------------------
#     log_term = tf.constant(0.0, dtype=rate_term.dtype)
#     if log_sum_weight > 0.0:
#         layer = None
#         for name in ["input_layer", "sparse_input_layer"]:
#             try:
#                 layer = model.get_layer(name); break
#             except ValueError:
#                 pass
#         if layer is not None:
#             w       = tf.convert_to_tensor(layer.trainable_weights[0])
#             w       = tf.where(tf.math.is_finite(w), w, tf.zeros_like(w))
#             eps     = 5e-3
#             log_term = 0.01 * log_sum_weight * (
#                 tf.reduce_mean(tf.math.log(tf.abs(w) + eps)) - tf.math.log(eps)
#             )

#     # --------------------------------------------------------------
#     # 3.  NEW L-2 penalties   (uses helper above)
#     # --------------------------------------------------------------
#     l2_term = tf.constant(0.0, dtype=rate_term.dtype)

#     tau_bkg = 0.04         # threshold for background weights
#     tau_rec = 0.003          # threshold for recurrent weights

#     l2_term = tf.constant(0.0, dtype=rate_term.dtype)

#     if bkg_l2_weight > 0.0:
#         w_bkg = _var_by_name(model, "rest_of_brain_weights")
#         if w_bkg is not None:
#             l2_term += bkg_l2_weight * clipped_l2(w_bkg, tau_bkg)

#     if rec_l2_weight > 0.0:
#         w_rec = _var_by_name(model, "sparse_recurrent_weights")
#         if w_rec is not None:
#             l2_term += rec_l2_weight * clipped_l2(w_rec, tau_rec)



#     # --------------------------------------------------------------
#     # 4.  combined loss
#     # --------------------------------------------------------------
#     total_loss = rate_term + log_term + l2_term
#     tf.print("rate:", rate_term,
#              "log:",  log_term,
#              "L2:",   l2_term,
#              "total:", total_loss)

#     return total_loss, rate_term, log_term

# def compute_combined_loss2(
#     model,
#     model_spikes,                 # [seq_len, n_neurons]
#     target_rates,                 # [seq_len, n_neurons]
#     *,
#     target_scaling: float = 1.0,
#     sigma: int = 100.0,
#     start_idx: int | None = None,
#     end_idx:   int | None = None,
#     rate_weight: float = 0.000075,
#     log_sum_weight: float = 0.0,
#     bkg_l2_weight: float = 0.0,    # NEW  strength for bkg weights
#     rec_l2_weight: float = 0.0,    # NEW  strength for recurrent weights
#     rate_loss_type: str = "log_mse",  # NEW: "mse", "log_mse", or "soft_root"
#     soft_root_epsilon: float = 0.1  # NEW: avoid divide-by-zero in soft-root
# ):
#     """
#     total_loss =
#           rate_weight · rate_term
#         + log_sum_weight · ( α·Σ log  + … )
#         + bkg_l2_weight · ||W_bkg||²
#         + rec_l2_weight · ||W_rec||²

#     rate_term type is selected by `rate_loss_type`:
#         - "mse":       standard MSE
#         - "log_mse":   log-scaled MSE
#         - "soft_root": normalized MSE with denominator |target| + ε
#     """
#     def _var_by_name(model, key):
#         for v in model.trainable_variables:
#             if key in v.name:
#                 return v
#         return None

#     def clipped_l2(w, tau):
#         w_clean = tf.where(tf.math.is_finite(w), w, tf.zeros_like(w))
#         excess = tf.nn.relu(tf.abs(w_clean) - tau)
#         return tf.nn.l2_loss(excess) * 2.0

#     def _batch(x): return x if x.ndim == 3 else tf.expand_dims(x, 0)

#     # ─────────────────────────────────────────────────────────────
#     # 1. Rate-matching loss
#     # ─────────────────────────────────────────────────────────────
#     tar = gaussian_smoothing(_batch(target_scaling * target_rates * 1e3), sigma)
#     mod = gaussian_smoothing(_batch(model_spikes                * 1e3), sigma)

#     tar = tf.squeeze(tar, 0)[start_idx:end_idx]
#     mod = tf.squeeze(mod, 0)[start_idx:end_idx]

#     if rate_loss_type == "mse":
#         rate_term = rate_weight * tf.reduce_mean(tf.square(mod - tar)) / (target_scaling * target_scaling)

#     elif rate_loss_type == "log_mse":
#         tar_log = tf.math.log1p(tar)
#         mod_log = tf.math.log1p(mod)
#         rate_term = rate_weight * tf.reduce_mean(tf.square(mod_log - tar_log))

#     elif rate_loss_type == "soft_root":
#         rate_term = rate_weight * tf.reduce_mean(tf.square(mod - tar) / (tf.abs(tar) + soft_root_epsilon))

#     else:
#         raise ValueError(f"Unrecognized rate_loss_type: {rate_loss_type!r}")

#     # ─────────────────────────────────────────────────────────────
#     # 2. Log-sum weight regularizer
#     # ─────────────────────────────────────────────────────────────
#     log_term = tf.constant(0.0, dtype=rate_term.dtype)
#     if log_sum_weight > 0.0:
#         layer = None
#         for name in ["input_layer", "sparse_input_layer"]:
#             try:
#                 layer = model.get_layer(name)
#                 break
#             except ValueError:
#                 pass
#         if layer is not None:
#             w = tf.convert_to_tensor(layer.trainable_weights[0])
#             w = tf.where(tf.math.is_finite(w), w, tf.zeros_like(w))
#             eps = 5e-3
#             log_term = 0.01 * log_sum_weight * (
#                 tf.reduce_mean(tf.math.log(tf.abs(w) + eps)) - tf.math.log(eps)
#             )

#     # ─────────────────────────────────────────────────────────────
#     # 3. L2 penalties
#     # ─────────────────────────────────────────────────────────────
#     l2_term = tf.constant(0.0, dtype=rate_term.dtype)

#     tau_bkg = 0.04
#     tau_rec = 0.003

#     if bkg_l2_weight > 0.0:
#         w_bkg = _var_by_name(model, "rest_of_brain_weights")
#         if w_bkg is not None:
#             l2_term += bkg_l2_weight * clipped_l2(w_bkg, tau_bkg)

#     if rec_l2_weight > 0.0:
#         w_rec = _var_by_name(model, "sparse_recurrent_weights")
#         if w_rec is not None:
#             l2_term += rec_l2_weight * clipped_l2(w_rec, tau_rec)

#     # ─────────────────────────────────────────────────────────────
#     # 4. Combine terms
#     # ─────────────────────────────────────────────────────────────
#     total_loss = rate_term + log_term + l2_term
#     tf.print("rate:", rate_term,
#              "log:",  log_term,
#              "L2:",   l2_term,
#              "total:", total_loss)

#     return total_loss, rate_term, log_term

#Good! Aug 19, 2025
# def compute_combined_loss2(
#     model,
#     model_spikes,                 # [seq_len, n_neurons]
#     target_rates,                 # [seq_len, n_neurons]
#     *,
#     target_scaling: float = 1.0,
#     sigma: int = 100.0,
#     start_idx: int | None = None,
#     end_idx:   int | None = None,
#     rate_weight: float = 0.000075,
#     log_sum_weight: float = 0.0,
#     bkg_l2_weight: float = 0.0,
#     rec_l2_weight: float = 0.0,
#     input_l1_weight: float = 0.0,
#     input_l2_weight: float = 0.0,
#     recurrent_l1_weight: float = 0.0,  # NEW: recurrent L1 regularization
#     recurrent_l2_weight: float = 0.0,  # NEW: recurrent L2 regularization  
#     background_l1_weight: float = 0.0, # NEW: background L1 regularization
#     background_l2_weight: float = 0.0, # NEW: background L2 regularization
#     rate_loss_type: str = "mse", # NEW: "mse", "log_mse", or "soft_root"
#     soft_root_epsilon: float = 0.1  # NEW: avoid divide-by-zero in soft-root
# ):
#     def _var_by_name(model, key):
#         for v in model.trainable_variables:
#             if key in v.name:
#                 return v
#         return None

#     def clipped_l2(w, tau):
#         w_clean = tf.where(tf.math.is_finite(w), w, tf.zeros_like(w))
#         excess = tf.nn.relu(tf.abs(w_clean) - tau)
#         return tf.nn.l2_loss(excess) * 2.0

#     def _batch(x): return x if x.ndim == 3 else tf.expand_dims(x, 0)

#     # ─────────────────────────────────────────────
#     # 1. Rate-matching loss
#     # ─────────────────────────────────────────────
#     tar = gaussian_smoothing(_batch(target_scaling * target_rates * 1e3), sigma)
#     mod = gaussian_smoothing(_batch(model_spikes                * 1e3), sigma)

#     tar = tf.squeeze(tar, 0)[start_idx:end_idx]
#     mod = tf.squeeze(mod, 0)[start_idx:end_idx]

#     if rate_loss_type == "mse":
#         rate_term = rate_weight * tf.reduce_mean(tf.square(mod - tar)) / (target_scaling * target_scaling)
#     elif rate_loss_type == "log_mse":
#         tar_log = tf.math.log1p(tar)
#         mod_log = tf.math.log1p(mod)
#         rate_term = rate_weight * tf.reduce_mean(tf.square(mod_log - tar_log))
#     elif rate_loss_type == "soft_root":
#         rate_term = rate_weight * tf.reduce_mean(tf.square(mod - tar) / (tf.abs(tar) + soft_root_epsilon))
#     else:
#         raise ValueError(f"Unrecognized rate_loss_type: {rate_loss_type!r}")

#     # ─────────────────────────────────────────────
#     # 2. Log-sum regularizer (unchanged)
#     # ─────────────────────────────────────────────
#     log_term = tf.constant(0.0, dtype=rate_term.dtype)
#     if log_sum_weight > 0.0:
#         layer = None
#         for name in ["input_layer", "sparse_input_layer"]:
#             try:
#                 layer = model.get_layer(name)
#                 break
#             except ValueError:
#                 pass
#         if layer is not None:
#             w = tf.convert_to_tensor(layer.trainable_weights[0])
#             w = tf.where(tf.math.is_finite(w), w, tf.zeros_like(w))
#             eps = 5e-3
#             log_term = 0.01 * log_sum_weight * (
#                 tf.reduce_mean(tf.math.log(tf.abs(w) + eps)) - tf.math.log(eps)
#             )

#     # ─────────────────────────────────────────────
#     # 3. L1 and L2 regularization for all weight types
#     # ─────────────────────────────────────────────
    
#     # Initialize all regularization terms
#     l1_term = tf.constant(0.0, dtype=rate_term.dtype)
#     l2_term = tf.constant(0.0, dtype=rate_term.dtype)
    
#     # Helper function for safe L1 regularization (handles positive and negative weights)
#     def safe_l1_reg(weights, weight_value):
#         """Apply L1 regularization with proper handling of positive/negative weights."""
#         if weight_value > 0.0 and weights is not None:
#             weights_clean = tf.where(tf.math.is_finite(weights), weights, tf.zeros_like(weights))
#             return weight_value * tf.reduce_sum(tf.abs(weights_clean))
#         return tf.constant(0.0, dtype=rate_term.dtype)
    
#     # Helper function for safe L2 regularization  
#     def safe_l2_reg(weights, weight_value, use_clipped=False, tau=None):
#         """Apply L2 regularization, optionally with clipping."""
#         if weight_value > 0.0 and weights is not None:
#             weights_clean = tf.where(tf.math.is_finite(weights), weights, tf.zeros_like(weights))
#             if use_clipped and tau is not None:
#                 return weight_value * clipped_l2(weights_clean, tau)
#             else:
#                 return weight_value * tf.nn.l2_loss(weights_clean) * 2.0
#         return tf.constant(0.0, dtype=rate_term.dtype)
    
#     # Note: Clipped L2 thresholds are no longer used (switched to regular L2)
#     # tau_bkg = 0.04  # REMOVED
#     # tau_rec = 0.003  # REMOVED
    
#     # ---- INPUT WEIGHTS ----
#     w_input = None
#     for name in ["input_layer", "sparse_input_layer"]:
#         try:
#             layer = model.get_layer(name)
#             w_input = layer.trainable_weights[0]
#             break
#         except ValueError:
#             continue
    
#     if w_input is not None:
#         l1_term += safe_l1_reg(w_input, input_l1_weight)
#         l2_term += safe_l2_reg(w_input, input_l2_weight)
#     else:
#         tf.print("[WARN] input weights not found for regularization — skipping input regularization.")
    
#     # ---- RECURRENT WEIGHTS ----
#     w_rec = _var_by_name(model, "sparse_recurrent_weights")
#     if w_rec is not None:
#         # Support both old and new parameter names for backward compatibility
#         rec_l1_weight = recurrent_l1_weight  # New parameter name
#         rec_l2_weight_combined = rec_l2_weight + recurrent_l2_weight  # Old + new
        
#         l1_term += safe_l1_reg(w_rec, rec_l1_weight)
#         l2_term += safe_l2_reg(w_rec, rec_l2_weight_combined, use_clipped=False)  # CHANGED: use regular L2
#     else:
#         if recurrent_l1_weight > 0.0 or recurrent_l2_weight > 0.0 or rec_l2_weight > 0.0:
#             tf.print("[WARN] recurrent weights not found for regularization — skipping recurrent regularization.")
    
#     # ---- BACKGROUND WEIGHTS ----
#     w_bkg = _var_by_name(model, "rest_of_brain_weights")
#     if w_bkg is not None:
#         # Support both old and new parameter names for backward compatibility
#         bkg_l1_weight = background_l1_weight  # New parameter name
#         bkg_l2_weight_combined = bkg_l2_weight + background_l2_weight  # Old + new
        
#         l1_term += safe_l1_reg(w_bkg, bkg_l1_weight)
#         l2_term += safe_l2_reg(w_bkg, bkg_l2_weight_combined, use_clipped=False)  # CHANGED: use regular L2
#     else:
#         if background_l1_weight > 0.0 or background_l2_weight > 0.0 or bkg_l2_weight > 0.0:
#             tf.print("[WARN] background weights not found for regularization — skipping background regularization.")

#     # ─────────────────────────────────────────────
#     # 4. Combine terms
#     # ─────────────────────────────────────────────
#     total_loss = rate_term + log_term + l2_term + l1_term
    
#     # Debug output (commented out for performance)
#     # tf.print("rate:", rate_term,
#     #          "log:", log_term,
#     #          "L2:", l2_term,
#     #          "L1:", l1_term,
#     #          "total:", total_loss)

#     # Return 6 values for backward compatibility with existing code
#     # The calling code expects: loss, rate_term, log_term, _, _, _
#     input_l2_term = tf.constant(0.0, dtype=rate_term.dtype)  # Legacy compatibility
#     return total_loss, rate_term, log_term, l2_term, l1_term, input_l2_term

def compute_adaptive_neuron_thresholds(target_rates, percentile=75.0, target_scaling=1.0, sigma=250.0):
    """
    Compute per-neuron adaptive thresholds from target rates for use with adaptive weighting.
    
    IMPORTANT: This function applies the SAME preprocessing as done in compute_combined_loss2:
    1. Scale by target_scaling and multiply by 1000: target_scaling * target_rates * 1e3
    2. Apply Gaussian smoothing with specified sigma
    
    This ensures thresholds match the actual processed data used in loss computation.
    
    Args:
        target_rates: [time, neurons] numpy array or tensor of target firing rates
        percentile: percentile threshold for each neuron (0-100)
        target_scaling: scaling factor applied to target rates (same as in loss function)
        sigma: standard deviation for Gaussian smoothing (same as in loss function)
    
    Returns:
        thresholds: [neurons] numpy array of threshold values for each neuron
    """
    import numpy as np
    import tensorflow as tf
    
    # Convert to TensorFlow tensor for processing
    if hasattr(target_rates, 'numpy'):
        rates_tensor = target_rates
    else:
        rates_tensor = tf.constant(target_rates, dtype=tf.float32)
    
    # Apply the SAME preprocessing as in compute_combined_loss2:
    # 1. Scale by target_scaling and multiply by 1000
    scaled_rates = target_scaling * rates_tensor * 1e3
    
    # 2. Apply Gaussian smoothing (need to add batch dimension for gaussian_smoothing)
    if scaled_rates.ndim == 2:
        scaled_rates = tf.expand_dims(scaled_rates, 0)  # Add batch dimension
    
    smoothed_rates = gaussian_smoothing(scaled_rates, sigma)
    smoothed_rates = tf.squeeze(smoothed_rates, 0)  # Remove batch dimension
    
    # Convert back to numpy for percentile computation
    processed_rates = smoothed_rates.numpy()
    
    n_neurons = processed_rates.shape[1]
    thresholds = np.zeros(n_neurons, dtype=np.float32)
    
    for i in range(n_neurons):
        neuron_data = processed_rates[:, i]
        non_zero_values = neuron_data[neuron_data > 0]
        
        if len(non_zero_values) > 0:
            thresholds[i] = np.percentile(non_zero_values, percentile)
        else:
            thresholds[i] = 0.0
            
    return thresholds

# Graph mode optimization: Use @tf.function for significant performance improvement
@tf.function
def compute_combined_loss2(
    model,
    model_spikes,                 # [seq_len, n_neurons]
    target_rates,                 # [seq_len, n_neurons]
    *,
    target_scaling: float = 1.0,
    sigma: int = 100.0,
    start_idx: int | None = None,
    end_idx:   int | None = None,
    rate_weight: float = 0.000075,
    log_sum_weight: float = 0.0,
    bkg_l2_weight: float = 0.0,
    rec_l2_weight: float = 0.0,
    input_l1_weight: float = 0.0,
    input_l2_weight: float = 0.0,
    recurrent_l1_weight: float = 0.0,  # NEW: recurrent L1 regularization
    recurrent_l2_weight: float = 0.0,  # NEW: recurrent L2 regularization  
    background_l1_weight: float = 0.0, # NEW: background L1 regularization
    background_l2_weight: float = 0.0, # NEW: background L2 regularization
    rate_loss_type: str = "mse", # NEW: "mse", "log_mse", or "soft_root"
    soft_root_epsilon: float = 0.1,  # NEW: avoid divide-by-zero in soft-root
    # ADAPTIVE NEURON WEIGHTING PARAMETERS
    enable_adaptive_weighting: bool = False,  # NEW: enable adaptive per-neuron weighting
    adaptive_percentile: float = 75.0,        # NEW: percentile threshold for each neuron
    adaptive_high_weight: float = 1.0,        # NEW: weight for above-threshold timepoints
    adaptive_low_weight: float = 0.1,         # NEW: weight for below-threshold timepoints
    adaptive_neuron_thresholds: tf.Tensor | None = None  # NEW: precomputed thresholds [n_neurons]
):
    def _var_by_name(model, key):
        for v in model.trainable_variables:
            if key in v.name:
                return v
        return None

    def clipped_l2(w, tau):
        w_clean = tf.where(tf.math.is_finite(w), w, tf.zeros_like(w))
        excess = tf.nn.relu(tf.abs(w_clean) - tau)
        return tf.nn.l2_loss(excess) * 2.0

    def _batch(x): return x if x.ndim == 3 else tf.expand_dims(x, 0)

    # ─────────────────────────────────────────────
    # 1. Rate-matching loss
    # ─────────────────────────────────────────────
    tar = gaussian_smoothing(_batch(target_scaling * target_rates * 1e3), sigma)
    mod = gaussian_smoothing(_batch(model_spikes                * 1e3), sigma)

    tar = tf.squeeze(tar, 0)[start_idx:end_idx]
    mod = tf.squeeze(mod, 0)[start_idx:end_idx]

    if enable_adaptive_weighting and adaptive_neuron_thresholds is not None:
        # ─── ADAPTIVE PER-NEURON WEIGHTING ─────────────────────────────────────
        # Use precomputed neuron thresholds
        neuron_thresholds = adaptive_neuron_thresholds  # [neurons]
        
        # Create weight matrix: high weight for above-threshold, low weight for below
        # Broadcasting: tar [time, neurons] > neuron_thresholds [neurons] -> [time, neurons]
        weights = tf.where(
            tf.greater(tar, neuron_thresholds[None, :]),  # Broadcast thresholds across time
            adaptive_high_weight,
            adaptive_low_weight
        )
        
        # Apply adaptive weights to squared errors
        if rate_loss_type == "mse":
            squared_errors = tf.square(mod - tar)
            weighted_errors = weights * squared_errors
            rate_term = rate_weight * tf.reduce_mean(weighted_errors) / (target_scaling * target_scaling)
        elif rate_loss_type == "log_mse":
            tar_log = tf.math.log1p(tar)
            mod_log = tf.math.log1p(mod)
            squared_errors = tf.square(mod_log - tar_log)
            weighted_errors = weights * squared_errors
            rate_term = rate_weight * tf.reduce_mean(weighted_errors)
        elif rate_loss_type == "soft_root":
            squared_errors = tf.square(mod - tar) / (tf.abs(tar) + soft_root_epsilon)
            weighted_errors = weights * squared_errors
            rate_term = rate_weight * tf.reduce_mean(weighted_errors)
        else:
            raise ValueError(f"Unrecognized rate_loss_type: {rate_loss_type!r}")
        
    else:
        # ─── STANDARD MSE LOSS (UNCHANGED) ──────────────────────────────────────
        if rate_loss_type == "mse":
            rate_term = rate_weight * tf.reduce_mean(tf.square(mod - tar)) / (target_scaling * target_scaling)
        elif rate_loss_type == "log_mse":
            tar_log = tf.math.log1p(tar)
            mod_log = tf.math.log1p(mod)
            rate_term = rate_weight * tf.reduce_mean(tf.square(mod_log - tar_log))
        elif rate_loss_type == "soft_root":
            rate_term = rate_weight * tf.reduce_mean(tf.square(mod - tar) / (tf.abs(tar) + soft_root_epsilon))
        else:
            raise ValueError(f"Unrecognized rate_loss_type: {rate_loss_type!r}")

    # ─────────────────────────────────────────────
    # 2. Log-sum regularizer (with graph-mode compatible layer access)
    # ─────────────────────────────────────────────
    log_term = tf.constant(0.0, dtype=rate_term.dtype)
    if log_sum_weight > 0.0:
        # Use input_layer by default (try sparse_input_layer if there are issues)
        # Graph-mode compatible approach: use try/except at function level, not in graph
        try:
            layer = model.get_layer("input_layer")
            w = tf.convert_to_tensor(layer.trainable_weights[0])
            w = tf.where(tf.math.is_finite(w), w, tf.zeros_like(w))
            eps = 5e-3
            log_term = 0.01 * log_sum_weight * (
                tf.reduce_mean(tf.math.log(tf.abs(w) + eps)) - tf.math.log(eps)
            )
        except ValueError:
            # Fallback: if input_layer doesn't exist, skip log-sum regularization
            # Note: If you get this error, you may need to change "input_layer" to "sparse_input_layer"
            pass

    # ─────────────────────────────────────────────
    # 3. L1 and L2 regularization for all weight types
    # ─────────────────────────────────────────────
    
    # Initialize all regularization terms
    l1_term = tf.constant(0.0, dtype=rate_term.dtype)
    l2_term = tf.constant(0.0, dtype=rate_term.dtype)
    
    # Helper function for safe L1 regularization (handles positive and negative weights)
    def safe_l1_reg(weights, weight_value):
        """Apply L1 regularization with proper handling of positive/negative weights."""
        if weight_value > 0.0 and weights is not None:
            weights_clean = tf.where(tf.math.is_finite(weights), weights, tf.zeros_like(weights))
            # Cast weight_value to match tensor dtype to avoid type mismatch
            weight_value_tensor = tf.cast(weight_value, dtype=weights_clean.dtype)
            return weight_value_tensor * tf.reduce_sum(tf.abs(weights_clean))
        return tf.constant(0.0, dtype=rate_term.dtype)
    
    # Helper function for safe L2 regularization  
    def safe_l2_reg(weights, weight_value, use_clipped=False, tau=None):
        """Apply L2 regularization, optionally with clipping."""
        if weight_value > 0.0 and weights is not None:
            weights_clean = tf.where(tf.math.is_finite(weights), weights, tf.zeros_like(weights))
            # Cast weight_value to match tensor dtype to avoid type mismatch
            weight_value_tensor = tf.cast(weight_value, dtype=weights_clean.dtype)
            if use_clipped and tau is not None:
                return weight_value_tensor * clipped_l2(weights_clean, tau)
            else:
                return weight_value_tensor * tf.nn.l2_loss(weights_clean) * 2.0
        return tf.constant(0.0, dtype=rate_term.dtype)
    
    # Note: Clipped L2 thresholds are no longer used (switched to regular L2)
    # tau_bkg = 0.04  # REMOVED
    # tau_rec = 0.003  # REMOVED
    
    # ---- INPUT WEIGHTS ----
    w_input = None
    # Use input_layer by default (change to sparse_input_layer if needed)
    try:
        layer = model.get_layer("input_layer")
        w_input = layer.trainable_weights[0]
    except ValueError:
        # Fallback: try sparse_input_layer if input_layer doesn't exist
        try:
            layer = model.get_layer("sparse_input_layer")
            w_input = layer.trainable_weights[0]
        except ValueError:
            pass
    
    if w_input is not None:
        l1_term += safe_l1_reg(w_input, input_l1_weight)
        l2_term += safe_l2_reg(w_input, input_l2_weight)
    # Note: Removed tf.print warning for graph mode compatibility
    
    # ---- RECURRENT WEIGHTS ----
    w_rec = _var_by_name(model, "sparse_recurrent_weights")
    if w_rec is not None:
        # Support both old and new parameter names for backward compatibility
        rec_l1_weight = recurrent_l1_weight  # New parameter name
        rec_l2_weight_combined = rec_l2_weight + recurrent_l2_weight  # Old + new
        
        l1_term += safe_l1_reg(w_rec, rec_l1_weight)
        l2_term += safe_l2_reg(w_rec, rec_l2_weight_combined, use_clipped=False)  # CHANGED: use regular L2
    # Note: Removed tf.print warning for graph mode compatibility
    
    # ---- BACKGROUND WEIGHTS ----
    w_bkg = _var_by_name(model, "rest_of_brain_weights")
    if w_bkg is not None:
        # Support both old and new parameter names for backward compatibility
        bkg_l1_weight = background_l1_weight  # New parameter name
        bkg_l2_weight_combined = bkg_l2_weight + background_l2_weight  # Old + new
        
        l1_term += safe_l1_reg(w_bkg, bkg_l1_weight)
        l2_term += safe_l2_reg(w_bkg, bkg_l2_weight_combined, use_clipped=False)  # CHANGED: use regular L2
    # Note: Removed tf.print warning for graph mode compatibility

    # ─────────────────────────────────────────────
    # 4. Combine terms
    # ─────────────────────────────────────────────
    total_loss = rate_term + log_term + l2_term + l1_term
    
    # Debug output (commented out for performance)
    # tf.print("rate:", rate_term,
    #          "log:", log_term,
    #          "L2:", l2_term,
    #          "L1:", l1_term,
    #          "total:", total_loss)

    # Return 6 values for backward compatibility with existing code
    # The calling code expects: loss, rate_term, log_term, _, _, _
    input_l2_term = tf.constant(0.0, dtype=rate_term.dtype)  # Legacy compatibility
    return total_loss, rate_term, log_term, l2_term, l1_term, input_l2_term

def plot_selected_neuron_rates(model_traces, target_traces, neurons_to_plot, epoch, save_dir):
    """
    Plots the firing rate traces of selected neurons for both the model and target.

    Parameters
    ----------
    model_traces : list of np.ndarray
        List of smoothed model spike rates for the selected neurons.
    target_traces : list of np.ndarray
        List of smoothed target firing rates for the selected neurons.
    neurons_to_plot : list of int
        Indices of neurons to plot.
    epoch : int
        Current epoch number (for saving the figure).
    save_dir : str
        Directory for saving the figure.

    Returns
    -------
    None
    """
    num_neurons = len(neurons_to_plot)
    fig, axes = plt.subplots(num_neurons, 1, figsize=(10, 2 * num_neurons), sharex=True)

    if num_neurons == 1:
        axes = [axes]  # Ensure it is iterable for a single neuron

    for i, neuron_idx in enumerate(neurons_to_plot):
        ax = axes[i]
        time_axis = np.arange(model_traces[0].shape[0])  # Assuming time dimension is consistent

        # Plot model and target rates
        ax.plot(time_axis, target_traces[0][:, i], label="Target Rate", color="black", linestyle="dashed")
        ax.plot(time_axis, model_traces[0][:, i], label="Model Rate", color="blue", alpha=0.7)

        ax.set_title(f"Neuron {neuron_idx}")
        ax.set_ylabel("Rate (Hz)")
        ax.legend()

    axes[-1].set_xlabel("Time (ms)")

    # Save the figure
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"neuron_rates_epoch_{epoch}.png")
    plt.savefig(save_path)
    print(f"Neuron rate plot saved to: {save_path}")

    plt.close(fig)  # Close the plot to prevent memory issues




def pad_to_seq_len(input_data, seq_len):
    """
    Pad input data to match the specified sequence length (seq_len).

    Parameters:
    - input_data (np.ndarray): Array of shape (number of batches, seq_len, number of neurons).
    - seq_len (int): Target sequence length.

    Returns:
    - padded_data (np.ndarray): Input data padded to the specified seq_len.
    """
    # Check input shape
    if input_data.shape[1] == seq_len:
        # No padding needed
        return input_data
    elif input_data.shape[1] < seq_len:
        # Calculate the padding amount
        pad_amount = seq_len - input_data.shape[1]
        # Pad along the time dimension (axis 1) with zeros
        padded_data = np.pad(input_data,
                             ((0, 0), (0, pad_amount), (0, 0)),
                             mode='constant',
                             constant_values=0)
        return padded_data
    else:
        # Truncate to seq_len if input_data is longer
        return input_data[:, :seq_len, :]
    
def plot_loss_progress(step_losses_train, step_losses_test, epoch, step, testing_losses, loss_offset, display=False, save=False, save_dir='model_checkpoints'):
    """
    Plots training and testing loss progress dynamically.

    Parameters:
    - step_losses_train: List of training losses for each step.
    - step_losses_test: List of testing losses for each step.
    - epoch: Current epoch number.
    - step: Current step number within the epoch.
    - testing_losses: List of average test losses for each epoch.
    - display: If True, displays the plot.
    """
    #global loss_offset # Ensure offset remains consistent across calls
    

    # Adjust losses to avoid values <= 0 for log-scale
    step_losses_train = np.array(step_losses_train)
    step_losses_test = np.array(step_losses_test) if step_losses_test else np.array([])
    testing_losses = np.array(testing_losses) if testing_losses else np.array([])

    min_loss_candidates = [np.min(step_losses_train)]
    if len(step_losses_test) > 0:
        min_loss_candidates.append(np.min(step_losses_test))
    if len(testing_losses) > 0:
        min_loss_candidates.append(np.min(testing_losses))
    min_loss = np.min(min_loss_candidates)

    # Update loss offset if needed
    if min_loss <= 0:
        loss_offset = max(loss_offset, 1 - min_loss)

    step_losses_train += loss_offset
    step_losses_test += loss_offset
    testing_losses += loss_offset

    # Determine the number of subplots needed
    num_subplots = 2 if len(step_losses_test) >= 2 else 1
    fig, axes = plt.subplots(num_subplots, 1, figsize=(8, 5), constrained_layout=True)

    # Ensure `axes` is always iterable
    if num_subplots == 1:
        axes = [axes]

    # Training loss subplot
    axes[0].plot(step_losses_train, label="Training Step Loss", color="blue", alpha=0.7)
    axes[0].set_title("Training Loss Over Steps")
    axes[0].set_xlabel("Steps")
    axes[0].set_ylabel("Loss (log-scale)")
    axes[0].set_yscale("log")
    axes[0].set_xticks(np.arange(0, len(step_losses_train), 8))  # Tick at intervals of 8 steps (epochs)
    axes[0].grid(axis="x", linestyle="--", alpha=0.7)
    axes[0].legend()

    # Testing loss subplot (if test losses are available)
    if len(step_losses_test) >= 2:
        test_epochs = np.arange(1, epoch + 2)  # Epochs start at 1
        for i, stim_losses in enumerate(zip(*[iter(step_losses_test)] * 2)):  # Group test losses by 2 (2 stimuli)
            x_vals = [test_epochs[i]] * len(stim_losses)
            axes[1].scatter(x_vals, stim_losses, color="orange", alpha=0.7)

        # Plot average test loss with a line connecting the averages
        test_avg_losses = [np.mean(pair) for pair in zip(*[iter(step_losses_test)] * 2)]
        axes[1].plot(test_epochs[:len(test_avg_losses)], test_avg_losses, label="Average Test Loss", color="green", linewidth=2)

        axes[1].set_title("Testing Loss Over Epochs")
        axes[1].set_xlabel("Epochs")
        axes[1].set_ylabel("Loss (log-scale)")
        axes[1].set_yscale("log")
        axes[1].legend()
        axes[1].grid(True)

    if save:
        save_path = os.path.join(save_dir, f"loss_progress_epoch_{epoch}.png")
        plt.savefig(save_path)
        print(f"Loss plot saved to: {save_path}")
    
    if display:
        clear_output(wait=True)
        plt.show()
    
# def plot_loss_progress2multi(
#     epoch_losses_train,
#     epoch_losses_test,
#     epoch_rate_losses_train,
#     epoch_rate_losses_test,
#     epoch_input_regularizations,
#     epoch_weight_proportions,
#     epoch,
#     display=True,
#     save=False,
#     save_dir='model_checkpoints',
#     file_string='loss_progress'
# ):
#     """
#     Plot training/testing losses, rate losses, input regularization loss,
#     and weight proportions. Allows optional saving and/or display of the plot.

#     Now tracks per-epoch losses instead of per-step losses.
#     """

#     # Create subplots
#     fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
#     fig.suptitle(f"Epoch: {epoch}", fontsize=14)

#     # Plot training loss per epoch
#     axes[0, 0].plot(epoch_losses_train, 'o-', label="Training Loss", color="blue")
#     axes[0, 0].set_title("Training Loss Over Epochs")
#     axes[0, 0].set_xlabel("Epochs")
#     axes[0, 0].set_ylabel("Loss (log-scale)")
#     axes[0, 0].set_yscale("log")
#     axes[0, 0].grid(True)

#     # Plot testing loss per epoch
#     axes[0, 1].plot(epoch_losses_test, 'o-', label="Testing Loss", color="orange")
#     axes[0, 1].set_title("Testing Loss Over Epochs")
#     axes[0, 1].set_xlabel("Epochs")
#     axes[0, 1].set_ylabel("Loss (log-scale)")
#     axes[0, 1].set_yscale("log")
#     axes[0, 1].grid(True)

#     # Plot rate loss (training) per epoch
#     axes[1, 0].plot(epoch_rate_losses_train, '--o', label="Rate Loss (Train)", color="blue")
#     axes[1, 0].set_title("Rate Loss (Training Epochs)")
#     axes[1, 0].set_xlabel("Epochs")
#     axes[1, 0].set_ylabel("Loss (log-scale)")
#     axes[1, 0].set_yscale("log")
#     axes[1, 0].grid(True)

#     # Plot rate loss (testing) per epoch
#     axes[1, 1].plot(epoch_rate_losses_test, '--o', label="Rate Loss (Test)", color="orange")
#     axes[1, 1].set_title("Rate Loss (Testing Epochs)")
#     axes[1, 1].set_xlabel("Epochs")
#     axes[1, 1].set_ylabel("Loss (log-scale)")
#     axes[1, 1].set_yscale("log")
#     axes[1, 1].grid(True)

#     # Plot input regularization loss per epoch
#     axes[2, 0].plot(epoch_input_regularizations, 'o-', label="Input Regularization Loss", color="green")
#     axes[2, 0].set_title("Input Regularization Loss (Epochs)")
#     axes[2, 0].set_xlabel("Epochs")
#     axes[2, 0].set_ylabel("Loss (log-scale)")
#     axes[2, 0].set_yscale("log")
#     axes[2, 0].grid(True)

#     # Save or display
#     if save:
#         os.makedirs(save_dir, exist_ok=True)
#         plt.savefig(os.path.join(save_dir, f"{file_string}_epoch_{epoch}.png"))
#     if display:
#         plt.show()
#     else:
#         plt.close(fig)


# def plot_loss_progress2(
#     step_losses_train,
#     step_losses_test,
#     rate_losses_train,
#     rate_losses_test,
#     input_regularizations,
#     weight_proportions,
#     epoch,
#     display=True,
#     save=False,
#     save_dir='model_checkpoints',
#     file_string='loss_progress'
# ):
#     """
#     Plot training/testing losses, rate losses, input regularization loss,
#     and weight proportions. Allows optional saving and/or display of the plot.

#     Parameters:
#     -----------
#     step_losses_train : list or array
#         Training losses over steps.
#     step_losses_test : list or array
#         Testing losses over epochs.
#     rate_losses_train : list or array
#         Rate losses during training steps.
#     rate_losses_test : list or array
#         Rate losses during testing epochs.
#     input_regularizations : list or array
#         Input regularization losses (e.g. controlling weights).
#     weight_proportions : list or array
#         Weight proportions below threshold.
#     epoch : int
#         Current epoch.
#     display : bool
#         If True, display the figure.
#     save : bool
#         If True, save the figure to 'save_dir' with a file name pattern
#         based on `file_string`, epoch, and step.
#     save_dir : str
#         Directory to save the figure (if save=True).
#     file_string : str
#         Base filename to use when saving the plot.
#     """
#     # Create subplots
#     fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
#     fig.suptitle(f"Epoch: {epoch}", fontsize=14)

#     # Plot training loss (log scale) with points
#     axes[0, 0].plot(step_losses_train, 'o-', label="Training Loss", color="blue")
#     axes[0, 0].set_title("Training Loss Over Steps")
#     axes[0, 0].set_xlabel("Steps")
#     axes[0, 0].set_ylabel("Loss (log-scale)")
#     axes[0, 0].set_yscale("log")
#     axes[0, 0].grid(True)

#     # Plot testing loss (log scale) with points
#     axes[0, 1].plot(step_losses_test, 'o-', label="Testing Loss", color="orange")
#     axes[0, 1].set_title("Testing Loss Over Epochs")
#     axes[0, 1].set_xlabel("Epochs")
#     axes[0, 1].set_ylabel("Loss (log-scale)")
#     axes[0, 1].set_yscale("log")
#     axes[0, 1].grid(True)

#     # Plot rate loss (training) (log scale) with points, dashed line
#     axes[1, 0].plot(rate_losses_train, '--o', label="Rate Loss (Train)", color="blue")
#     axes[1, 0].set_title("Rate Loss (Training Steps)")
#     axes[1, 0].set_xlabel("Steps")
#     axes[1, 0].set_ylabel("Loss (log-scale)")
#     axes[1, 0].set_yscale("log")
#     axes[1, 0].grid(True)

#     # Plot rate loss (testing) (log scale) with points, dashed line
#     axes[1, 1].plot(rate_losses_test, '--o', label="Rate Loss (Test)", color="orange")
#     axes[1, 1].set_title("Rate Loss (Testing Epochs)")
#     axes[1, 1].set_xlabel("Epochs")
#     axes[1, 1].set_ylabel("Loss (log-scale)")
#     axes[1, 1].set_yscale("log")
#     axes[1, 1].grid(True)

#     # Plot input regularization loss (log scale) with points
#     axes[2, 0].plot(input_regularizations, 'o-', label="Input Regularization Loss", color="green")
#     axes[2, 0].set_title("Input Regularization Loss")
#     axes[2, 0].set_xlabel("Steps")
#     axes[2, 0].set_ylabel("Loss (log-scale)")
#     axes[2, 0].set_yscale("log")
#     axes[2, 0].grid(True)

#     # Plot weight proportion below threshold with points
#     axes[2, 1].plot(weight_proportions, 'o-', label="% Weights < Threshold", color="brown")
#     axes[2, 1].set_title("Weight Proportion Below Threshold")
#     axes[2, 1].set_xlabel("Steps")
#     axes[2, 1].set_ylabel("Proportion")
#     axes[2, 1].grid(True)

#     # Optionally save
#     if save:
#         os.makedirs(save_dir, exist_ok=True)
#         filename = f"{file_string}_epoch_{epoch}.png"
#         save_path = os.path.join(save_dir, filename)
#         plt.savefig(save_path)
#         print(f"Loss plot saved to: {save_path}")

#     # Optionally display
#     if display:
#         plt.show()
#     else:
#         plt.close(fig)

def plot_loss_progress2multi(
    epoch_losses_train,
    epoch_losses_test,
    epoch_rate_losses_train,
    epoch_rate_losses_test,
    epoch_input_regularizations,
    epoch_input_regularizations_test,
    epoch_weight_proportions,
    epoch,
    sampled_weights_log=None,
    target_scaling_values=None,  # Add target_scaling tracking
    display=True,
    save=False,
    save_dir='model_checkpoints',
    file_string='loss_progress',
    weight_threshold=1e-10,
    # New parameters for pruning target visualization
    target_final_sparsity=None,
    pruning_stop_epochs_before_end=None,
    num_epochs=None,
    initial_sparsity=0.0
):
    """
    Plot epoch-level training/testing losses, rate losses, target scaling values, weight proportions,
    and optionally, sampled weights over epochs.
    
    If target_scaling_values is provided, it replaces the input regularization plot in the bottom left.
    
    Parameters for pruning target visualization:
    - target_final_sparsity: Target final sparsity (e.g., 0.9 for 90% sparsity)
    - pruning_stop_epochs_before_end: Number of epochs before end to stop pruning
    - num_epochs: Total number of training epochs
    - initial_sparsity: Initial sparsity at epoch 0 (default 0.0)
    """
    import matplotlib.pyplot as plt
    import os
    import numpy as np

    # Determine subplot grid size based on whether weight samples are provided
    if sampled_weights_log is not None:
        fig, axes = plt.subplots(3, 3, figsize=(16, 10), constrained_layout=True)
    else:
        fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)

    fig.suptitle(f"Epoch: {epoch}", fontsize=14)

    # Plot epoch-level training loss
    axes[0, 0].plot(epoch_losses_train, 'o-', label="Training Loss", color="blue")
    axes[0, 0].set_title("Training Loss (Epochs)")
    axes[0, 0].set_xlabel("Epochs")
    axes[0, 0].set_ylabel("Loss (log-scale)")
    axes[0, 0].set_yscale("log")
    axes[0, 0].grid(True)

    # Plot epoch-level testing loss
    axes[0, 1].plot(epoch_losses_test, 'o-', label="Testing Loss", color="orange")
    axes[0, 1].set_title("Testing Loss (Epochs)")
    axes[0, 1].set_xlabel("Epochs")
    axes[0, 1].set_ylabel("Loss (log-scale)")
    axes[0, 1].set_yscale("log")
    axes[0, 1].grid(True)

    # Plot epoch-level training rate loss
    axes[1, 0].plot(epoch_rate_losses_train, 'o-', label="Train Rate Loss", color="blue")
    axes[1, 0].set_title("Rate Loss (Training Epochs)")
    axes[1, 0].set_xlabel("Epochs")
    axes[1, 0].set_ylabel("Loss (log-scale)")
    axes[1, 0].set_yscale("log")
    axes[1, 0].grid(True)

    # Plot epoch-level testing rate loss
    axes[1, 1].plot(epoch_rate_losses_test, 'o-', label="Test Rate Loss", color="orange")
    axes[1, 1].set_title("Rate Loss (Testing Epochs)")
    axes[1, 1].set_xlabel("Epochs")
    axes[1, 1].set_ylabel("Loss (log-scale)")
    axes[1, 1].set_yscale("log")
    axes[1, 1].grid(True)

    # Plot target scaling over epochs (replaces input regularization plot)
    if target_scaling_values is not None and len(target_scaling_values) > 0:
        axes[2, 0].plot(target_scaling_values, 'o-', label="Target Scaling", color="purple")
        axes[2, 0].set_title("Target Scaling (Epochs)")
        axes[2, 0].set_xlabel("Epochs")
        axes[2, 0].set_ylabel("Scaling Factor")
        axes[2, 0].legend()
        axes[2, 0].grid(True)
        
        # Add horizontal line at 1.0 for reference
        axes[2, 0].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Target (1.0)')
        axes[2, 0].legend()
    else:
        # Fallback to input regularization plot if target_scaling_values not provided
        axes[2, 0].plot(epoch_input_regularizations, 'o-', label="Train Input Reg", color="green")
        axes[2, 0].plot(epoch_input_regularizations_test, 'o-', label="Test Input Reg", color="lime")
        axes[2, 0].set_title("Input Regularization (Epochs)")
        axes[2, 0].set_xlabel("Epochs")
        axes[2, 0].set_ylabel("Loss (log-scale)")
        axes[2, 0].set_yscale("log")
        axes[2, 0].legend()
        axes[2, 0].grid(True)

    # Plot weight proportion
    axes[2, 1].plot(epoch_weight_proportions, 'o-', label=f"% Weights < {weight_threshold:.0e}", color="brown")
    
    # Add pruning target line if parameters are provided
    if (target_final_sparsity is not None and 
        pruning_stop_epochs_before_end is not None and 
        num_epochs is not None):
        
        # Calculate target sparsity progression
        pruning_end_epoch = num_epochs - pruning_stop_epochs_before_end
        epochs_array = np.arange(len(epoch_weight_proportions))
        target_sparsity_progression = []
        
        for ep in epochs_array:
            if ep >= pruning_end_epoch:
                # After pruning stops, maintain final target
                target_sparsity_progression.append(target_final_sparsity)
            else:
                # Calculate progressive target using exponential decay formula
                remaining_epochs = pruning_end_epoch - ep
                if remaining_epochs > 0:
                    current_density = 1.0 - initial_sparsity  # Starting density
                    target_density = 1.0 - target_final_sparsity  # Final density
                    
                    # Same formula as in pruning code: density decreases exponentially
                    # density_at_epoch = current_density * retention_factor^epochs_elapsed
                    retention_factor = (target_density / current_density) ** (1.0 / pruning_end_epoch)
                    current_target_density = current_density * (retention_factor ** ep)
                    current_target_sparsity = 1.0 - current_target_density
                    target_sparsity_progression.append(current_target_sparsity)
                else:
                    target_sparsity_progression.append(target_final_sparsity)
        
        # Plot target as dashed line
        axes[2, 1].plot(epochs_array, target_sparsity_progression, '--', 
                       label=f"Pruning Target", color="red", linewidth=2, alpha=0.8)
    
    axes[2, 1].set_title(f"Weight Proportion Below Threshold ({weight_threshold:.0e})")
    axes[2, 1].set_xlabel("Epochs")
    axes[2, 1].set_ylabel("Proportion")
    axes[2, 1].set_ylim([0, 1])
    axes[2, 1].legend()
    axes[2, 1].grid(True)

    # Plot sampled weights (hardcoded)
    if sampled_weights_log is not None:
        print("\n[DEBUG] Plotting sampled weights")
        if not all(k in sampled_weights_log for k in ['aud_in', 'bkg', 'recurrent']):
            print("[WARNING] One or more keys missing from sampled_weights_log:", sampled_weights_log.keys())

        for conn_type in ['aud_in', 'bkg', 'recurrent']:
            if conn_type not in sampled_weights_log:
                print(f"[ERROR] Missing key '{conn_type}' in sampled_weights_log")
                continue
            print(f"[DEBUG] {conn_type} → {len(sampled_weights_log[conn_type])} epochs sampled")

        jitter = lambda n: np.random.uniform(-0.4, 0.4, size=n)

        for epoch_idx, weights in enumerate(sampled_weights_log['aud_in']):
            axes[0, 2].scatter(weights, epoch_idx + jitter(len(weights)), s=2, alpha=0.5)
        axes[0, 2].set_title("aud_in weights")
        axes[0, 2].set_ylabel("Epoch")
        axes[0, 2].set_xscale("log")

        for epoch_idx, weights in enumerate(sampled_weights_log['bkg']):
            axes[1, 2].scatter(weights, epoch_idx + jitter(len(weights)), s=2, alpha=0.5)
        axes[1, 2].set_title("bkg weights")
        axes[1, 2].set_ylabel("Epoch")

        for epoch_idx, weights in enumerate(sampled_weights_log['recurrent']):
            axes[2, 2].scatter(weights, epoch_idx + jitter(len(weights)), s=2, alpha=0.5)
        axes[2, 2].set_title("recurrent weights")
        axes[2, 2].set_xlabel("Weight value")
        axes[2, 2].set_ylabel("Epoch")


    if save:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{file_string}_epoch_{epoch}.png")
        plt.savefig(save_path)
        print(f"Loss plot saved to: {save_path}")

    if display:
        plt.show()
    else:
        plt.close(fig)



def compute_weight_proportions(model,
                               threshold: float = 1e-10,
                               key: str = "sparse_input_weights"):
    """
    Return the fraction of *AUD-IN* weights whose absolute value
    is below `threshold`.

    Parameters
    ----------
    model : tf.keras.Model
        Your RSNN model.
    threshold : float, optional
        Cut-off that defines a “small” weight.  Default = 5 × 10⁻⁵.
    key : str, optional
        Sub-string that uniquely identifies the AUD-IN weight
        tensors (default ``"sparse_input_weights"``).

    Returns
    -------
    List[float]
        One proportion per AUD-IN tensor (usually length 1).
    """
    proportions = []

    for i, var in enumerate(model.trainable_variables):
        if key not in var.name:          # ← skip everything except AUD-IN
            continue

        w = np.abs(var.numpy())          # |w|
        prop = np.mean(w < threshold)    # vectorised
        proportions.append(prop)

        print(f"Variable {i} ({var.name}): "
              f"fraction |w|<{threshold:g} = {prop:.4f}")

    if not proportions:
        print("[compute_weight_proportions] WARNING – no variables matched "
              f'key "{key}".  Returning [0.0].')
        proportions = [0.0]

    return proportions



def initialize_weight_tracking(model):
    """
    Initializes weight tracking by storing initial weights.

    Parameters:
    - model: The neural network model.

    Returns:
    - prev_weights: List of NumPy arrays representing the initial weights.
    - accumulated_changes: List of empty lists to accumulate per-step changes.
    """
    prev_weights = [var.numpy().copy() for var in model.trainable_variables]
    accumulated_changes = [[] for _ in model.trainable_variables]
    return prev_weights, accumulated_changes

def track_step_weight_changes(model, prev_weights, accumulated_changes):
    """
    Tracks proportional weight changes at each step and accumulates them.

    Parameters:
    - model: The neural network model.
    - prev_weights: List of previous weight values before the step.
    - accumulated_changes: List of lists, accumulating stepwise weight changes.

    Returns:
    - updated_weights: List of updated weights after this step.
    """
    for i, var in enumerate(model.trainable_variables):
        current_w = var.numpy()
        proportional_change = np.abs(current_w - prev_weights[i]) / (np.abs(prev_weights[i]) + 1e-8)
        accumulated_changes[i].append(proportional_change.flatten())  # Store changes for this step
        prev_weights[i] = current_w.copy()  # Update previous weights
    return prev_weights

def compute_epoch_summary(accumulated_changes):
    """
    Computes the average weight change per quantile bin over the entire epoch.

    Parameters:
    - accumulated_changes: List of lists with stepwise weight change arrays.

    Returns:
    - epoch_summary: List of aggregated average changes per trainable variable.
    """
    epoch_summary = []
    for changes in accumulated_changes:
        if changes:
            stacked_changes = np.vstack(changes)  # Convert to 2D array [steps, weights]
            avg_change = np.mean(stacked_changes, axis=0)  # Average over steps
            epoch_summary.append(avg_change)
        else:
            epoch_summary.append(None)  # No updates for this variable
    return epoch_summary

def plot_weight_change_summary(epoch_summary, model, epoch, display=True, save=True, save_dir='weight_tracking'):
    """
    Plots weight change summaries at the end of each epoch.

    Parameters:
    - epoch_summary: List of per-variable averaged weight changes.
    - model: The neural network model.
    - epoch: The current epoch number.
    - display: Whether to display the plot.
    - save: Whether to save the plot.
    - save_dir: Directory to save the plot.
    """
    num_vars = len(model.trainable_variables)
    fig, axes = plt.subplots(num_vars, 1, figsize=(8, 6 * num_vars))

    if num_vars == 1:
        axes = [axes]  # Ensure iterable if there's only one variable

    bins = [0, 0.1, 0.2, 0.5, 1, 2]  # % change bins
    labels = ["<10%", "10-20%", "20-50%", "50-100%", ">100%"]

    for i, (var, avg_changes) in enumerate(zip(model.trainable_variables, epoch_summary)):
        if avg_changes is None:
            continue  # Skip if no updates happened

        # Compute quantiles
        weight_values = var.numpy().flatten()
        quantiles = np.percentile(np.abs(weight_values), [25, 50, 75])
        low_q = weight_values < quantiles[0]
        mid_q = (weight_values >= quantiles[0]) & (weight_values < quantiles[2])
        high_q = weight_values >= quantiles[2]

        def categorize_changes(weight_subset):
            """ Categorize weight changes into bins. """
            hist, _ = np.histogram(weight_subset.flatten(), bins=bins)
            return hist

        hist_low = categorize_changes(avg_changes[low_q])
        hist_mid = categorize_changes(avg_changes[mid_q])
        hist_high = categorize_changes(avg_changes[high_q])

        x = np.arange(len(labels))
        width = 0.25

        ax = axes[i]
        ax.bar(x - width, hist_low, width=width, label="Lower 25% Weights", color="blue")
        ax.bar(x, hist_mid, width=width, label="Middle 50% Weights", color="orange")
        ax.bar(x + width, hist_high, width=width, label="Upper 25% Weights", color="red")

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(f"Weight Change Summary - Variable {i} ({var.name})")
        ax.set_xlabel("Proportional Weight Change")
        ax.set_ylabel("Count")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.7)

    if save:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"weight_changes_epoch_{epoch}.png")
        plt.savefig(save_path)
        print(f"Weight change summary saved to: {save_path}")

    if display:
        plt.show()
    else:
        plt.close(fig)


@tf.function
def train_step_chunked(
    batch_input_spikes,    # shape: [batch_size, total_seq_len, n_input]
    batch_target_rates,    # shape: [batch_size, total_seq_len, n_neurons]
    batch_start_indices,   # shape: [batch_size], e.g. int32
    batch_end_indices,     # shape: [batch_size], e.g. int32
    flags,
    extractor_model,       # same as your current code
    state_variables,       # same as your current code (if you need/want it)
    model,                 # your main tf.keras.Model
    optimizer,             # tf.keras.optimizers.Optimizer
    log_sum_weight_adaptive
):
    """
    Performs truncated backprop through time (BPTT) on each sequence in the batch.
    
    - For each item in the batch, we unroll the network in chunks of size `flags.chunk_size`.
    - After each chunk, we apply a gradient update (so we use a fresh GradientTape per chunk).
    - We carry over the final hidden state from chunk k to chunk k+1, so that the model
      experiences the time continuity of the full sequence. We do *not* retain the large
      computation graph from earlier chunks in memory, thus saving GPU RAM.

    Args:
        batch_input_spikes: float tensor [batch_size, total_seq_len, n_input]
        batch_target_rates: float tensor [batch_size, total_seq_len, n_neurons]
        batch_start_indices: int tensor [batch_size], e.g. 0 for your code
        batch_end_indices:   int tensor [batch_size], e.g. the final time index for each stimulus
        flags: An object containing your training flags, including `chunk_size`.
        extractor_model: Your model-for-extraction that returns `[RSNN output, final model output]`.
        state_variables: Hidden states (if you use them explicitly), or can be ignored if not needed.
        model: The main tf.keras.Model you’re training (same model used inside extractor_model).
        optimizer: Your chosen optimizer.
        log_sum_weight_adaptive: The floating scalar you pass to `compute_combined_loss2`.
    Returns:
        batch_loss: Mean loss across the entire batch (summed over chunks).
        batch_rate_loss: Mean rate-based loss across the entire batch.
        batch_input_reg: Mean input regularization across the entire batch.
    """
    
    # For convenience, grab the RSNN layer if you need to zero/init the hidden state
    rsnn_layer = model.get_layer("rsnn")
    
    batch_size = tf.shape(batch_input_spikes)[0]
    chunk_size = flags.chunk_size
    
    # Initialize accumulators for the entire batch
    total_loss = tf.constant(0.0, dtype=tf.float32)
    total_rate_loss = tf.constant(0.0, dtype=tf.float32)
    total_input_reg = tf.constant(0.0, dtype=tf.float32)
    
    # Loop over items in the batch
    for i in tf.range(batch_size):
        # Extract the per-item spikes and target
        input_spikes_i = batch_input_spikes[i]   # shape: [total_seq_len, n_input]
        target_rates_i = batch_target_rates[i]   # shape: [total_seq_len, n_neurons]
        start_idx_i = batch_start_indices[i]
        end_idx_i   = batch_end_indices[i]
        
        # We'll accumulate the losses across chunks for this single sequence
        sum_loss_i = tf.constant(0.0, dtype=tf.float32)
        sum_rate_loss_i = tf.constant(0.0, dtype=tf.float32)
        sum_input_reg_i = tf.constant(0.0, dtype=tf.float32)
        
        # Reset hidden state for this sequence i (batch-size=1 scenario, or you can do your actual batch_size if needed)
        running_state = rsnn_layer.cell.zero_state(1, dtype=tf.float32)
        
        # We'll move in steps of `chunk_size` from start_idx_i up to end_idx_i
        t0 = start_idx_i
        while t0 < end_idx_i:
            t1 = tf.minimum(t0 + chunk_size, end_idx_i)
            
            # Use a separate GradientTape for each chunk so we can apply an update right away
            with tf.GradientTape() as tape:
                # Slice out the chunk [t0 : t1] for input + target
                chunk_input_spikes_i = input_spikes_i[t0 : t1]  # shape: [chunk_len, n_input]
                chunk_target_rates_i = target_rates_i[t0 : t1]  # shape: [chunk_len, n_neurons]
                
                # Expand dims so the model sees it as batch=1 for that chunk
                chunk_input_spikes_i = tf.expand_dims(chunk_input_spikes_i, axis=0)  # [1, chunk_len, n_input]
                chunk_target_rates_i = tf.expand_dims(chunk_target_rates_i, axis=0)  # [1, chunk_len, n_neurons]
                
                # Roll out the model on this chunk. 
                # NOTE: If your `tu.roll_out` function does not currently accept an initial state,
                # you need to add that. For example, you might define:
                #
                #    tu.roll_out(spikes, flags, extractor_model, state_variables, initial_state=running_state)
                #
                # so that the model continues from the previous hidden state. 
                rollout_results = tu.roll_out(
                    chunk_input_spikes_i, 
                    flags, 
                    extractor_model, 
                    state_variables, 
                    initial_state=running_state
                )
                
                # Extract the chunk's spikes from the rollout
                # shape: [1, chunk_len, n_neurons], typically
                spikes_chunk = rollout_results["spikes"]
                
                # Compute the loss for just this chunk. We'll treat the chunk’s local time as [0 .. chunk_len]
                chunk_len = t1 - t0
                loss_chunk, rate_loss_chunk, input_reg_chunk = tu.compute_combined_loss2(
                    model,
                    spikes_chunk[0],          # shape: [chunk_len, n_neurons]
                    chunk_target_rates_i[0],  # shape: [chunk_len, n_neurons]
                    start_idx=0,
                    end_idx=chunk_len,        # i.e. use all time steps in the chunk
                    target_scaling=flags.target_scaling,
                    sigma=flags.sigma_smoothing,
                    log_sum_weight=log_sum_weight_adaptive,
                    rate_weight=flags.rate_weight
                )
            
            # Apply gradients for this chunk
            gradients = tape.gradient(loss_chunk, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            # Accumulate chunk-wise losses for item i
            sum_loss_i      += loss_chunk
            sum_rate_loss_i += rate_loss_chunk
            sum_input_reg_i += input_reg_chunk
            
            # Update `running_state` so next chunk starts where this one ended
            if "final_state" in rollout_results:
                running_state = rollout_results["final_state"]
            
            # Move t0 -> t1
            t0 = t1
        
        # After processing all chunks for item i, add to the batch-level total
        total_loss      += sum_loss_i
        total_rate_loss += sum_rate_loss_i
        total_input_reg += sum_input_reg_i
    
    # Compute the final average *per sequence* in the batch
    # (You could also divide by total number of chunks if you prefer a chunk-wise average.)
    batch_loss = total_loss / tf.cast(batch_size, tf.float32)
    batch_rate_loss = total_rate_loss / tf.cast(batch_size, tf.float32)
    batch_input_reg = total_input_reg / tf.cast(batch_size, tf.float32)
    
    return batch_loss, batch_rate_loss, batch_input_reg


#### CURRENTLY UNUSED FUNCTIONS ####

def partial_reversion(model, optimizer, current_checkpoint, best_checkpoint, alpha):
    """
    Perform partial reversion of model weights and optimizer state
    
    Args:
        model: The current model
        optimizer: The current optimizer
        current_checkpoint: Checkpoint containing current weights/state
        best_checkpoint: Checkpoint containing best weights/state
        alpha: Mixing factor (0 = keep current, 1 = use best)
    """
    # Mix model weights using checkpointed weights
    for curr_weights, best_weights in zip(current_checkpoint.weights, best_checkpoint.weights):
        mixed_weights = alpha * best_weights + (1 - alpha) * curr_weights
        var_idx = len(model.trainable_variables) - len(current_checkpoint.weights)
        model.trainable_variables[var_idx].assign(mixed_weights)
    
    # Mix optimizer state using checkpointed states
    for curr_state, best_state in zip(current_checkpoint.optimizer_state, best_checkpoint.optimizer_state):
        mixed_state = alpha * best_state + (1 - alpha) * curr_state
        var_idx = len(optimizer.variables()) - len(current_checkpoint.optimizer_state)
        optimizer.variables()[var_idx].assign(mixed_state)


# ═══════════════════════════════════════════════════════════════════════════════
# SPIKE PADDING FOR GAUSSIAN SMOOTHING EDGE EFFECT MITIGATION
# ═══════════════════════════════════════════════════════════════════════════════

def gaussian_smooth_with_spike_padding(spikes, sigma, padding_before=None, padding_after=None):
    """
    Enhanced Gaussian smoothing with spike padding to mitigate edge effects.
    
    Args:
        spikes: Spike tensor [time, neurons] - the main chunk to be smoothed
        sigma: Gaussian smoothing standard deviation
        padding_before: Optional spike tensor to pad before the main chunk
        padding_after: Optional spike tensor to pad after the main chunk
        
    Returns:
        smoothed_rates: Gaussian-smoothed rates for the main chunk only
    """
    import tensorflow as tf
    
    # If no padding provided, fall back to standard smoothing
    if padding_before is None and padding_after is None:
        # Add batch dimension, smooth, then remove batch dimension
        spikes_batched = tf.expand_dims(spikes, axis=0)  # [1, time, neurons]
        smoothed_batched = gaussian_smoothing(spikes_batched, sigma)
        return smoothed_batched[0]  # Remove batch dimension
    
    # Construct padded spike sequence
    spike_segments = []
    
    if padding_before is not None:
        spike_segments.append(tf.stop_gradient(padding_before))
    
    spike_segments.append(spikes)  # Main chunk - keeps gradients
    
    if padding_after is not None:
        spike_segments.append(tf.stop_gradient(padding_after))
    
    # Concatenate all segments
    padded_spikes = tf.concat(spike_segments, axis=0)  # [padded_time, neurons]
    
    # Add batch dimension for gaussian_smoothing
    padded_spikes_batched = tf.expand_dims(padded_spikes, axis=0)  # [1, padded_time, neurons]
    
    # Apply Gaussian smoothing to the full padded sequence
    smoothed_padded_batched = gaussian_smoothing(padded_spikes_batched, sigma)
    smoothed_padded = smoothed_padded_batched[0]  # Remove batch dimension: [padded_time, neurons]
    
    # Extract only the main chunk portion
    before_length = tf.shape(padding_before)[0] if padding_before is not None else 0
    after_length = tf.shape(padding_after)[0] if padding_after is not None else 0
    main_length = tf.shape(spikes)[0]
    
    # Extract the smoothed rates for the main chunk only
    start_idx = before_length
    end_idx = before_length + main_length
    
    smoothed_main = smoothed_padded[start_idx:end_idx, :]
    
    return smoothed_main


def compute_combined_loss2_with_spike_padding(
    model_spikes, target_rates, input_l1_weight=0.0, input_l2_weight=0.0,
    recurrent_l1_weight=0.0, recurrent_l2_weight=0.0, background_l1_weight=0.0, background_l2_weight=0.0,
    rate_weight=1.0, log_sum_weight=0.0, target_scaling=1.0, sigma=200.0,
    POOL_MAT=None, enable_pooling=True, 
    padding_before=None, padding_after=None,
    adaptive_neuron_thresholds=None, model=None):
    """
    Enhanced loss computation with spike padding for better Gaussian smoothing.
    
    This version uses precomputed spikes from state rollouts to pad the main chunk,
    providing more realistic temporal context for edge-effect-free smoothing.
    
    Args:
        model_spikes: Spike tensor [time, neurons] for the main chunk
        target_rates: Target rate tensor [time, neurons]
        padding_before: Spike tensor to pad before main chunk (from state precomputation)
        padding_after: Spike tensor to pad after main chunk (from state precomputation)
        ... (other args same as compute_combined_loss2)
        
    Returns:
        Same as compute_combined_loss2: loss, rate_term, log_term, pooled_rates, target_rates, model_rates
    """
    import tensorflow as tf
    
    # Apply enhanced Gaussian smoothing with spike padding
    model_rates = gaussian_smooth_with_spike_padding(
        model_spikes, sigma, padding_before, padding_after
    )
    
    # Rest of loss computation is identical to original function
    # Pool model rates if pooling is enabled
    if enable_pooling and POOL_MAT is not None:
        # model_rates: [time, neurons] → [time, target_groups]
        pooled_rates = tf.matmul(model_rates, POOL_MAT)
    else:
        # Direct 1-to-1 comparison
        pooled_rates = model_rates
    
    # Ensure target rates have correct scaling
    scaled_target_rates = target_rates * target_scaling
    
    # Adaptive neuron weighting (if enabled)
    if adaptive_neuron_thresholds is not None:
        # Compute per-neuron weights based on target rate vs threshold
        # pooled_rates shape: [time, neurons], thresholds shape: [neurons]
        above_threshold = tf.cast(
            tf.reduce_mean(scaled_target_rates, axis=0) > adaptive_neuron_thresholds,
            tf.float32
        )
        # Weight: 1.0 for above threshold, 0.1 for below threshold neurons
        neuron_weights = above_threshold + 0.1 * (1.0 - above_threshold)
        
        # Apply weights to the rate loss computation
        rate_diff = pooled_rates - scaled_target_rates
        weighted_rate_diff = rate_diff * neuron_weights[None, :]  # Broadcast over time
        rate_term = tf.reduce_mean(tf.square(weighted_rate_diff))
    else:
        # Standard MSE loss
        rate_term = tf.reduce_mean(tf.square(pooled_rates - scaled_target_rates))
    
    # Log-sum regularization (sparsity encouragement)
    if log_sum_weight > 0.0:
        epsilon = 1e-8
        log_term = log_sum_weight * tf.reduce_mean(tf.math.log(pooled_rates + epsilon))
    else:
        log_term = tf.constant(0.0, dtype=tf.float32)  # Ensure it's a TensorFlow tensor
    
    # L1/L2 regularization on model weights
    reg_term = tf.constant(0.0, dtype=tf.float32)  # Ensure it's a TensorFlow tensor
    if model is not None:
        for var in model.trainable_variables:
            var_name = var.name.lower()
            
            if "sparse_input_weights" in var_name or "input" in var_name:
                if input_l1_weight > 0.0:
                    reg_term += input_l1_weight * tf.reduce_sum(tf.abs(var))
                if input_l2_weight > 0.0:
                    reg_term += input_l2_weight * tf.reduce_sum(tf.square(var))
                    
            elif "sparse_recurrent_weights" in var_name or "recurrent" in var_name:
                if recurrent_l1_weight > 0.0:
                    reg_term += recurrent_l1_weight * tf.reduce_sum(tf.abs(var))
                if recurrent_l2_weight > 0.0:
                    reg_term += recurrent_l2_weight * tf.reduce_sum(tf.square(var))
                    
            elif "rest_of_brain_weights" in var_name or "background" in var_name:
                if background_l1_weight > 0.0:
                    reg_term += background_l1_weight * tf.reduce_sum(tf.abs(var))
                if background_l2_weight > 0.0:
                    reg_term += background_l2_weight * tf.reduce_sum(tf.square(var))
    
    # Combine all terms
    total_loss = rate_term + log_term + reg_term
    
    return total_loss, rate_term, log_term, pooled_rates, scaled_target_rates, model_rates


def compute_combined_loss2_with_presmoothed_targets(
    model, model_spikes, target_rates_presmoothed,
    enable_pooling=True, POOL_MAT=None,
    sigma=150, start_idx=0, end_idx=-1,
    rate_weight=1.0, log_sum_weight=1e-7, target_scaling=1.0,
    input_l1_weight=0.0, input_l2_weight=0.0,
    recurrent_l1_weight=0.0, recurrent_l2_weight=0.0,
    background_l1_weight=0.0, background_l2_weight=0.0,
    padding_before=None, padding_after=None):
    """
    Enhanced loss computation with pre-smoothed target rates and spike padding.
    
    Key advantages:
    - Eliminates target rate edge effects by using pre-smoothed targets
    - Still uses spike padding for model rates to eliminate model edge effects
    - More efficient as target smoothing is done once per stimulus, not per chunk
    
    Args:
        model: The neural network model
        model_spikes: Model spike tensor [time, neurons] for the current chunk
        target_rates_presmoothed: Pre-smoothed target rates [time, neurons] for the current chunk
        enable_pooling: Whether to use neuron pooling
        POOL_MAT: Pooling matrix if pooling is enabled
        sigma: Gaussian smoothing sigma for model spikes
        start_idx, end_idx: Time window for loss computation
        rate_weight: Weight for rate alignment loss
        log_sum_weight: Weight for log-sum regularization
        input_l1_weight, input_l2_weight: L1/L2 weights for input regularization
        recurrent_l1_weight, recurrent_l2_weight: L1/L2 weights for recurrent regularization
        background_l1_weight, background_l2_weight: L1/L2 weights for background regularization
        padding_before: Spike tensor to pad before main chunk (from state precomputation)
        padding_after: Spike tensor to pad after main chunk (from state precomputation)
        
    Returns:
        total_loss: Combined loss value
        rate_term: Rate alignment loss component
        log_term: Log-sum regularization component
        pooled_rates: Model rates after pooling (if enabled)
        target_rates_used: Target rates used in comparison
        model_rates: Model rates after smoothing
    """
  # import tensorflow as tf
    
    # Apply enhanced Gaussian smoothing with spike padding to model spikes
    # Scale model spikes to Hz first (same as original function)
    model_spikes_hz = model_spikes * 1000.0
    
    # Scale padding spikes to Hz if provided
    padding_before_hz = padding_before * 1000.0 if padding_before is not None else None
    padding_after_hz = padding_after * 1000.0 if padding_after is not None else None
    
    model_rates = gaussian_smooth_with_spike_padding(
        model_spikes_hz, sigma, padding_before_hz, padding_after_hz
    )
    
    # Target rates are already pre-smoothed and scaled - just extract time window
    if end_idx == -1:
        target_rates_used = target_rates_presmoothed[start_idx:]
        model_rates_windowed = model_rates[start_idx:]
    else:
        target_rates_used = target_rates_presmoothed[start_idx:end_idx]
        model_rates_windowed = model_rates[start_idx:end_idx]
    
    # Pool model rates if pooling is enabled
    if enable_pooling and POOL_MAT is not None:
        # model_rates_windowed: [time, neurons] → [time, target_groups]
        pooled_rates = tf.matmul(model_rates_windowed, POOL_MAT)
    else:
        # Direct 1-to-1 comparison
        pooled_rates = model_rates_windowed
    
    # Rate alignment loss (MSE between log-scaled smoothed model rates and pre-smoothed targets)
    #eps = 0.1  # Small epsilon to prevent log(0)
    #rate_mse = tf.reduce_mean(tf.square(tf.math.log(pooled_rates + eps) - tf.math.log(target_rates_used + eps)))
    ## Rate alignment loss (MSE between smoothed model rates and pre-smoothed targets)
    rate_mse = tf.reduce_mean(tf.square(pooled_rates - target_rates_used))
    # Apply same normalization as original function for consistency
    rate_term = rate_weight * rate_mse / (target_scaling * target_scaling)
    
    # Log-sum regularization on trainable variables
    if log_sum_weight > 0.0 and len(model.trainable_variables) > 0:
        first_var = model.trainable_variables[0]
        log_term = log_sum_weight * tf.math.log(tf.reduce_sum(tf.abs(first_var)) + 1e-8)
    else:
        log_term = tf.constant(0.0, dtype=tf.float32)  # Ensure it's a TensorFlow tensor
    
    # L1/L2 regularization on different weight types
    reg_term = tf.constant(0.0, dtype=tf.float32)  # Ensure it's a TensorFlow tensor
    if any([input_l1_weight, input_l2_weight, recurrent_l1_weight, recurrent_l2_weight, 
            background_l1_weight, background_l2_weight]):
        
        for var in model.trainable_variables:
            var_name = var.name.lower()
            
            if "sparse_input_weights" in var_name or "input" in var_name:
                if input_l1_weight > 0.0:
                    reg_term += input_l1_weight * tf.reduce_sum(tf.abs(var))
                if input_l2_weight > 0.0:
                    reg_term += input_l2_weight * tf.reduce_sum(tf.square(var))
                    
            elif "sparse_recurrent_weights" in var_name or "recurrent" in var_name:
                if recurrent_l1_weight > 0.0:
                    reg_term += recurrent_l1_weight * tf.reduce_sum(tf.abs(var))
                if recurrent_l2_weight > 0.0:
                    reg_term += recurrent_l2_weight * tf.reduce_sum(tf.square(var))
                    
            elif "rest_of_brain_weights" in var_name or "background" in var_name:
                if background_l1_weight > 0.0:
                    reg_term += background_l1_weight * tf.reduce_sum(tf.abs(var))
                if background_l2_weight > 0.0:
                    reg_term += background_l2_weight * tf.reduce_sum(tf.square(var))
    
    # Combine all terms
    total_loss = rate_term + log_term + reg_term
    
    return total_loss, rate_term, log_term, pooled_rates, target_rates_used, model_rates_windowed