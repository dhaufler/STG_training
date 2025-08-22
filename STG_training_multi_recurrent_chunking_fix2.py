# stg_training_chunked.py – chunked STG trainer (minimal diffs vs your earlier code)
# SPDX-License-Identifier: MIT

import os, random, pickle, ctypes.util, gc, math, socket, json, time
from time import time as time_func

import numpy as np
import pandas as pd
import tensorflow as tf
import absl

from v1_model_utils import load_sparse, models
from general_utils import training_utils as tu
from general_utils import troubleshooting_utils as tblu

dtype = tf.float32

# ───── GPU helper ─────────────────────────────────────────────────────────────

def get_gpu_memory():
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return
    try:
        info = tf.config.experimental.get_memory_info("GPU:0")
        #print(f"GPU Memory: {info['current']/2**20:.1f}/{info['peak']/2**20:.1f} MB")
    except Exception as e:
        print("tf mem-info failed:", e)
    os.system("nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv | head -n2")

# ───── chunk samplers (non-overlapping) ──────────────────────────────────────

def make_chunk_sampler(total_len_ms: int, chunk_ms: int):
    """Generate non-overlapping chunks of fixed size."""
    current_pos = 0
    while current_pos < total_len_ms:
        end_pos = min(current_pos + chunk_ms, total_len_ms)
        if end_pos - current_pos >= 10:  # Minimum chunk size
            yield current_pos, end_pos
        current_pos = end_pos

def make_discrete_random_chunk_sampler(total_len_ms: int, chunk_options: list):
    """
    Generate non-overlapping chunks using discrete random selection from chunk_options.
    
    Args:
        total_len_ms: Total stimulus duration
        chunk_options: List of possible chunk lengths (e.g., [400, 500, 600])
    
    Yields:
        (start, end) tuples for each chunk
    """
    if len(chunk_options) == 1:
        # Single chunk size: use fixed chunking
        chunk_ms = int(chunk_options[0])
        current_pos = 0
        while current_pos < total_len_ms:
            end_pos = min(current_pos + chunk_ms, total_len_ms)
            if end_pos - current_pos >= 10:  # Minimum chunk size
                yield current_pos, end_pos
            current_pos = end_pos
        return
    
    # Multiple chunk sizes: random selection
    chunk_lengths = [int(x) for x in chunk_options]
    current_pos = 0
    
    while current_pos < total_len_ms:
        remaining_len = total_len_ms - current_pos
        
        # Filter chunk options that fit in remaining space
        valid_chunks = [c for c in chunk_lengths if c <= remaining_len and c >= 10]
        
        if not valid_chunks:
            # Use remaining space if no valid chunks fit
            if remaining_len >= 10:
                yield current_pos, total_len_ms
            break
        
        # Randomly select from valid chunks
        selected_chunk = np.random.choice(valid_chunks)
        
        # Handle last chunk: use all remaining space if close to selected size
        if remaining_len - selected_chunk < 50:  # Within 50ms, just use remaining
            yield current_pos, total_len_ms
            break
        else:
            yield current_pos, current_pos + selected_chunk
            current_pos += selected_chunk

def get_sigma_smoothing(flags, training=True):
    """
    Get the appropriate sigma smoothing value based on training mode and flags.
    
    Args:
        flags: Command line flags
        training: Whether this is for training (True) or testing/plotting (False)
    
    Returns:
        float: Sigma smoothing value to use
    """
    if not training:
        # Always use fixed test sigma for testing and plotting
        return flags.sigma_smoothing_test
    
    # For training: check sigma_smoothing format
    if len(flags.sigma_smoothing) == 1:
        # Single value: use fixed sigma for training
        return float(flags.sigma_smoothing[0])
    elif len(flags.sigma_smoothing) == 2:
        # Two values: randomize between min and max
        min_sigma = float(flags.sigma_smoothing[0])
        max_sigma = float(flags.sigma_smoothing[1])
        return np.random.uniform(min_sigma, max_sigma)
    else:
        # Invalid format: fallback to first value with warning
        print(f"WARNING: sigma_smoothing should have 1 or 2 values, got {len(flags.sigma_smoothing)}. Using first value.")
        return float(flags.sigma_smoothing[0])

# Log sampled weights
def sample_weights(model, sample_size=1000):
    sampled = {'aud_in': [], 'bkg': [], 'recurrent': []}
    for var in model.trainable_variables:
        name = var.name
        flat = var.numpy().flatten()
        sample = np.random.choice(flat, size=min(sample_size, flat.size), replace=False)
        if "sparse_input_weights" in name:
            sampled['aud_in'] = sample
        elif "rest_of_brain_weights" in name:
            sampled['bkg'] = sample
        elif "sparse_recurrent_weights" in name:
            sampled['recurrent'] = sample
    return sampled

def _sanitize_weights(var):
    """Set NaN/Inf entries of `var` to zero IN-PLACE."""
    clean = tf.where(tf.math.is_finite(var), var, tf.zeros_like(var))
    var.assign(clean)

def _percentile_threshold(x, keep_frac):
    """
    Return magnitude threshold so that a fraction `keep_frac`
    of the *non-masked* entries are kept.
    """
    flat = tf.reshape(tf.abs(x), [-1])                 # 1-D

    size_f  = tf.cast(tf.size(flat), tf.float32)       # float
    k_float = size_f * keep_frac                       # how many to keep
    k_int   = tf.cast(tf.math.ceil(k_float), tf.int32) # at least 1

    # if k_int == 0 (possible when keep_frac == 0) return +inf
    def _empty(): return tf.constant(np.float32(np.inf), flat.dtype)
    def _non_empty():
        topk = tf.math.top_k(flat, k=k_int, sorted=False).values
        return tf.reduce_min(topk)

    return tf.cond(k_int > 0, _non_empty, _empty)


# ───── MAIN ───────────────────────────────────────────────────────────────────

def main(_):
    os.environ.update({
        "TF_CPP_MIN_LOG_LEVEL": "2",
        "TF_ENABLE_ONEDNN_OPTS": "1",
        "TF_GPU_THREAD_MODE": "gpu_private",
        "TF_GPU_ALLOCATOR": "cuda_malloc_async",
    })
    for d in tf.config.list_physical_devices("GPU"):
        try: tf.config.experimental.set_memory_growth(d, True)
        except: pass

    print("GPUs:", tf.config.list_physical_devices("GPU"))
    get_gpu_memory()

    flags = absl.app.flags.FLAGS

    flags.net_dir = os.path.join(flags.base_dir, flags.net_dir)
    flags.input_dir = os.path.join(flags.net_dir, flags.input_dir)
    flags.target_rates_dir = os.path.join(flags.net_dir, flags.target_rates_dir)
    flags.checkpoint_loaddir = os.path.join(flags.net_dir, flags.checkpoint_loaddir)
    flags.checkpoint_savedir = os.path.join(flags.net_dir, flags.checkpoint_savedir)

    random.seed(flags.seed)
    np.random.seed(flags.seed)
    tf.random.set_seed(flags.seed)

    strategy = tf.distribute.OneDeviceStrategy("/gpu:0") if flags.strategy=="standard" else tf.distribute.MirroredStrategy()

    with strategy.scope():
        network, lgn_input, bkg_input = load_sparse.load_stg(flags=flags, n_neurons=flags.neurons)
        
        # ─── POOLING SET-UP ─────────────────────────────────────────────
        target_cell_ids = np.asarray(network["target_cell_id"])          # length = N
        
        if flags.enable_pooling:
            # Standard pooling: group model neurons by target_cell_id
            uniq_ids, grp_idx = np.unique(target_cell_ids, return_inverse=True)
            n_targets  = len(uniq_ids)
            UNPOOL_MAP = tf.constant(grp_idx, dtype=tf.int32)   # shape (N,)
            
            # weight = 1 / (# neurons in the group)  ⇒ group mean
            counts     = np.bincount(grp_idx).astype(np.float32)
            W_pool_np  = np.zeros((len(target_cell_ids), n_targets), dtype=np.float32)
            W_pool_np[np.arange(len(target_cell_ids)), grp_idx] = 1.0 / counts[grp_idx]
            
            POOL_MAT   = tf.constant(W_pool_np, dtype=tf.float32)            # (N, n_targets)
            print(f"✔ pooling matrix built: {len(target_cell_ids)} → {n_targets} targets")
            
            uniq_target_ids = uniq_ids         
            n_targets      = len(uniq_target_ids)
        else:
            # Direct 1-to-1 comparison: no pooling
            n_targets = len(target_cell_ids)
            UNPOOL_MAP = None
            POOL_MAT = None
            uniq_target_ids = target_cell_ids  # Use all neurons as unique targets
            print(f"✔ direct 1-to-1 comparison enabled: {n_targets} neurons (no pooling)")
        
        _cached_plot_orders = {}       # maps stim-ID  →  order list
        # ────────────────────────────────────────────────────────────────

        model = models.create_model(
            network, lgn_input, bkg_input, data_dir=flags.net_dir, seq_len=None,
            n_input=flags.n_input, n_output=flags.n_output, batch_size=flags.batch_size, dtype=dtype,
            input_weight_scale=1.0, dampening_factor=0.5, recurrent_dampening_factor=0.5,
            gauss_std=0.3, lr_scale=100.0,
            train_input=flags.train_input, train_recurrent=flags.train_recurrent,
            neuron_output=flags.neuron_output, pseudo_gauss=flags.pseudo_gauss,
            use_state_input=False, return_state=True, hard_reset=flags.hard_reset, add_metric=True, max_delay=5,
        )
        model.build((1, None, flags.n_input))
        # ─── FEED-FORWARD: hard-zero recurrent weights ─────────────────────
        if flags.feedforward:
            # w_rec lives inside the RSNN cell; leave it allocated but clamp it to 0
            rsnn_cell = model.get_layer("rsnn").cell
            if hasattr(rsnn_cell, "w_rec"):          # most code paths name it this way
                rsnn_cell.w_rec.assign(tf.zeros_like(rsnn_cell.w_rec))
                tf.print("[feedforward]  w_rec matrix zeroed.")
            # defensive fallback in case the variable has a different name
            for v in model.variables:                # includes *non-*trainables
                if "sparse_recurrent_weights" in v.name:
                    v.assign(tf.zeros_like(v))
        # ─────────────────────────────────────────────────────────────────

        MASKS = {}     # variable -> binary mask  (0 = frozen, 1 = trainable)

        # Note: Mask initialization moved to after checkpoint loading

        # ═══════════════════════════════════════════════════════════════════
        # CONFIGURATION SUMMARY
        # ═══════════════════════════════════════════════════════════════════
        print("\n" + "="*70)
        print("TRAINING CONFIGURATION SUMMARY")
        print("="*70)
        print(f"Model neurons: {flags.neurons}")
        print(f"Target comparison mode: {'POOLED (grouped)' if flags.enable_pooling else 'DIRECT (1-to-1)'}")
        if flags.enable_pooling:
            print(f"   └─ {len(target_cell_ids)} neurons → {n_targets} target groups")
        else:
            print(f"   └─ {n_targets} neurons ↔ {n_targets} targets (direct)")
        
        # Show chunking configuration
        chunk_info = f"🔄 Chunking: {flags.chunk_len_ms}"
        if len(flags.chunk_len_ms) > 1:
            chunk_info += " (discrete random selection)"
        else:
            chunk_info += f" ms (fixed chunks)"
        print(chunk_info)
        
        # Show sigma smoothing configuration
        sigma_info = f"🌊 Smoothing σ: train={flags.sigma_smoothing}"
        if len(flags.sigma_smoothing) > 1:
            sigma_info += " (randomized)"
        else:
            sigma_info += " (fixed)"
        sigma_info += f", test={flags.sigma_smoothing_test}"
        print(sigma_info)
        
        print(f"Learning rates: input={flags.learning_rate_input}")
        print(f"Random seed: {flags.seed}")
        print("="*70 + "\n")

        print("\n🔍 Model trainable variables:")
        for var in model.trainable_variables:
            print(f"{var.name:<40} shape={var.shape}")

        # ---- Set initial weights here ----
        # Set STG → STG recurrent weights to zero
        for var in model.trainable_variables:
            if "sparse_recurrent_weights" in var.name:
                var.assign(tf.zeros_like(var))
                print(f"Zeroed out: {var.name}")
        print("Resetting weights using tu.randomize_weights...")

        #aud_in_median = tu.randomize_weights(model, input_type="aud_in", shape=0.5, scale=0.0005, zero_percentage=0.0)
        #bkg_median = tu.randomize_weights(model, input_type="bkg", shape=0.5, scale=0.0008, zero_percentage=0.0)
        aud_in_median = tu.randomize_weights(
            model,
            input_type="aud_in",
            shape=0.5,
            scale=0.00001, #was 0.001
            zero_percentage=0.0,      # keep if you like
            neg_fraction=0.0          # 0 % negative
        )

        bkg_median = tu.randomize_weights(
            model,
            input_type="bkg",
            shape=0.5,
            scale=0.0005,             # same *base* scale …
            bkg_multiplier=4.0,       
            zero_percentage=0.0
        )
        print(f"AUD_IN median weight after init: {aud_in_median:.6f}")
        print(f"BKG median weight after init: {bkg_median:.6f}")
        # ---- Done setting weights ----

        optimizer = tf.keras.optimizers.Adam(learning_rate=flags.learning_rate_input)
        
        # ── checkpoint helpers ─────────────────────────────────────────
        ckpt_savedir = os.path.join(flags.checkpoint_savedir, "model_ckpts")
        ckpt_loaddir = os.path.join(flags.checkpoint_loaddir, "model_ckpts")
        os.makedirs(ckpt_savedir, exist_ok=True)

        checkpoint = tf.train.Checkpoint(
            step=tf.Variable(0, dtype=tf.int64),
            optimizer=optimizer,
            model=model
        )

        # Save manager (for ongoing training)
        save_manager = tf.train.CheckpointManager(
            checkpoint,
            ckpt_savedir,
            max_to_keep=flags.max_to_keep
        )

        # Load manager (for resuming training)
        load_manager = tf.train.CheckpointManager(
            checkpoint,
            ckpt_loaddir,
            max_to_keep=1
        )

        # ── Optional checkpoint restore ────────────────────────────────
        print(f"[Checkpoint] Trying to load from: {ckpt_loaddir}")
        print(f"[Checkpoint] load_manager.latest_checkpoint = {load_manager.latest_checkpoint}")
        if flags.load_checkpoint and load_manager.latest_checkpoint:
            checkpoint.restore(load_manager.latest_checkpoint).expect_partial()
            print("✔ Restored from", load_manager.latest_checkpoint)
        else:
            print("✱ Training from scratch")

        # ─── FEED-FORWARD: hard-zero recurrent weights ─────────────────────
        if flags.feedforward:
            rsnn_cell = model.get_layer("rsnn").cell
            if hasattr(rsnn_cell, "w_rec"):
                rsnn_cell.w_rec.assign(tf.zeros_like(rsnn_cell.w_rec))
                tf.print("[feedforward]  w_rec matrix zeroed.")
            for v in model.variables:
                if "sparse_recurrent_weights" in v.name:
                    v.assign(tf.zeros_like(v))
        # ─────────────────────────────────────────────────────────────────

        # ─── INITIALIZE MASKS FROM LOADED WEIGHTS (if checkpoint was loaded) ───
        checkpoint_was_loaded = flags.load_checkpoint and load_manager.latest_checkpoint
        
        if checkpoint_was_loaded:
            print("\n🔄 Initializing masks from loaded checkpoint weights...")
            
            for v in model.trainable_variables:
                if "sparse_input_weights" in v.name:
                    # Create mask based on loaded weights: 0 where weights are 0, 1 elsewhere
                    # This preserves the existing pruning pattern from the checkpoint
                    loaded_mask = tf.cast(tf.not_equal(v, 0.0), v.dtype)
                    MASKS[v.ref()] = loaded_mask
                    
                    # Log the sparsity inherited from checkpoint
                    total_weights = tf.size(v, out_type=tf.int64)
                    active_weights = tf.math.count_nonzero(loaded_mask)
                    inherited_sparsity = 1.0 - (tf.cast(active_weights, tf.float32) / tf.cast(total_weights, tf.float32))
                    
                    print(f"  {v.name}: inherited sparsity = {inherited_sparsity:.3f} "
                          f"({active_weights}/{total_weights} weights active)")
                else:
                    # For non-pruned layers, use full mask
                    MASKS[v.ref()] = tf.ones_like(v, dtype=v.dtype)
                    
            print("✔ Masks initialized from checkpoint - existing pruning pattern preserved")
        else:
            # Training from scratch: initialize all masks to 1 (fully trainable)
            for v in model.trainable_variables:
                MASKS[v.ref()] = tf.ones_like(v, dtype=v.dtype)
            print("✱ Masks initialized to fully trainable (training from scratch)")

        # ─── MASK BEHAVIOR SUMMARY ──────────────────────────────────────────
        print("\n  Pruning behavior for this training:")
        print("   • Previously pruned weights (zeros) will remain permanently masked")
        print("   • Additional pruning may occur based on progressive schedule")
        print("   • No previously pruned weights can become active again")
        print("   • Masks are derived from loaded weights, not saved separately")

        # ─── TRAINING CONTROL SUMMARY ──────────────────────────────────────────
        print("\n   Training control flags:")
        print(f"   • Input weights training: {'ENABLED' if flags.train_input else 'DISABLED'}")
        print(f"   • Recurrent weights training: {'ENABLED' if flags.train_recurrent else 'DISABLED'}")
        
        # Show which trainable variables will be affected
        input_vars = []
        recurrent_vars = []
        other_vars = []
        
        for v in model.trainable_variables:
            if "sparse_input_weights" in v.name or "input" in v.name.lower():
                input_vars.append(v.name)
            elif "sparse_recurrent_weights" in v.name or "recurrent" in v.name.lower():
                recurrent_vars.append(v.name)
            else:
                other_vars.append(v.name)
        
        if input_vars:
            status = "will be trained" if flags.train_input else "will NOT be trained"
            print(f"   • Input variables ({status}): {', '.join(input_vars)}")
        if recurrent_vars:
            status = "will be trained" if flags.train_recurrent else "will NOT be trained"
            print(f"   • Recurrent variables ({status}): {', '.join(recurrent_vars)}")
        if other_vars:
            print(f"   • Other variables (always trained): {', '.join(other_vars)}")


    df = pd.read_csv("processed_sentences.csv")
    StimID_to_StimName = dict(zip(df.stim_number, df.file_name))
    
    # Load stimulus sets from flags (convert strings to integers)
    StimID_training_original = [int(x) for x in flags.training_stimuli]
    StimID_testing_original = [int(x) for x in flags.testing_stimuli]
    
    print(f"[Stimuli Config] Training stimuli: {StimID_training_original}")
    print(f"[Stimuli Config] Testing stimuli: {StimID_testing_original}")
    print(f"[Timing Config] Pruning stops {flags.pruning_stop_epochs_before_end} epochs before end")
    print(f"[Timing Config] Target scaling reaches 1.0 at {flags.scaling_stop_epochs_before_end} epochs before end")
    
    StimID_training_for_plotting = StimID_training_original[0]
    StimID_testing_for_plotting = StimID_testing_original[0]

    rsnn_layer = model.get_layer("rsnn")
    # extractor_model is no longer needed for chunked rollout with state carryover

    # STATE MANAGEMENT:
    # - global_training_state: carries state across all stimuli during training
    # - global_testing_state: carries state across all stimuli during testing
    # - sentence_state: kept for backward compatibility and plotting (per-stimulus state)
    global_training_state = rsnn_layer.cell.zero_state(1, tf.float32)
    global_testing_state = rsnn_layer.cell.zero_state(1, tf.float32)
    
    sentence_state = {
        sid: rsnn_layer.cell.zero_state(1, tf.float32)
        for sid in StimID_training_original + StimID_testing_original
    }

    @tf.function
    def roll_out_chunk(inp_kTN, init_state, return_diagnostics=False):
        """
        Roll out a chunk of the RNN using direct access to the RNN layer for correct state carryover.

        Args:
            inp_kTN: Tensor of shape [batch, time, n_input] (e.g., [1, chunk_size, n_input])
                - This is the LGN input chunk for the current window.
            init_state: tuple of tensors, each of shape [batch, n_neurons] (or as required by the cell)
                - This is the RNN state to use as the initial state for this chunk.
            return_diagnostics: bool, whether to return voltage and current for diagnostic plotting

        Returns:
            spikes: Tensor of shape [batch, time, n_neurons] (or as returned by the RNN)
            new_state: tuple of tensors, each of shape [batch, n_neurons] (or as returned by the RNN)
            voltage: (optional) Tensor of shape [batch, time, n_neurons] - only if return_diagnostics=True
            current: (optional) Tensor of shape [batch, time, n_neurons] - only if return_diagnostics=True

        Input structure:
            - The RNN layer expects processed input currents with shape matching the full model pipeline.
            - Raw inputs must be processed through LGNInputLayer and BKGInputLayer first.
            - For use_state_input=False, we also need a zero state_input tensor.
            - The RNN state is passed as the initial_state argument to the RNN layer.
        """
        # Generate background input (zeros) for this chunk
        step_bkg = tf.zeros((tf.shape(inp_kTN)[0], tf.shape(inp_kTN)[1], flags.neurons), dtype=inp_kTN.dtype)
        
        # Process inputs through the input layers to get the correct format
        lgn_layer = model.get_layer("input_layer")  # LGNInputLayer
        bkg_layer = model.get_layer("noise_layer")  # BKGInputLayer
        
        # Process LGN and background inputs to get currents
        lgn_currents = lgn_layer(inp_kTN)  # [batch, time, 5*n_neurons]
        bkg_currents = bkg_layer(step_bkg)  # [batch, time, 5*n_neurons] 
        
        # For use_state_input=False, we still need a zero state_input tensor for concatenation
        state_input_zeros = tf.zeros((tf.shape(inp_kTN)[0], tf.shape(inp_kTN)[1], flags.neurons), dtype=inp_kTN.dtype)
        
        # Concatenate as expected by the model: lgn_inputs + bkg_inputs + state_input
        rnn_input = tf.concat([lgn_currents, bkg_currents, state_input_zeros], axis=-1)
        
        # Directly call the RNN layer for stateful chunked rollout
        rsnn_layer = model.get_layer("rsnn")
        rnn_outputs = rsnn_layer(rnn_input, initial_state=init_state)
        # rnn_outputs: (hidden, state1, state2, ...) where hidden is (spikes, voltage, current)
        if isinstance(rnn_outputs, (tuple, list)):
            hidden = rnn_outputs[0]  # This is the tuple (spikes, voltage, current)
            spikes = hidden[0]       # Extract just the spikes
            new_state = tuple(rnn_outputs[1:])
            
            if return_diagnostics:
                voltage = hidden[1]  # Extract voltage
                current = hidden[2]  # Extract current
                return spikes, new_state, voltage, current
            else:
                return spikes, new_state
        else:
            # If only hidden is returned (no states)
            hidden = rnn_outputs
            spikes = hidden[0]
            new_state = ()
            
            if return_diagnostics:
                voltage = hidden[1]
                current = hidden[2]
                return spikes, new_state, voltage, current
            else:
                return spikes, new_state

    epoch_losses_train = []
    epoch_rate_losses_train = []
    epoch_input_regularizations = []
    epoch_input_regularizations_test = []
    epoch_losses_test = []
    epoch_rate_losses_test = []
    epoch_weight_proportions = []
    epoch_target_scaling_values = []  # Track target_scaling changes over epochs
    sampled_weights_log = {'aud_in': [], 'bkg': [], 'recurrent': []}

    def run_step(sid: int, training=True, 
                 current_input_l1_weight=0.0, current_input_l2_weight=0.0,
                 current_recurrent_l1_weight=0.0, current_recurrent_l2_weight=0.0,
                 current_background_l1_weight=0.0, current_background_l2_weight=0.0,
                 epoch=0, global_state=None):
        """
        Process a single stimulus with chunked rollout.
        
        State management:
        - If global_state is provided: Uses global state for inter-stimulus continuity
        - If global_state is None: Uses per-stimulus state (backward compatibility)
        
        Returns: chunk_losses, chunk_rate_terms, chunk_log_terms, final_state
        """
        stim_dur_ms = int(np.floor((1.5 + df.loc[df.stim_number == sid, 'duration_ucsf'].values[0]) * 1000 - 1))

        aud_full, trg_full, _ = tu.load_aud_input_and_target_rates(
            StimID             = sid,
            input_dir          = flags.input_dir,
            target_rates_dir   = flags.target_rates_dir,
            StimID_to_StimName = StimID_to_StimName,
            seq_len            = stim_dur_ms,
            n_input            = flags.n_input,
            batch_size         = 1,
            t_start            = 0,
            N_neurons          = n_targets,
            target_cell_ids    = uniq_target_ids,
            repeat_number      = None  # ← this ensures random selection
        )

        aud_full = aud_full[0].astype(np.float32)
        trg_full = trg_full[0].astype(np.float32)

        T_full = aud_full.shape[0]
        
        # Choose chunking method based on chunk_len_ms format
        chunk_ranges = list(make_discrete_random_chunk_sampler(
            total_len_ms=T_full, 
            chunk_options=flags.chunk_len_ms
        ))
        
        chunk_losses, chunk_rate_terms, chunk_log_terms = [], [], []

        # Diagnostic plotting: only for first training stimulus in first epoch
        plot_diagnostics = (flags.plot_internal_state and 
                          sid == StimID_training_for_plotting and 
                          epoch == 0 and 
                          training)
        voltage_traces = []
        current_traces = []
        chunk_boundaries = []
        time_offset = 0

        # State management: use global state for inter-stimulus continuity
        # Fall back to per-stimulus state if no global state provided (backward compatibility)
        if global_state is not None:
            current_state = global_state
        else:
            current_state = sentence_state[sid]

        for start, end in chunk_ranges:
            aud = aud_full[start:end, :]
            trg = trg_full[start:end, :]
            if aud.shape[0] < 2 or trg.shape[0] < 2:
                continue

            inp_kTN = tf.expand_dims(tf.convert_to_tensor(aud, dtype=tf.float32), 0)
            init_st = current_state

            # Get appropriate sigma for this stimulus (randomized for training, fixed for testing)
            current_sigma = get_sigma_smoothing(flags, training=training)

            with tf.GradientTape(persistent=training) as tape:
                if plot_diagnostics:
                    spk_kTN, new_st, voltage_kTN, current_kTN = roll_out_chunk(inp_kTN, init_st, return_diagnostics=True)
                    # Store diagnostics for first 5 neurons only
                    voltage_traces.append(voltage_kTN[0, :, :5].numpy())  # [time, 5_neurons]
                    current_traces.append(current_kTN[0, :, :5].numpy())  # [time, 5_neurons]
                    time_offset += voltage_kTN.shape[1]
                    chunk_boundaries.append(time_offset)
                else:
                    spk_kTN, new_st = roll_out_chunk(inp_kTN, init_st, return_diagnostics=False)
                
                model_spk = spk_kTN[0]
                
                # Apply pooling only if enabled
                if flags.enable_pooling:
                    model_spk_for_loss = tf.matmul(model_spk, POOL_MAT)
                else:
                    model_spk_for_loss = model_spk
                    
                loss, rate_term, log_term, _, _, _ = tu.compute_combined_loss2(
                    model,
                    model_spk_for_loss,
                    trg,
                    target_scaling = flags.target_scaling,
                    sigma          = current_sigma,  # Use randomized sigma for training
                    log_sum_weight = flags.log_sum_weight,
                    rate_weight    = flags.rate_weight,
                    input_l1_weight = current_input_l1_weight,
                    input_l2_weight = current_input_l2_weight,
                    recurrent_l1_weight = current_recurrent_l1_weight,
                    recurrent_l2_weight = current_recurrent_l2_weight,
                    background_l1_weight = current_background_l1_weight,
                    background_l2_weight = current_background_l2_weight,
                    # ADAPTIVE NEURON WEIGHTING PARAMETERS (NEW)
                    enable_adaptive_weighting=enable_adaptive_weighting,
                    adaptive_neuron_thresholds=adaptive_neuron_thresholds,
                    adaptive_high_weight=1.5,
                    adaptive_low_weight=0.1
                )


            if training:
                grads = tape.gradient(loss, model.trainable_variables)
                
                # Apply training control flags: zero gradients for disabled weight types
                masked_grads = []
                for g, v in zip(grads, model.trainable_variables):
                    if g is None:
                        masked_grads.append(None)
                        continue
                    
                    # Check if this variable should be trained based on flags
                    should_train = True
                    
                    # Input weights: controlled by train_input flag
                    if "sparse_input_weights" in v.name or "input" in v.name.lower():
                        should_train = flags.train_input
                        
                    # Recurrent weights: controlled by train_recurrent flag  
                    elif "sparse_recurrent_weights" in v.name or "recurrent" in v.name.lower():
                        should_train = flags.train_recurrent
                        
                    # Background/other weights: always train (could add flag if needed)
                    # elif "rest_of_brain_weights" in v.name or "bkg" in v.name.lower():
                    #     should_train = True
                    
                    if should_train:
                        # Apply pruning mask
                        masked_grad = g * MASKS[v.ref()]
                    else:
                        # Zero gradient to disable training
                        masked_grad = tf.zeros_like(g)
                    
                    masked_grads.append(masked_grad)
                
                optimizer.apply_gradients(zip(masked_grads, model.trainable_variables))
            
            # Update state for next chunk: use both global state and per-stimulus state
            current_state = new_st  # Update the current state for next chunk in this stimulus
            # Note: global_state parameter is passed by value, so we can't modify the caller's reference
            # The caller will use the returned current_state to update their global_state
            sentence_state[sid] = new_st  # Always update per-stimulus state for backward compatibility

            chunk_losses.append(loss.numpy())
            chunk_rate_terms.append(rate_term.numpy())
            chunk_log_terms.append(log_term.numpy())

        # Generate diagnostic plots if enabled
        if plot_diagnostics and len(voltage_traces) > 0:
            import matplotlib.pyplot as plt
            
            # Concatenate all chunks
            voltage_cat = np.concatenate(voltage_traces, axis=0)  # [total_time, 5]
            current_cat = np.concatenate(current_traces, axis=0)  # [total_time, 5]
            chunk_boundaries_arr = np.array(chunk_boundaries)
            
            # Time axis
            t = np.arange(voltage_cat.shape[0])
            
            # Create save directory
            save_dir = os.path.join(flags.checkpoint_savedir, 'internal_state_plots')
            os.makedirs(save_dir, exist_ok=True)
            
            # Voltage plot (full)
            plt.figure(figsize=(14, 8))
            for i in range(5):
                plt.plot(t, voltage_cat[:, i], label=f'Neuron {i}', alpha=0.8)
            for b in chunk_boundaries_arr[:-1]:  # Don't plot the last boundary
                plt.axvline(b, color='red', linestyle='--', alpha=0.7, linewidth=1)
            plt.title(f'Voltage Traces - First 5 Neurons (Stim {sid}, Epoch {epoch})\nRed lines show chunk boundaries')
            plt.xlabel('Time (ms)')
            plt.ylabel('Voltage')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'voltage_stim{sid}_epoch{epoch}_full.png'), dpi=150)
            plt.close()
            
            # Voltage plot (zoomed: first 1200ms)
            max_time = min(1200, voltage_cat.shape[0])
            plt.figure(figsize=(14, 8))
            for i in range(5):
                plt.plot(t[:max_time], voltage_cat[:max_time, i], label=f'Neuron {i}', alpha=0.8)
            for b in chunk_boundaries_arr[:-1]:
                if b <= max_time:
                    plt.axvline(b, color='red', linestyle='--', alpha=0.7, linewidth=1)
            plt.title(f'Voltage Traces - First 5 Neurons (Stim {sid}, Epoch {epoch}, 0-1200ms)\nRed lines show chunk boundaries')
            plt.xlabel('Time (ms)')
            plt.ylabel('Voltage')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'voltage_stim{sid}_epoch{epoch}_zoom.png'), dpi=150)
            plt.close()
            
            # Current plot (full)
            plt.figure(figsize=(14, 8))
            for i in range(5):
                plt.plot(t, current_cat[:, i], label=f'Neuron {i}', alpha=0.8)
            for b in chunk_boundaries_arr[:-1]:
                plt.axvline(b, color='red', linestyle='--', alpha=0.7, linewidth=1)
            plt.title(f'Current Traces - First 5 Neurons (Stim {sid}, Epoch {epoch})\nRed lines show chunk boundaries')
            plt.xlabel('Time (ms)')
            plt.ylabel('Current')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'current_stim{sid}_epoch{epoch}_full.png'), dpi=150)
            plt.close()
            
            # Current plot (zoomed: first 1200ms)
            plt.figure(figsize=(14, 8))
            for i in range(5):
                plt.plot(t[:max_time], current_cat[:max_time, i], label=f'Neuron {i}', alpha=0.8)
            for b in chunk_boundaries_arr[:-1]:
                if b <= max_time:
                    plt.axvline(b, color='red', linestyle='--', alpha=0.7, linewidth=1)
            plt.title(f'Current Traces - First 5 Neurons (Stim {sid}, Epoch {epoch}, 0-1200ms)\nRed lines show chunk boundaries')
            plt.xlabel('Time (ms)')
            plt.ylabel('Current')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'current_stim{sid}_epoch{epoch}_zoom.png'), dpi=150)
            plt.close()
            
            print(f"[Diagnostic] Saved voltage and current plots for Stim {sid}, Epoch {epoch} to {save_dir}")

        return chunk_losses, chunk_rate_terms, chunk_log_terms, current_state

    # ═══════════════════════════════════════════════════════════════════════════════
    # SAVE TRAINING CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════════════
    
    # Save all flag values to a configuration file for reference
    config_save_path = os.path.join(flags.checkpoint_savedir, 'training_configuration.json')
    config_data = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'hostname': socket.gethostname(),
        'training_stimuli': StimID_training_original,
        'testing_stimuli': StimID_testing_original,
        'flags': {}
    }
    
    # Collect all flag values
    for flag_name in sorted(dir(flags)):
        if not flag_name.startswith('_'):  # Skip private attributes
            try:
                flag_value = getattr(flags, flag_name)
                # Convert numpy types and other non-serializable types to Python native types
                if hasattr(flag_value, 'item'):  # numpy scalar
                    flag_value = flag_value.item()
                elif hasattr(flag_value, 'tolist'):  # numpy array
                    flag_value = flag_value.tolist()
                config_data['flags'][flag_name] = flag_value
            except:
                config_data['flags'][flag_name] = str(getattr(flags, flag_name))
    
    # Create directory if it doesn't exist
    os.makedirs(flags.checkpoint_savedir, exist_ok=True)
    
    # Save as JSON for easy reading
    with open(config_save_path, 'w') as f:
        json.dump(config_data, f, indent=2, sort_keys=True)
    
    print(f"[Config] Saved training configuration to: {config_save_path}")
    print(f"[Config] Training stimuli: {len(StimID_training_original)} stimuli")
    print(f"[Config] Testing stimuli: {len(StimID_testing_original)} stimuli")
    print(f"[Config] Pruning stops {flags.pruning_stop_epochs_before_end} epochs before end")
    print(f"[Config] Target scaling reaches 1.0 at {flags.scaling_stop_epochs_before_end} epochs before end")
    
    # Log sigma smoothing configuration
    if len(flags.sigma_smoothing) == 2:
        min_sigma = float(flags.sigma_smoothing[0])
        max_sigma = float(flags.sigma_smoothing[1])
        print(f"[Config] Sigma smoothing: RANDOMIZED training ({min_sigma}-{max_sigma}), FIXED testing ({flags.sigma_smoothing_test})")
    elif len(flags.sigma_smoothing) == 1:
        train_sigma = float(flags.sigma_smoothing[0])
        print(f"[Config] Sigma smoothing: FIXED training ({train_sigma}), FIXED testing ({flags.sigma_smoothing_test})")
    else:
        print(f"[Config] Sigma smoothing: WARNING - invalid format, using first value for training")

    # Log chunking configuration
    if len(flags.chunk_len_ms) == 1:
        chunk_len = int(flags.chunk_len_ms[0])
        print(f"[Config] Chunking: FIXED length ({chunk_len}ms)")
    else:
        chunk_options = [int(x) for x in flags.chunk_len_ms]
        print(f"[Config] Chunking: DISCRETE RANDOM selection from {chunk_options}ms (solves TensorFlow retracing)")

    # Store the original target_scaling value for exponential decay
    original_target_scaling = float(flags.target_scaling)
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # ADAPTIVE NEURON WEIGHTING SETUP (NEW - ADD THIS SECTION)
    # ═══════════════════════════════════════════════════════════════════════════════
    adaptive_neuron_thresholds = None
    enable_adaptive_weighting = False  # TEMPORARILY DISABLED - TEST STANDARD MSE
    
    if enable_adaptive_weighting:
        print(f"[Adaptive Weighting] Computing neuron thresholds from ALL training stimuli...")
        
        # Collect target rates from ALL training stimuli for robust threshold computation
        all_target_rates = []
        total_time_points = 0
        
        for i, sid in enumerate(StimID_training_original):
            print(f"[Adaptive Weighting] Loading stimulus {sid} ({i+1}/{len(StimID_training_original)})...")
            
            stim_dur_ms = int(np.floor((1.5 + df.loc[df.stim_number == sid, 'duration_ucsf'].values[0]) * 1000 - 1))
            
            _, trg_stimulus, _ = tu.load_aud_input_and_target_rates(
                StimID=sid,
                input_dir=flags.input_dir,
                target_rates_dir=flags.target_rates_dir,
                StimID_to_StimName=StimID_to_StimName,
                seq_len=stim_dur_ms,
                n_input=flags.n_input,
                batch_size=1,
                t_start=0,
                N_neurons=n_targets,
                target_cell_ids=uniq_target_ids,
                repeat_number=None
            )
            
            trg_stimulus = trg_stimulus[0].astype(np.float32)  # [time, neurons]
            all_target_rates.append(trg_stimulus)
            total_time_points += trg_stimulus.shape[0]
            print(f"   Loaded {trg_stimulus.shape[0]} time points, {trg_stimulus.shape[1]} neurons")
        
        # Concatenate all stimuli along time axis: [total_time, neurons]
        concatenated_targets = np.concatenate(all_target_rates, axis=0)
        print(f"[Adaptive Weighting] Concatenated data: {concatenated_targets.shape[0]} total time points, {concatenated_targets.shape[1]} neurons")
        
        # Compute thresholds using the concatenated data from all training stimuli
        # Use test sigma for consistent threshold computation (not randomized)
        adaptive_neuron_thresholds = tf.constant(
            tu.compute_adaptive_neuron_thresholds(
                concatenated_targets, 
                percentile=85.0,
                target_scaling=flags.target_scaling,
                sigma=flags.sigma_smoothing_test
            ),
            dtype=tf.float32
        )
        
        print(f"[Adaptive Weighting] Computed thresholds for {tf.shape(adaptive_neuron_thresholds)[0]} neurons")
        print(f"[Adaptive Weighting] Threshold range: {tf.reduce_min(adaptive_neuron_thresholds):.4f} - {tf.reduce_max(adaptive_neuron_thresholds):.4f}")
        print(f"[Adaptive Weighting] Weights: 1.0 (above threshold) / 0.1 (below threshold)")
        print(f"[Adaptive Weighting] Used {len(StimID_training_original)} training stimuli for robust threshold estimation")
    else:
        print("[Adaptive Weighting] Disabled - using standard MSE loss")
    
    # ═══════════════════════════════════════════════════════════════════════════════
    
    for epoch in range(flags.num_epochs):
        t0 = time_func()
        # ── schedule target_scaling ──────────────────────────────
        # Only apply progressive scaling if initial value is not already 1.0
        if original_target_scaling != 1.0:
            # Exponential decay from original value to 1.0, reaching 1.0 at specified epochs before end
            # This follows the same mathematical pattern as pruning: constant proportional decrease each epoch
            # Formula: target_scaling = original_target_scaling * retention_factor^epoch
            # where retention_factor = (1.0 / original_target_scaling)^(1/scaling_end_epoch)
            scaling_end_epoch = flags.num_epochs - flags.scaling_stop_epochs_before_end  # reach 1.0 here (controlled by flag)
            
            if scaling_end_epoch <= 0:
                # No time for scaling - keep original value
                flags.target_scaling = original_target_scaling
                if epoch == 0:
                    print(f"[Target Scaling] No time for scaling (scaling_end_epoch={scaling_end_epoch}) - keeping constant at {original_target_scaling:.3f}")
            elif epoch >= scaling_end_epoch:
                # Final epochs: keep at 1.0
                flags.target_scaling = 1.0
            elif epoch == 0:
                # First epoch: use original value and log the plan
                flags.target_scaling = original_target_scaling
                # Calculate the retention factor for exponential decay
                retention_factor = (1.0 / original_target_scaling) ** (1.0 / scaling_end_epoch)
                proportional_decrease = 1.0 - retention_factor
                print(f"[Target Scaling] Starting exponential scaling from {original_target_scaling:.3f} to 1.0 over {scaling_end_epoch} epochs")
                print(f"[Target Scaling] Retention factor: {retention_factor:.4f} (proportional decrease: {proportional_decrease:.1%} per epoch)")
            else:
                # Exponential decay: constant proportional decrease each epoch
                # This ensures smooth approach to target while maintaining constant percentage reduction
                retention_factor = (1.0 / original_target_scaling) ** (1.0 / scaling_end_epoch)
                flags.target_scaling = original_target_scaling * (retention_factor ** epoch)
                
                # Ensure we hit exactly 1.0 at scaling_end_epoch (handle floating point precision)
                if epoch == scaling_end_epoch - 1:
                    flags.target_scaling = 1.0
            
            if epoch == 0 or epoch >= scaling_end_epoch or epoch % 10 == 0:  # Log periodically
                print(f"[Target Scaling] Epoch {epoch+1}: target_scaling={flags.target_scaling:.3f}")

        # ─────────────────────────────────────────────────────────

        all_losses = []
        all_rate_terms = []
        all_log_terms = []

        # ─── ADAPTIVE L1/L2 REGULARIZATION SCHEDULE ───────────────────────────────
        # Compute regularization weights for all three weight types: input, recurrent, background
        
        # Use most recent training loss as scale reference
        reference_loss = epoch_rate_losses_train[-1] if epoch_rate_losses_train else 1e-3
        
        # Initialize regularization weights
        current_input_l1_weight = 0.0
        current_input_l2_weight = 0.0
        current_recurrent_l1_weight = 0.0
        current_recurrent_l2_weight = 0.0
        current_background_l1_weight = 0.0
        current_background_l2_weight = 0.0
        
        # Helper function to compute adaptive weights
        def compute_adaptive_weights(weights_np, l1_fraction, l2_fraction, weight_type_name, current_sparsity=0.0, current_epoch=0):
            """
            Compute regularization weights such that:
            - L1 regularization contributes l1_fraction of the current reference loss
            - L2 regularization contributes l2_fraction of the current reference loss
            - L1 is scaled down as sparsity increases to prevent over-aggressive regularization
            - L1 has a burn-in period to avoid initialization issues
            
            Args:
                weights_np: numpy array of weights
                l1_fraction: desired L1 regularization as fraction of reference loss
                l2_fraction: desired L2 regularization as fraction of reference loss
                weight_type_name: string for logging
                current_sparsity: current sparsity level (0.0 to 1.0) for L1 scaling
                current_epoch: current training epoch (0-indexed) for burn-in calculation
            
            Returns:
                l1_weight, l2_weight: regularization weights to pass to loss function
            """
            if weights_np is None or weights_np.size == 0:
                return 0.0, 0.0
                
            # Compute penalty values (what the loss function will compute)
            l1_penalty_val = np.sum(np.abs(weights_np))
            l2_penalty_val = np.sum(weights_np ** 2)
            
            # Avoid divide-by-zero
            l1_penalty_val = max(l1_penalty_val, 1e-12)
            l2_penalty_val = max(l2_penalty_val, 1e-12)
            
            # Scale down L1 regularization as sparsity increases
            if flags.enable_l1_sparsity_scaling and flags.l1_scaling_method != 'none':
                if flags.l1_scaling_method == 'power':
                    # Power law scaling: (1-sparsity)^exponent
                    sparsity_scale_factor = max(flags.l1_min_scale_factor, 
                                              (1.0 - current_sparsity) ** flags.l1_sparsity_scaling_exponent)
                elif flags.l1_scaling_method == 'sigmoid':
                    # Sigmoid scaling: smooth transition around 50% sparsity
                    sigmoid_input = flags.l1_sigmoid_steepness * (current_sparsity - 0.5)
                    sigmoid_factor = 1.0 / (1.0 + np.exp(sigmoid_input))  # 1 at low sparsity, 0 at high sparsity
                    sparsity_scale_factor = max(flags.l1_min_scale_factor, sigmoid_factor)
                elif flags.l1_scaling_method == 'exponential':
                    # Exponential decay: exp(-rate * sparsity)
                    exp_factor = np.exp(-flags.l1_exponential_rate * current_sparsity)
                    sparsity_scale_factor = max(flags.l1_min_scale_factor, exp_factor)
                else:
                    # Default to power method if unknown
                    sparsity_scale_factor = max(flags.l1_min_scale_factor, 
                                              (1.0 - current_sparsity) ** flags.l1_sparsity_scaling_exponent)
                
                effective_l1_fraction = l1_fraction * sparsity_scale_factor
            else:
                sparsity_scale_factor = 1.0
                effective_l1_fraction = l1_fraction
            
            # Apply burn-in scaling to L1 regularization (gradually ramp up from 0 to full strength)
            burnin_scale_factor = 1.0
            if flags.l1_burnin_epochs > 0 and current_epoch < flags.l1_burnin_epochs and flags.l1_burnin_method != 'none':
                if flags.l1_burnin_method == 'linear':
                    # Linear ramp: 0 to 1 over burn-in epochs
                    burnin_scale_factor = current_epoch / flags.l1_burnin_epochs
                elif flags.l1_burnin_method == 'quadratic':
                    # Quadratic ramp: smoother start, faster finish
                    burnin_progress = current_epoch / flags.l1_burnin_epochs
                    burnin_scale_factor = burnin_progress ** 2
                else:
                    # Default to linear if unknown method
                    burnin_scale_factor = current_epoch / flags.l1_burnin_epochs
                
                # Ensure we don't go below 0 or above 1
                burnin_scale_factor = max(0.0, min(1.0, burnin_scale_factor))
                effective_l1_fraction *= burnin_scale_factor
            
            # Compute weights such that: weight * penalty_val = fraction * reference_loss
            # So: weight = (fraction * reference_loss) / penalty_val
            l1_weight = (effective_l1_fraction * reference_loss) / l1_penalty_val if effective_l1_fraction > 0 else 0.0
            l2_weight = (l2_fraction * reference_loss) / l2_penalty_val if l2_fraction > 0 else 0.0
            
            # Log the expected contribution for debugging
            expected_l1_contrib = l1_weight * l1_penalty_val if effective_l1_fraction > 0 else 0.0
            expected_l2_contrib = l2_weight * l2_penalty_val if l2_fraction > 0 else 0.0
            
            if l1_fraction > 0 or l2_fraction > 0:
                print(f"  {weight_type_name}: epoch={current_epoch}, sparsity={current_sparsity:.3f}")
                print(f"  {weight_type_name}: sparsity_scale={sparsity_scale_factor:.3f}, burnin_scale={burnin_scale_factor:.3f}")
                print(f"  {weight_type_name}: penalty_L1={l1_penalty_val:.2e}, penalty_L2={l2_penalty_val:.2e}")
                print(f"  {weight_type_name}: effective_L1_fraction={effective_l1_fraction:.2e} (original={l1_fraction:.2e})")
                print(f"  {weight_type_name}: expected_L1_contrib={expected_l1_contrib:.2e} (target={effective_l1_fraction * reference_loss:.2e})")
                print(f"  {weight_type_name}: expected_L2_contrib={expected_l2_contrib:.2e} (target={l2_fraction * reference_loss:.2e})")
            
            return l1_weight, l2_weight
        
        # Helper function to calculate current sparsity
        def calculate_current_sparsity(weights_np, mask=None, weight_type_name="unknown"):
            """Calculate current sparsity level for weights"""
            if weights_np is None or weights_np.size == 0:
                return 0.0
            
            if mask is not None:
                # Use mask if available (for sparse layers)
                total_weights = mask.size
                active_weights = np.count_nonzero(mask)
                sparsity = 1.0 - (active_weights / total_weights)
            else:
                # Use threshold-based calculation (for non-sparse layers)
                threshold = 1e-10
                total_weights = weights_np.size
                active_weights = np.count_nonzero(np.abs(weights_np) > threshold)
                sparsity = 1.0 - (active_weights / total_weights)
            
            print(f"  {weight_type_name}: current sparsity = {sparsity:.3f}")
            return sparsity
        
        # ──── INPUT WEIGHTS ────
        w_input_np = None
        input_mask = None
        for name in ["input_layer", "sparse_input_layer"]:
            try:
                layer = model.get_layer(name)
                w_input_np = layer.trainable_weights[0].numpy()
                # Check if this layer has a mask
                for v in model.trainable_variables:
                    if "sparse_input_weights" in v.name:
                        input_mask = MASKS.get(v.ref(), None)
                        if input_mask is not None:
                            input_mask = input_mask.numpy()
                        break
                break
            except ValueError:
                continue
        
        if w_input_np is not None:
            input_sparsity = calculate_current_sparsity(w_input_np, input_mask, "input")
            current_input_l1_weight, current_input_l2_weight = compute_adaptive_weights(
                w_input_np, flags.input_l1_fraction, flags.input_l2_fraction, "input", input_sparsity, epoch
            )
        
        # ──── RECURRENT WEIGHTS ────
        w_recurrent_np = None
        for v in model.trainable_variables:
            if "sparse_recurrent_weights" in v.name or "recurrent" in v.name.lower():
                w_recurrent_np = v.numpy()
                break
        
        if w_recurrent_np is not None:
            recurrent_sparsity = calculate_current_sparsity(w_recurrent_np, None, "recurrent")
            current_recurrent_l1_weight, current_recurrent_l2_weight = compute_adaptive_weights(
                w_recurrent_np, flags.recurrent_l1_fraction, flags.recurrent_l2_fraction, "recurrent", recurrent_sparsity, epoch
            )
        
        # ──── BACKGROUND WEIGHTS ────
        w_background_np = None
        for v in model.trainable_variables:
            if "rest_of_brain_weights" in v.name or "bkg" in v.name.lower() or "background" in v.name.lower():
                w_background_np = v.numpy()
                break
        
        if w_background_np is not None:
            background_sparsity = calculate_current_sparsity(w_background_np, None, "background")
            current_background_l1_weight, current_background_l2_weight = compute_adaptive_weights(
                w_background_np, flags.background_l1_fraction, flags.background_l2_fraction, "background", background_sparsity, epoch
            )
        
        # ──── LOGGING ────
        # print(f"[Epoch {epoch+1}] Adaptive regularization (reference_loss={reference_loss:.3e}):")
        # print(f"  Fractions: input_L1={flags.input_l1_fraction}, input_L2={flags.input_l2_fraction}")
        # print(f"  Fractions: recurrent_L1={flags.recurrent_l1_fraction}, recurrent_L2={flags.recurrent_l2_fraction}")
        # print(f"  Fractions: background_L1={flags.background_l1_fraction}, background_L2={flags.background_l2_fraction}")
        # print(f"  Computed weights:")
        # print(f"    Input:      L1={current_input_l1_weight:.3e} | L2={current_input_l2_weight:.3e}")
        # print(f"    Recurrent:  L1={current_recurrent_l1_weight:.3e} | L2={current_recurrent_l2_weight:.3e}")
        # print(f"    Background: L1={current_background_l1_weight:.3e} | L2={current_background_l2_weight:.3e}")

        # Shuffle training stimuli order each epoch if shuffle flag is enabled
        if flags.shuffle:
            StimID_training_shuffled = StimID_training_original.copy()
            random.shuffle(StimID_training_shuffled)
            print(f"[Epoch {epoch+1}] Training stimuli order: {StimID_training_shuffled}")
        else:
            StimID_training_shuffled = StimID_training_original
            print(f"[Epoch {epoch+1}] Training stimuli order (fixed): {StimID_training_shuffled}")

        # Reset global training state at the beginning of each epoch
        global_training_state = rsnn_layer.cell.zero_state(1, tf.float32)
        #print(f"[Epoch {epoch+1}] State continuity: ENABLED across stimuli (reset at epoch start)")

        for sid in StimID_training_shuffled:
            chunk_losses, chunk_rate_terms, chunk_log_terms, final_state = run_step(
                sid, training=True, 
                current_input_l1_weight=current_input_l1_weight, 
                current_input_l2_weight=current_input_l2_weight,
                current_recurrent_l1_weight=current_recurrent_l1_weight,
                current_recurrent_l2_weight=current_recurrent_l2_weight,
                current_background_l1_weight=current_background_l1_weight,
                current_background_l2_weight=current_background_l2_weight,
                epoch=epoch,
                global_state=global_training_state
            )
            # Update global state for next stimulus
            global_training_state = final_state
            
            all_losses.extend(chunk_losses)
            all_rate_terms.extend(chunk_rate_terms)
            all_log_terms.extend(chunk_log_terms)
            print(f"Stim {sid}: mean chunk loss {np.mean(chunk_losses) if chunk_losses else 0.0:.4f}")
            gc.collect()

        avg_loss = np.mean(all_losses) if all_losses else 0.0
        avg_rate = np.mean(all_rate_terms) if all_rate_terms else 0.0
        avg_log = np.mean(all_log_terms) if all_log_terms else 0.0

        epoch_losses_train.append(avg_loss)
        epoch_rate_losses_train.append(avg_rate)
        epoch_input_regularizations.append(avg_log)
        epoch_target_scaling_values.append(float(flags.target_scaling))  # Track current target_scaling value

        print(f"Epoch {epoch+1}/{flags.num_epochs} avg loss {avg_loss:.4f} (elapsed {time_func()-t0:.1f}s)")
        get_gpu_memory()

        # ---------- epoch-end sanitation + progressive pruning ----------
        target_final_sparsity = flags.target_final_sparsity  # Get target sparsity from command line flags
        pruning_end_epoch = flags.num_epochs - flags.pruning_stop_epochs_before_end  # no pruning in the final epochs (controlled by flag)
        
        if epoch < pruning_end_epoch:
            # Progressive pruning: constant proportion of REMAINING weights pruned each epoch
            # Goal: prune same % of current active weights each epoch
            
            target_prune_fraction_per_epoch = 0.0  # Will be calculated based on target_final_sparsity and remaining epochs
            
            # Check current sparsity BEFORE calculating what to prune
            for v in model.trainable_variables:
                if "sparse_input_weights" in v.name:
                    current_mask = MASKS[v.ref()]
                    total_weights = tf.size(current_mask, out_type=tf.int64)
                    current_active_weights = tf.math.count_nonzero(current_mask)
                    current_sparsity = 1.0 - (tf.cast(current_active_weights, tf.float32) / tf.cast(total_weights, tf.float32))
                    
                    print(f"[Pruning Check] Epoch {epoch+1}: Current sparsity: {current_sparsity:.3f}, Target: {target_final_sparsity:.3f}")
                    
                    # If already at or above target sparsity, skip pruning entirely
                    if current_sparsity >= target_final_sparsity:
                        print(f"[Pruning] Epoch {epoch+1}: Already at target sparsity - skipping pruning")
                        target_prune_fraction_per_epoch = 0.0  # No pruning
                        break
                    
                    # Calculate how much we need to prune to reach target by pruning_end_epoch
                    remaining_epochs = pruning_end_epoch - epoch
                    current_density = tf.cast(current_active_weights, tf.float32) / tf.cast(total_weights, tf.float32)
                    target_density = 1.0 - target_final_sparsity
                    
                    if remaining_epochs > 0 and current_density > target_density:
                        # Calculate constant fraction to prune each epoch
                        # If we prune fraction 'p' each epoch: density_final = density_current * (1-p)^remaining_epochs
                        # Solving: target_density = current_density * (1-p)^remaining_epochs
                        # p = 1 - (target_density / current_density)^(1/remaining_epochs)
                        retention_factor = (target_density / current_density) ** (1.0 / remaining_epochs)
                        target_prune_fraction_per_epoch = 1.0 - retention_factor
                        target_prune_fraction_per_epoch = max(0.0, min(target_prune_fraction_per_epoch, 0.5))  # Cap between 0% and 50%
                        
                        print(f"[Pruning] Epoch {epoch+1}: Will prune {target_prune_fraction_per_epoch:.1%} of active weights "
                              f"(current_density={current_density:.3f}, target={target_density:.3f}, remaining_epochs={remaining_epochs})")
                    else:
                        # Already at target or no time left - don't prune
                        target_prune_fraction_per_epoch = 0.0
                        print(f"[Pruning] Epoch {epoch+1}: At target density - no pruning needed")
                    
                    break  # Only check the first sparse_input_weights layer
            else:
                # Fallback if no sparse_input_weights found - calculate based on target_final_sparsity
                remaining_epochs = pruning_end_epoch - epoch
                if remaining_epochs > 0:
                    # Assume we're starting from roughly 0% sparsity if we can't measure it
                    current_density = 1.0  # worst case assumption
                    target_density = 1.0 - target_final_sparsity
                    retention_factor = (target_density / current_density) ** (1.0 / remaining_epochs)
                    target_prune_fraction_per_epoch = 1.0 - retention_factor
                    target_prune_fraction_per_epoch = max(0.0, min(target_prune_fraction_per_epoch, 0.5))  # Cap between 0% and 50%
                else:
                    target_prune_fraction_per_epoch = 0.0
                print(f"[Pruning] Epoch {epoch+1}: Using fallback prune_fraction={target_prune_fraction_per_epoch:.1%} (no sparse_input_weights found)")
            
            # Apply pruning only if we have a positive prune fraction
            if target_prune_fraction_per_epoch > 0.0:
                for v in model.trainable_variables:
                    name = v.name

                    # (a) always scrub NaN/Inf for safety
                    _sanitize_weights(v)

                    # ---------- prune ONLY aud_in weights ----------
                    if "sparse_input_weights" in name:
                        # Count current active weights (those not yet pruned)
                        current_mask = MASKS[v.ref()]
                        weights_before = tf.math.count_nonzero(current_mask)
                        total_weights = tf.size(current_mask, out_type=tf.int64)
                        
                        # Calculate how many weights to prune this epoch
                        # Prune target_prune_fraction_per_epoch of currently active weights
                        current_active_weights = tf.cast(weights_before, tf.float32)
                        weights_to_prune = tf.cast(current_active_weights * target_prune_fraction_per_epoch, tf.int64)
                        target_active_weights = weights_before - weights_to_prune
                        target_active_weights = tf.maximum(target_active_weights, tf.constant(1, dtype=tf.int64))  # Keep at least 1 weight
                        
                        # Get absolute values of weights (only consider currently active weights)
                        abs_weights = tf.abs(v) * current_mask
                        
                        # Flatten and find threshold to keep exactly target_active_weights
                        flat_abs_weights = tf.reshape(abs_weights, [-1])
                        
                        # Use top_k to find the target_active_weights largest magnitudes
                        flat_size = tf.size(flat_abs_weights, out_type=tf.int64)
                    if target_active_weights < flat_size and target_active_weights > 0:
                        # Convert to int32 for top_k (which requires int32)
                        k_int32 = tf.cast(target_active_weights, tf.int32)
                        top_k_values = tf.math.top_k(flat_abs_weights, k=k_int32, sorted=False).values
                        threshold = tf.reduce_min(top_k_values)
                        
                        # Create new mask: keep weights with magnitude >= threshold
                        new_mask = tf.cast(tf.greater(tf.abs(v), threshold), v.dtype)
                        
                        # Handle ties at threshold by using exact top_k indices
                        current_above_threshold = tf.math.count_nonzero(new_mask)
                        current_above_threshold_int64 = tf.cast(current_above_threshold, tf.int64)
                        
                        if current_above_threshold_int64 > target_active_weights:
                            # Use top_k indices to create exact mask
                            flat_indices = tf.math.top_k(flat_abs_weights, k=k_int32, sorted=False).indices
                            new_mask_flat = tf.zeros_like(flat_abs_weights, dtype=v.dtype)
                            new_mask_flat = tf.tensor_scatter_nd_update(
                                new_mask_flat, 
                                tf.expand_dims(flat_indices, 1), 
                                tf.ones(k_int32, dtype=v.dtype)
                            )
                            new_mask = tf.reshape(new_mask_flat, tf.shape(v))
                    else:
                        # Keep all current weights if target is too small or large
                        new_mask = current_mask
                    
                    # Apply the new mask and ensure pruned weights are exactly zero
                    v.assign(v * new_mask)
                    MASKS[v.ref()] = new_mask

                    # Enhanced logging
                    weights_after = tf.math.count_nonzero(new_mask)
                    weights_pruned = weights_before - weights_after
                    
                    if weights_before > 0:
                        proportional_impact = tf.cast(weights_pruned, tf.float32) / tf.cast(weights_before, tf.float32)
                        actual_sparsity = 1.0 - (tf.cast(weights_after, tf.float32) / tf.cast(total_weights, tf.float32))
                        
                        # tf.print("Epoch", epoch+1, "|", name, 
                        #         "| Pruned:", weights_pruned, "/", weights_before, 
                        #         f"({proportional_impact*100:.1f}% of active)",
                        #         "| Remaining active:", weights_after,
                        #         "| Total sparsity:", f"{actual_sparsity*100:.1f}%")

                else:
                    # keep these weights fully trainable (mask = 1)
                    MASKS[v.ref()] = tf.ones_like(v, dtype=v.dtype)
            else:
                # target_prune_fraction_per_epoch == 0.0 - no pruning needed, just sanitize
                print(f"[Pruning] Epoch {epoch+1}: No pruning needed (prune_fraction={target_prune_fraction_per_epoch:.1%})")
                for v in model.trainable_variables:
                    _sanitize_weights(v)
                    if "sparse_input_weights" in v.name:
                        # Apply existing mask to maintain current sparsity and ensure zeros are exact
                        v.assign(v * MASKS[v.ref()])
                    else:
                        # keep these weights fully trainable (mask = 1)  
                        MASKS[v.ref()] = tf.ones_like(v, dtype=v.dtype)
        else:
            # Final sparsity phase: mask is frozen, no pruning
            print(f"[Pruning] Epoch {epoch+1}: Final sparsity phase - mask frozen at sparsity={target_final_sparsity:.3f}")
            
            # Only sanitize weights, keep existing masks unchanged
            for v in model.trainable_variables:
                name = v.name
                # (a) always scrub NaN/Inf for safety
                _sanitize_weights(v)
                
                # Apply existing mask to zero out pruned weights (in case of numerical drift)
                if "sparse_input_weights" in name:
                    current_mask = MASKS[v.ref()]
                    v.assign(v * current_mask)
                    
                    # Optional: Log final sparsity stats without changing mask
                    weights_active = tf.math.count_nonzero(current_mask)
                    total_weights = tf.size(current_mask, out_type=tf.int64)
                    current_sparsity = 1.0 - (tf.cast(weights_active, tf.float32) / tf.cast(total_weights, tf.float32))
                    
                    if epoch == pruning_end_epoch:  # Only log once when entering final phase
                        tf.print("Epoch", epoch+1, "|", name, 
                                "| Final mask frozen | Active weights:", weights_active, "/", total_weights,
                                "| Final sparsity:", current_sparsity)
                # Note: MASKS[v.ref()] is NOT updated - it remains frozen from the last pruning epoch

        # -----------------------------------------------------------------


        # ===================
        # ===================
        # TESTING PHASE (NEW)
        # ===================
        all_test_losses = []
        all_test_rate_terms = []
        all_test_log_terms = []

        # Reset global testing state at the beginning of each epoch
        global_testing_state = rsnn_layer.cell.zero_state(1, tf.float32)

        for sid in StimID_testing_original:
            chunk_losses, chunk_rate_terms, chunk_log_terms, final_state = run_step(
                sid,
                training=False,
                current_input_l1_weight=current_input_l1_weight,
                current_input_l2_weight=current_input_l2_weight,
                current_recurrent_l1_weight=current_recurrent_l1_weight,
                current_recurrent_l2_weight=current_recurrent_l2_weight,
                current_background_l1_weight=current_background_l1_weight,
                current_background_l2_weight=current_background_l2_weight,
                epoch=epoch,
                global_state=global_testing_state
            )
            # Update global state for next stimulus
            global_testing_state = final_state

            all_test_losses.extend(chunk_losses)
            all_test_rate_terms.extend(chunk_rate_terms)
            all_test_log_terms.extend(chunk_log_terms)


        avg_test_loss = np.mean(all_test_losses) if all_test_losses else 0.0
        avg_test_rate = np.mean(all_test_rate_terms) if all_test_rate_terms else 0.0
        avg_test_log = np.mean(all_test_log_terms) if all_test_log_terms else 0.0

        epoch_losses_test.append(avg_test_loss)
        epoch_rate_losses_test.append(avg_test_rate)
        epoch_input_regularizations_test.append(avg_test_log)

        print(f"[Test] Epoch {epoch+1}: avg test loss {avg_test_loss:.4f}")

        # Compute weight proportion below threshold using utility function
        # NOTE: This should now match the sparsity reported in pruning since we ensure exact zeros
        epoch_weight_proportion = np.mean(tu.compute_weight_proportions(model, threshold=1e-10))
        epoch_weight_proportions.append(epoch_weight_proportion)

        # Sample and store weights
        epoch_sample = sample_weights(model)
        sampled_weights_log['aud_in'].append(epoch_sample['aud_in'])
        sampled_weights_log['bkg'].append(epoch_sample['bkg'])
        sampled_weights_log['recurrent'].append(epoch_sample['recurrent'])

        if (epoch + 1) % 5 == 0:
            # Save training progress plot
            tu.plot_loss_progress2multi(
                epoch_losses_train,
                epoch_losses_test,
                epoch_rate_losses_train,
                epoch_rate_losses_test,
                epoch_input_regularizations,
                epoch_input_regularizations_test,
                epoch_weight_proportions,
                epoch,
                sampled_weights_log=sampled_weights_log,
                target_scaling_values=epoch_target_scaling_values,  # Add target_scaling tracking
                display=False,
                save=True,
                save_dir=os.path.join(flags.checkpoint_savedir, "training_progress"),
                file_string="loss_plots",
                weight_threshold=1e-10,
                # Add pruning target visualization parameters
                target_final_sparsity=flags.target_final_sparsity,
                pruning_stop_epochs_before_end=flags.pruning_stop_epochs_before_end,
                num_epochs=flags.num_epochs,
                initial_sparsity=0.0  # Assuming we start with no sparsity
            )


            # Raster plotting code

            StimID_for_raster = StimID_training_for_plotting
            stim_dur_ms = int(np.floor((1.5 + df.loc[df.stim_number == StimID_for_raster, 'duration_ucsf'].values[0]) * 1000 - 1))

            # Load full aud and target rates
            aud_full, trg_full, _ = tu.load_aud_input_and_target_rates(
                StimID             = StimID_for_raster,
                input_dir          = flags.input_dir,
                target_rates_dir   = flags.target_rates_dir,
                StimID_to_StimName = StimID_to_StimName,
                seq_len            = stim_dur_ms,
                n_input            = flags.n_input,
                batch_size         = 1,
                t_start            = 0,
                N_neurons          = n_targets,          # pooled size
                target_cell_ids    = uniq_target_ids,    # unique IDs
                repeat_number      = 0
            )


            # -------- FIX: safe indexing -------------
            aud_full = aud_full[0] if aud_full.ndim == 3 else aud_full
            trg_full = trg_full[0] if trg_full.ndim == 3 else trg_full

            # Now tensorify
            aud_full = tf.convert_to_tensor(aud_full, dtype=tf.float32)
            trg_full = tf.convert_to_tensor(trg_full, dtype=tf.float32)

            # -------- FIX: correct inp_kTN and bkg shapes -------------
            inp_kTN = tf.expand_dims(aud_full, axis=0) if len(aud_full.shape) == 2 else aud_full
            bkg = tf.zeros((tf.shape(inp_kTN)[0], tf.shape(inp_kTN)[1], flags.neurons), dtype=tf.float32)

            # Convert to numpy and diagnose
            def ensure_numpy(name, x):
                if isinstance(x, tf.Tensor):
                    return x.numpy()
                elif isinstance(x, np.ndarray):
                    return x
                else:
                    raise TypeError(f"[{name}] is unexpected type: {type(x)}")

            # Pick one training stimulus for raster plotting
            StimID_for_raster = StimID_training_for_plotting
            stim_duration = 3300  # fixed for raster display
            

            aud_full, trg_full, _ = tu.load_aud_input_and_target_rates(
                StimID             = StimID_for_raster,
                input_dir          = flags.input_dir,
                target_rates_dir   = flags.target_rates_dir,
                StimID_to_StimName = StimID_to_StimName,
                seq_len            = stim_duration,
                n_input            = flags.n_input,
                batch_size         = 1,
                t_start            = 0,
                N_neurons          = n_targets,          # pooled size
                target_cell_ids    = uniq_target_ids,    # unique IDs
                repeat_number      = 0
            )


            aud_full = aud_full[0] if aud_full.ndim == 3 else aud_full
            trg_full = trg_full[0] if trg_full.ndim == 3 else trg_full   # (T, n_targets)

            # --- build input tensor ------------------------------------
            input_spikes = tf.convert_to_tensor(aud_full, dtype=tf.float32)[None, ...]  # (1,T,n_input)
            input_spikes = tu.pad_to_seq_len(input_spikes, stim_duration)

            # --- build pooled target tensor ----------------------------
            target_rates_pool = tf.convert_to_tensor(trg_full, dtype=tf.float32)[None, ...]  # (1,T,n_targets)
            target_rates_pool = tu.pad_to_seq_len(target_rates_pool, stim_duration)

            # --- roll out model using single pass for plotting ----------------------------------------
            def full_rollout_for_plotting(input_spikes_full):
                """Roll out the full sequence in a single pass for raster plotting."""
                # Initialize with zero state
                init_state = model.get_layer("rsnn").cell.zero_state(1, dtype=tf.float32)
                
                # Single rollout of the entire sequence
                full_spikes, _ = roll_out_chunk(input_spikes_full, init_state)
                
                return {
                    "spikes": full_spikes,
                    "voltage": tf.zeros_like(full_spikes),  # Dummy voltage for compatibility
                }
            
            rollout_results = full_rollout_for_plotting(input_spikes)  # spikes (1,T,N)

            # --- Handle target rates based on pooling mode ------------------------
            if flags.enable_pooling:
                # Standard pooling mode: un-pool to full N for plotting
                target_rates_full = tf.gather(target_rates_pool[0], UNPOOL_MAP, axis=1)[None, ...]        # (1,T,N)
            else:
                # Direct mode: target rates are already 1-to-1 with neurons
                target_rates_full = target_rates_pool  # (1,T,N) - already correct size
            #rollout_results["spikes"] = tf.gather(rollout_results["spikes"], UNPOOL_MAP, axis=2)      # (1,T,N)

            # ---------------------------------------------------------
            # CLUSTERING ORDER  (build once per stimulus)
            # ---------------------------------------------------------
            order = _cached_plot_orders.get(StimID_for_raster)
            if order is None:
                if flags.enable_pooling:
                    # Use smart clustering with pooling
                    order = tu.build_full_cluster_order(
                        target_rates_pool[0].numpy(),    # shape (T, n_targets)
                        UNPOOL_MAP.numpy()               # shape (N)
                    )
                    print(f"[Raster] computed pooled clustering for Stim {StimID_for_raster} (cached)")
                else:
                    # Direct mode: compute standard hierarchical clustering and cache it
                    order = tu.hierarchical_cluster_target_rates(target_rates_full[0].numpy())
                    print(f"[Raster] computed standard clustering for Stim {StimID_for_raster} (cached)")
                _cached_plot_orders[StimID_for_raster] = order
            else:
                print(f"[Raster] using cached order for Stim {StimID_for_raster}")

            # --- plot ---------------------------------------------------
            tu.plot_target_rates_and_model_spikes(
                target_rates        = target_rates_full,
                rollout_results     = rollout_results,
                epoch               = epoch,
                perform_clustering = False,          # ← Clustering already done and cached
                neuron_order       = order,          # ← Always use cached order
                target_scaling      = flags.target_scaling,
                sigma               = flags.sigma_smoothing_test,  # Use fixed test sigma for plotting
                z_scored            = False,
                display             = False,
                save                = True,
                save_dir            = os.path.join(flags.checkpoint_savedir, "rasters"),
                file_string         = "training",
                POOL_MAT           = POOL_MAT if flags.enable_pooling else None,
                UNPOOL_MAP         = UNPOOL_MAP if flags.enable_pooling else None
            )


            # Repeat for a test stimulus
            StimID_for_raster = StimID_testing_for_plotting
            stim_duration = 3300  # fixed for raster display

            aud_full, trg_full, _ = tu.load_aud_input_and_target_rates(
                StimID             = StimID_for_raster,
                input_dir          = flags.input_dir,
                target_rates_dir   = flags.target_rates_dir,
                StimID_to_StimName = StimID_to_StimName,
                seq_len            = stim_duration,
                n_input            = flags.n_input,
                batch_size         = 1,
                t_start            = 0,
                N_neurons          = n_targets,          # pooled size
                target_cell_ids    = uniq_target_ids,    # unique IDs
                repeat_number      = 0
            )


            aud_full = aud_full[0] if aud_full.ndim == 3 else aud_full
            trg_full = trg_full[0] if trg_full.ndim == 3 else trg_full   # (T, n_targets)

            # --- build input tensor ------------------------------------
            input_spikes = tf.convert_to_tensor(aud_full, dtype=tf.float32)[None, ...]  # (1,T,n_input)
            input_spikes = tu.pad_to_seq_len(input_spikes, stim_duration)

            # --- build pooled target tensor ----------------------------
            target_rates_pool = tf.convert_to_tensor(trg_full, dtype=tf.float32)[None, ...]  # (1,T,n_targets)
            target_rates_pool = tu.pad_to_seq_len(target_rates_pool, stim_duration)

            # --- roll out model using single pass for plotting ----------------------------------------
            def full_rollout_for_plotting_test(input_spikes_full):
                """Roll out the full sequence in a single pass for raster plotting."""
                # Initialize with zero state
                init_state = model.get_layer("rsnn").cell.zero_state(1, dtype=tf.float32)
                
                # Single rollout of the entire sequence
                full_spikes, _ = roll_out_chunk(input_spikes_full, init_state)
                
                return {
                    "spikes": full_spikes,
                    "voltage": tf.zeros_like(full_spikes),  # Dummy voltage for compatibility
                }
            
            rollout_results = full_rollout_for_plotting_test(input_spikes)  # spikes (1,T,N)

            # --- Handle target rates based on pooling mode ------------------------
            if flags.enable_pooling:
                # Standard pooling mode: un-pool to full N for plotting
                target_rates_full = tf.gather(target_rates_pool[0], UNPOOL_MAP, axis=1)[None, ...]        # (1,T,N)
            else:
                # Direct mode: target rates are already 1-to-1 with neurons
                target_rates_full = target_rates_pool  # (1,T,N) - already correct size
            #rollout_results["spikes"] = tf.gather(rollout_results["spikes"], UNPOOL_MAP, axis=2)      # (1,T,N)

            # ---------------------------------------------------------
            # CLUSTERING ORDER  (build once per stimulus)
            # ---------------------------------------------------------
            order = _cached_plot_orders.get(StimID_for_raster)
            if order is None:
                if flags.enable_pooling:
                    # Use smart clustering with pooling
                    order = tu.build_full_cluster_order(
                        target_rates_pool[0].numpy(),    # shape (T, n_targets)
                        UNPOOL_MAP.numpy()               # shape (N)
                    )
                    print(f"[Raster] computed pooled clustering for Stim {StimID_for_raster} (cached)")
                else:
                    # Direct mode: compute standard hierarchical clustering and cache it
                    order = tu.hierarchical_cluster_target_rates(target_rates_full[0].numpy())
                    print(f"[Raster] computed standard clustering for Stim {StimID_for_raster} (cached)")
                _cached_plot_orders[StimID_for_raster] = order
            else:
                print(f"[Raster] using cached order for Stim {StimID_for_raster}")

            # --- plot ---------------------------------------------------
            tu.plot_target_rates_and_model_spikes(
                target_rates        = target_rates_full,
                rollout_results     = rollout_results,
                epoch               = epoch,
                perform_clustering = False,          # ← Clustering already done and cached
                neuron_order       = order,          # ← Always use cached order
                target_scaling      = flags.target_scaling,
                sigma               = flags.sigma_smoothing_test,  # Use fixed test sigma for plotting
                z_scored            = False,
                display             = False,
                save                = True,
                save_dir            = os.path.join(flags.checkpoint_savedir, "rasters"),
                file_string         = "testing",
                POOL_MAT           = POOL_MAT if flags.enable_pooling else None,
                UNPOOL_MAP         = UNPOOL_MAP if flags.enable_pooling else None
            )


        # ───────────────── SAVE CHECKPOINT EVERY 10 EPOCHS ─────────────
        checkpoint.step.assign_add(1)            # advance counter

        if (epoch + 1) % 10 == 0:
            save_path = save_manager.save(checkpoint_number=epoch + 1)
            print(f"💾  checkpoint written to {save_path}")
        # ───────────────────────────────────────────────────────────────








# Replace with your directory
_data_dir = '/allen/programs/mindscope/workgroups/brain-model/darrell.haufler/STG_training_V3/STG_training'
_net_dir = 'test_apr1_2025_4K_10'
# ───── flags ──────────────────────────────────────────────────────────────────

def _define_flags():
    f = absl.app.flags
    
    # ═══════════════════════════════════════════════════════════════════
    # ACTIVE FLAGS - Currently used in the code
    # ═══════════════════════════════════════════════════════════════════
    
    # Directory and file paths
    absl.app.flags.DEFINE_string('base_dir', _data_dir,'')
    absl.app.flags.DEFINE_string('net_dir',_net_dir,'')
    absl.app.flags.DEFINE_string('input_dir', 'aud_in_spikes','')
    absl.app.flags.DEFINE_string('target_rates_dir', 'target_rates','')
    absl.app.flags.DEFINE_string('checkpoint_loaddir', 'test_4','')
    absl.app.flags.DEFINE_string('checkpoint_savedir', 'test_4','')
    
    # Model architecture
    absl.app.flags.DEFINE_integer('neurons', 4452,'') #4452 #40052
    absl.app.flags.DEFINE_integer('n_input', 5000,'')
    absl.app.flags.DEFINE_integer('n_output', 0,'')
    absl.app.flags.DEFINE_integer('batch_size', 1,'')
    
    # Training control
    absl.app.flags.DEFINE_boolean('load_checkpoint', False,'')
    absl.app.flags.DEFINE_boolean('train_input', True,'Enable/disable training of input weights (sparse_input_weights)')
    absl.app.flags.DEFINE_boolean('train_recurrent', True,'Enable/disable training of recurrent weights (sparse_recurrent_weights)')
    absl.app.flags.DEFINE_float('learning_rate_input', 0.00025,'')
    absl.app.flags.DEFINE_integer('num_epochs', 10,'')
    absl.app.flags.DEFINE_boolean('shuffle', True,'')
    
    # Model behavior
    absl.app.flags.DEFINE_boolean('neuron_output', False,'')
    absl.app.flags.DEFINE_boolean('pseudo_gauss', False,'')
    absl.app.flags.DEFINE_boolean('hard_reset', False,'')
    absl.app.flags.DEFINE_boolean('feedforward', False,'')
    
    # Loss computation and scaling
    absl.app.flags.DEFINE_float('target_scaling',1.0,'')
    absl.app.flags.DEFINE_list('sigma_smoothing', [100.0], 'Gaussian smoothing sigma: single value [250] or range [min,max] for training randomization')
    absl.app.flags.DEFINE_float('sigma_smoothing_test', 100.0, 'Fixed Gaussian smoothing sigma for testing and plotting')
    absl.app.flags.DEFINE_float('rate_weight',0.01,'')
    absl.app.flags.DEFINE_float('log_sum_weight',0.0,'') #10000.0
    absl.app.flags.DEFINE_float('target_final_sparsity', 0.95, 'Target sparsity level (fraction of weights to prune) at end of pruning schedule')
    
    # L1/L2 Regularization - target fractions of total loss
    absl.app.flags.DEFINE_float('input_l1_fraction', 0.0, 'L1 regularization target fraction for input weights') #0.00025
    absl.app.flags.DEFINE_float('input_l2_fraction', 0.0, 'L2 regularization target fraction for input weights')
    absl.app.flags.DEFINE_float('recurrent_l1_fraction', 0.0, 'L1 regularization target fraction for recurrent weights')
    absl.app.flags.DEFINE_float('recurrent_l2_fraction', 0.0, 'L2 regularization target fraction for recurrent weights')
    absl.app.flags.DEFINE_float('background_l1_fraction', 0.0, 'L1 regularization target fraction for background weights')
    absl.app.flags.DEFINE_float('background_l2_fraction', 0.0, 'L2 regularization target fraction for background weights')
    
    # L1 Sparsity-Aware Scaling Configuration
    # ═══════════════════════════════════════════════════════════════════════════════════════════════
    # PROBLEM: As progressive pruning removes weights, the L1 penalty sum(|weights|) decreases, causing
    # the adaptive L1 weight to increase dramatically: l1_weight = (fraction × loss) / penalty.
    # This makes L1 regularization overly aggressive late in training, destabilizing learning when 
    # few weights remain active.
    #
    # SOLUTION: Scale down L1 regularization strength based on current sparsity level. As the network
    # becomes sparser, L1 influence is proportionally reduced to maintain training stability while
    # preserving beneficial sparsification effects early in training.
    #
    # BURN-IN PERIOD: Gradually ramp up L1 regularization over initial epochs to avoid instability
    # from poor weight initialization. L1 strength increases from 0 to full strength over the burn-in.
    #
    # RECOMMENDED SETTINGS:
    #   --l1_scaling_method=power --l1_sparsity_scaling_exponent=0.5 --l1_min_scale_factor=0.01
    #   --l1_burnin_epochs=5 --l1_burnin_method=linear
    #   This provides smooth polynomial decay with moderate scaling plus gentle L1 introduction.
    # ═══════════════════════════════════════════════════════════════════════════════════════════════
    absl.app.flags.DEFINE_integer('l1_burnin_epochs', 5, 'Number of epochs for L1 regularization burn-in (gradual ramp-up)')
    absl.app.flags.DEFINE_string('l1_burnin_method', 'linear', 'Burn-in method: "linear", "quadratic", or "none"')
    absl.app.flags.DEFINE_float('l1_sparsity_scaling_exponent', 0.25, 'Exponent for sparsity scaling: scale = (1-sparsity)^exponent')
    absl.app.flags.DEFINE_float('l1_min_scale_factor', 0.05, 'Minimum scaling factor for L1 regularization (prevents complete shutdown)')
    absl.app.flags.DEFINE_boolean('enable_l1_sparsity_scaling', True, 'Enable sparsity-aware L1 regularization scaling')
    absl.app.flags.DEFINE_string('l1_scaling_method', 'power', 'Method for L1 scaling: "power", "sigmoid", "exponential", or "none"')
    absl.app.flags.DEFINE_float('l1_sigmoid_steepness', 10.0, 'Steepness parameter for sigmoid L1 scaling')
    absl.app.flags.DEFINE_float('l1_exponential_rate', 5.0, 'Decay rate for exponential L1 scaling')
    
    # System and execution
    absl.app.flags.DEFINE_integer('seed', 3000,'')
    absl.app.flags.DEFINE_string('strategy', 'standard','')
    absl.app.flags.DEFINE_integer('max_to_keep', 100,'')
    
    # Chunked processing
    f.DEFINE_list("chunk_len_ms", ['200'], "chunk lengths in ms: single value [500] or multiple values [400,500,600] for random selection per stimulus")
    
    # Model-to-target comparison method
    f.DEFINE_boolean("enable_pooling", True, "Enable pooling of model neurons to target groups. If False, use direct 1-to-1 neuron comparison with targets")
    
    # Diagnostic plotting
    f.DEFINE_boolean("plot_internal_state", True, "Enable diagnostic plotting of voltage/current traces for first 5 neurons")
    
    # ═══════════════════════════════════════════════════════════════════
    # DEPRECATED/UNUSED FLAGS - Defined but not currently used
    # ═══════════════════════════════════════════════════════════════════
    
    # Legacy training parameters
    absl.app.flags.DEFINE_integer('seq_len', 3300,'')  # Not used in chunked processing
    absl.app.flags.DEFINE_boolean('save_int_checkpoint', True,'')  # Not referenced in code
    absl.app.flags.DEFINE_boolean('save_best', True,'')  # Not referenced in code
    absl.app.flags.DEFINE_boolean('profile', False,'')  # Not referenced in code
    absl.app.flags.DEFINE_float('loss_offset', 1e-8,'')  # Not referenced in code
    
    # Legacy epoch/batch parameters  
    absl.app.flags.DEFINE_integer('N_batch', 1,'')  # Not referenced in code
    absl.app.flags.DEFINE_integer('steps_per_epoch', 8,'')  # Not referenced in code
    absl.app.flags.DEFINE_boolean('reset_weights', True,'')  # Not referenced in code
    absl.app.flags.DEFINE_integer('patience', 20,'')  # Not referenced in code
    
    # Legacy model configuration
    absl.app.flags.DEFINE_boolean('core_only', False,'')  # Not referenced in code
    absl.app.flags.DEFINE_boolean('connected_selection', False,'')  # Not referenced in code
    absl.app.flags.DEFINE_integer('neurons_per_output', 1,'')  # Not referenced in code
    absl.app.flags.DEFINE_integer('window_size', 100,'')  # Not referenced in code
    absl.app.flags.DEFINE_string('delays', '10,0','')  # Not referenced in code
    
    # Legacy execution parameters
    absl.app.flags.DEFINE_string('comment', '','')  # Not referenced in code
    absl.app.flags.DEFINE_boolean('caching', False,'')  # Not referenced in code
    absl.app.flags.DEFINE_float('reversion_threshold', 0.1,'')  # Not referenced in code
    absl.app.flags.DEFINE_boolean('use_model_reversion', False,'')  # Not referenced in code
    absl.app.flags.DEFINE_boolean('train_noise', True,'')  # Not referenced in code
    
    # Legacy loss components
    absl.app.flags.DEFINE_float('zscore_mse',0.0,'')  # Not referenced in code
    absl.app.flags.DEFINE_float('zscore_var',0.0,'')  # Not referenced in code
    absl.app.flags.DEFINE_float('min_activity',0,'')  # Not referenced in code
    absl.app.flags.DEFINE_float('max_activity',0.0,'')  # Not referenced in code
    absl.app.flags.DEFINE_float('time_point_min',0.0,'')  # Not referenced in code
    absl.app.flags.DEFINE_float('time_point_max',0.0,'')  # Not referenced in code
    
    # Legacy chunking parameters
    f.DEFINE_integer("n_repeats_per_chunk", 1,    "noisy repeats per chunk")  # Not referenced in code

    # ═══════════════════════════════════════════════════════════════════════════════
    # STIMULUS SETS AND TIMING PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════════════
    
    # Training stimulus sets - easily swappable by commenting/uncommenting
    # absl.app.flags.DEFINE_list('training_stimuli', 
    #     ['101', '102', '103', '104', '105', '106', '107', '113', '1', '3', '4', '9', '11', '15', '20', '25', '26', '29', '32', '36', '39', '40', '41', '43', '45', '48', '49', '51', '53', '56', '57', '58', '61', '62', '68', '75', '76', '80', '84', '88'],
    #     'List of stimulus IDs for training (current default)')
    
    # Alternative training sets (commented out - uncomment to use)
    # absl.app.flags.DEFINE_list('training_stimuli', 
    #     ['101', '102', '103', '104', '105', '106', '107', '108', '113'],
    #     'List of stimulus IDs for training (extended set)')
    
    absl.app.flags.DEFINE_list('training_stimuli', 
        ['105'],
        'List of stimulus IDs for training (minimal set for testing)')
    
    # absl.app.flags.DEFINE_list('training_stimuli', 
    #     ['32', '36', '39', '40', '41', '43', '45', '48', '49', '51'],
    #     'List of stimulus IDs for training (mid-range set)')
    
    # absl.app.flags.DEFINE_list('training_stimuli', 
    #     ['1', '3', '4', '9', '101', '102', '103', '104', '105', '106', '107', '108', '113', '15', '20', '25', '26', '29', '32', '36', '39', '40', '41', '43', '45', '48', '49', '51', '53', '56', '57', '58', '61', '62', '68', '75', '76', '80', '84', '88', '90', '91', '92', '93', '95'],
    #     'List of stimulus IDs for training (full set)')
    
    # Testing stimulus sets
    absl.app.flags.DEFINE_list('testing_stimuli', 
        ['100', '108'],
        'List of stimulus IDs for testing (current default)')
    
    # Alternative testing sets (commented out - uncomment to use)
    # absl.app.flags.DEFINE_list('testing_stimuli', 
    #     ['100', '108', '90', '91', '92', '93', '95'],
    #     'List of stimulus IDs for testing (extended set)')
    
    # Progressive schedule timing parameters
    absl.app.flags.DEFINE_integer('pruning_stop_epochs_before_end', 20,
        'Number of epochs before the final epoch to stop pruning (allows network to stabilize)')
    
    absl.app.flags.DEFINE_integer('scaling_stop_epochs_before_end', 40,
        'Number of epochs before the final epoch to reach target_scaling=1.0 (allows network to stabilize)')



_define_flags()

if __name__ == "__main__":
    print("HOST:", socket.gethostname())
    absl.app.run(main)
