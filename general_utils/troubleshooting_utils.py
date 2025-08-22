
# Import custom utilities
from general_utils import training_utils as tu

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'  # Keep oneDNN optimizations enabled

import tensorflow as tf
"""
TF warnings to ignore:
    The oneDNN message is about numerical precision variations, not performance
    The cuDNN/cuFFT/cuBLAS factory registration warnings are just about duplicate 
    registrations, not functionality
    The CPU feature guard message suggests some CPU optimizations aren't enabled, 
    but GPU operations (which are your primary compute path) aren't affected
"""
from time import time
import cProfile
import functools
import time
from pstats import Stats



#################### Profiling ####################

# Timing decorator for function-level profiling
def timer_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

# Apply decorators to utility functions
@timer_decorator
def load_aud_input_and_target_rates(*args, **kwargs):
    return tu.load_aud_input_and_target_rates(*args, **kwargs)

@timer_decorator
def pad_to_seq_len(*args, **kwargs):
    return tu.pad_to_seq_len(*args, **kwargs)

@timer_decorator
def roll_out(*args, **kwargs):
    return tu.roll_out(*args, **kwargs)

@timer_decorator
def compute_combined_loss(*args, **kwargs):
    return tu.compute_combined_loss(*args, **kwargs)

# Main training loop with profiling   
def train_with_profiling(StimID_training, model, optimizer, flags, extractor_model, state_variables, **kwargs):
    profiler = cProfile.Profile()
    profiler.enable()
    
    input_dir = flags.input_dir
    target_rates_dir = flags.target_rates_dir
    seq_len = flags.seq_len
    N_neurons = flags.neurons
    n_input = flags.n_input
    batch_size = flags.batch_size

    StimID_training = [100, 101, 102, 103, 104, 105, 106, 107]

    StimID_to_StimName = {
        100: "fcaj0_si1479", 101: "fdfb0_si1948", 102: "fisb0_si2209",
        103: "mbbr0_si2315", 104: "mdlc2_si2244", 105: "fcaj0_si1804",
        106: "mdls0_si998", 107: "mjdh0_si1984", 108: "mjmm0_si625", 113: "fdxw0_si2141"
    }
    
    StimID_to_StimDuration = {
        100: 1240.9, 101: 1516.6, 102: 1632.1, 103: 1784.5,
        104: 1376.1, 105: 1656.1, 106: 1349.6, 107: 1173.7,
        108: 2222.7, 113: 1242.1}
    
    print("---------------Training Phase Starting---------------")
    step_losses_train = []
    epoch_loss_train = 0
    epoch_metrics = {
            'training': {'loss': 0},
            'testing': {'loss': 0}
        }
    
    # GPU memory tracking
    try:
        gpu_devices = tf.config.list_physical_devices('GPU')
        if gpu_devices:
            print(f"Found {len(gpu_devices)} GPU(s)")
            for i, device in enumerate(gpu_devices):
                print(f"Device: {device.name}")
                # Get memory info using correct device name format
                with tf.device(f'/GPU:{i}'):
                    memory_info = tf.config.experimental.get_memory_info('GPU:0')
                    print(f"Initial GPU {i} memory: {memory_info['current'] / 1024**2:.2f} MB")
    except Exception as e:
        print(f"Could not get GPU memory info: {e}")

    for step, StimID in enumerate(StimID_training):
        step_start_time = time.time()
        
        # Define temporal window
        stim_duration = StimID_to_StimDuration[StimID]
        start_index = int(500 - 200)
        end_index = int(500 + stim_duration + 200)
        
        # Track GPU memory for each step
        if step == 0:
            try:
                for i, device in enumerate(gpu_devices):
                    with tf.device(f'/GPU:{i}'):
                        memory_before = tf.config.experimental.get_memory_info('GPU:0')
                        print(f"GPU {i} memory before first step: {memory_before['current'] / 1024**2:.2f} MB")
            except Exception as e:
                print(f"Could not get GPU memory info: {e}")
        
        # Load and prepare input data
        aud_in_spikes, target_rates, _ = load_aud_input_and_target_rates(
            StimID=StimID,
            input_dir=input_dir,
            target_rates_dir=target_rates_dir,
            StimID_to_StimName=StimID_to_StimName,
            seq_len=seq_len,
            n_input=n_input,
            batch_size=batch_size,
            t_start=1,
            N_neurons=N_neurons
        )
        
        # Data preprocessing with timing
        preprocess_start = time.time()
        input_spikes = pad_to_seq_len(tf.convert_to_tensor(aud_in_spikes, dtype=tf.float32), seq_len)
        target_rates = pad_to_seq_len(tf.convert_to_tensor(target_rates, dtype=tf.float32), seq_len)
        print(f"Preprocessing took {time.time() - preprocess_start:.4f} seconds")
        
        # Forward pass and loss computation
        training_start = time.time()
        with tf.GradientTape() as tape:
            # Track the peak memory usage during forward pass
            if step == 0:
                try:
                    for i, device in enumerate(gpu_devices):
                        with tf.device(f'/GPU:{i}'):
                            memory_before_forward = tf.config.experimental.get_memory_info('GPU:0')
                            print(f"GPU {i} memory before forward pass: {memory_before_forward['current'] / 1024**2:.2f} MB")
                except Exception as e:
                    print(f"Could not get GPU memory info: {e}")
            
            rollout_results = roll_out(input_spikes, flags, extractor_model, state_variables)
            
            loss = compute_combined_loss(
                model,
                rollout_results['spikes'], 
                target_rates,
                exclude_start_ms=start_index, 
                exclude_end_ms=end_index
            )
            epoch_metrics['training']['loss'] += loss.numpy()
        
        # Gradient computation and optimization with timing
        grad_start = time.time()
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        print(f"Gradient computation took {time.time() - grad_start:.4f} seconds")
        print(f"Total training operations took {time.time() - training_start:.4f} seconds")
        
        # Track peak memory after backward pass
        if step == 0:
            try:
                for i, device in enumerate(gpu_devices):
                    with tf.device(f'/GPU:{i}'):
                        memory_after = tf.config.experimental.get_memory_info('GPU:0')
                        print(f"GPU {i} memory after backward pass: {memory_after['current'] / 1024**2:.2f} MB")
                        print(f"Peak memory increase: {(memory_after['current'] - memory_before['current']) / 1024**2:.2f} MB")
            except Exception as e:
                print(f"Could not get GPU memory info: {e}")
                
        
        
        # Tracking metrics
        step_losses_train.append(loss.numpy())
        epoch_loss_train += loss.numpy()
        
        # Step summary
        step_time = time.time() - step_start_time
        print(f"Step {step + 1}/{len(StimID_training)}: Loss = {loss.numpy():.4f}")
        print(f"Total step time: {step_time:.4f} seconds") 
        if step == 0:
            print("First step (includes compilation time):", step_time)
        elif step == 1:
            print("Second step (uses cached kernels):", step_time)
        print("-" * 50)
    
    # Disable cProfile and print results
    profiler.disable()
    stats = Stats(profiler)
    stats.sort_stats('cumulative')
    print("\nProfiling Results:")
    print("-" * 50)
    stats.print_stats(30)  # Print top 30 time-consuming functions
    
    return step_losses_train, epoch_loss_train, epoch_metrics, step