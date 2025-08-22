STG_training_multi_recurrent_chunking_fix2.py
-Basic Chunked Training with Random Chunk Sizes
-Implements non-overlapping chunked processing for memory-efficient training
-Supports random chunk size selection from configurable options (e.g., [400,500,600]ms)
-Basic RNN state carryover between chunks within stimuli
-Simple L1/L2 regularization with sparsity-aware scaling configuration
-Targets 200ms default chunk size for rapid iteration
-Foundation version for chunked training approach

STG_training_multi_recurrent_chunking_fix3.py
-Overlapping Chunks with State Pre-computation for setting chunk initial states
-Adds overlapping chunk processing (500ms chunks, 250ms stride = 50% overlap)
-Introduces RNN state pre-computation at chunk boundaries for efficiency
-Implements gradient accumulation across multiple chunks before weight updates
-Enhanced temporal context through overlapping regions
-State validation system for accuracy verification
-Improved training stability through better temporal continuity

STG_training_multi_recurrent_chunking_fix3.py
-Full dual-approach edge effect mitigation system:
	1. Spike padding: Pre-computed spikes from model rollout to mitigate boundry effects (250ms before/after chunk)
	2. Applies Gaussian smoothing of target rates to full stimuli (rather than within chunk)
