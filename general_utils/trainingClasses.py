import datetime

class TrainingConfig:
    def __init__(self, 
                 patience = 5, 
                 min_delta = 0.001,
                 checkpoint_dir = "model_checkpoints", 
                 max_checkpoints = 10,
                 reversion_threshold = 0.1, # Relative increase in loss to trigger reversion
                 partial_reversion_alpha = 0.5):
        # Early stopping parameters
        self.patience = patience  # Number of epochs to wait before early stopping
        self.min_delta = min_delta  # Minimum change in loss to qualify as an improvement
        
        # Checkpointing parameters
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints  # Maximum number of checkpoint files to keep
        
        # Reversion parameters
        self.reversion_threshold = reversion_threshold  # Relative increase in loss to trigger reversion
        self.partial_reversion_alpha = partial_reversion_alpha  # Mixing factor for partial reversion
        
        # Validation metrics
        self.metrics = {
            'training': {
                'loss': 0.0 #lambda y_true, y_pred: compute_combined_loss_with_plot(y_pred, y_true),
                #'spike_rate_error': lambda y_true, y_pred: tf.reduce_mean(tf.abs(y_true - y_pred)),
            },
            'testing': {
                'loss': 0.0 #lambda y_true, y_pred: compute_combined_loss_with_plot(y_pred, y_true),
                #'spike_rate_error': lambda y_true, y_pred: tf.reduce_mean(tf.abs(y_true - y_pred)),
            }
        }

class ModelCheckpoint:
    def __init__(self, model, optimizer, epoch, metrics):
        """
        Initialize checkpoint with current model and optimizer state
        
        Args:
            model: The model to checkpoint
            optimizer: The optimizer to checkpoint
            epoch: Current epoch number
            metrics: Dictionary of current metric values
        """
        self.weights = [var.numpy() for var in model.trainable_variables]
        self.optimizer_state = [var.numpy() for var in optimizer.variables()]
        self.epoch = epoch
        self.metrics = metrics
        self.timestamp = datetime.datetime.now()

    def restore_model_and_optimizer(self, model, optimizer, restore_optimizer_state):
        """
        Restore both model weights and optimizer state
        
        Args:
            model: The model to restore weights to
            optimizer: The optimizer to restore state to
        """
        # Validate lengths match before restoring
        if len(self.weights) != len(model.trainable_variables):
            raise ValueError("Model architecture changed - weights cannot be restored")
           
        # Restore model weights
        for var, weights in zip(model.trainable_variables, self.weights):
            var.assign(weights)
            
        if restore_optimizer_state:
            if len(self.optimizer_state) != len(optimizer.variables()):
                raise ValueError("Optimizer configuration changed - state cannot be restored")
            else:
                # Restore optimizer state
                for var, state in zip(optimizer.variables(), self.optimizer_state):
                    var.assign(state)