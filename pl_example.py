import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import TransformerEncoderLayer
from torch.utils.data import DataLoader, Dataset
from lightning.fabric import Fabric
from lightning.fabric.strategies import XLAFSDPStrategy # Can also use string "xla_fsdp"
import os
import shutil  # Add shutil import
import time  # Add time import

# Define a simple model (e.g., for NLP or Vision)
# For XLAFSDPStrategy, consider which layers are large and good candidates for sharding
class SimpleModel(nn.Module):
    def __init__(self, vocab_size=1000, embed_dim=256, num_heads=4, num_layers=2, seq_len=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        encoder_layers = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim * seq_len, vocab_size) # Flatten and predict
        self.seq_len = seq_len
        self.embed_dim = embed_dim

    def forward(self, src):
        src = self.embedding(src) + self.pos_encoder[:, :src.size(1)]
        output = self.transformer_encoder(src)
        output = output.reshape(output.size(0), -1) # Flatten
        output = self.fc(output)
        return output

# Define a dummy Dataset for illustration
class DummyDataset(Dataset):
    def __init__(self, size=10000, seq_len=64, vocab_size=1000):
        self.size = size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        # Pre-generate data to avoid randomness issues across ranks if not careful with seeding
        self.data = torch.randint(0, self.vocab_size, (self.size, self.seq_len))
        self.targets = torch.randint(0, self.vocab_size, (self.size, self.seq_len)) # Example target format

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# Main training function
def main_training_function(fabric: Fabric):
    fabric.print(f"Fabric initialized on global rank {fabric.global_rank} of {fabric.world_size}")
    fabric.print(f"Local rank {fabric.local_rank}, Node rank {fabric.node_rank}")

    # Create a unique checkpoint directory with timestamp
    timestamp = int(time.time())
    checkpoint_base_dir = f"./checkpoints_{timestamp}"
    if fabric.global_rank == 0:
        os.makedirs(checkpoint_base_dir, exist_ok=True)
    fabric.barrier()

    fabric.print(f"=== Worker Information ===")
    fabric.print(f"Global Rank: {fabric.global_rank} of {fabric.world_size}")
    fabric.print(f"Local Rank: {fabric.local_rank}")
    fabric.print(f"Node Rank: {fabric.node_rank}")
    fabric.print(f"Device: {fabric.device}")

    # NOTE: (IMPORTANT) Model instantiation with fabric.init_module for large models
    # empty_init=True defers parameter initialization until FSDP wraps the layers, saving host memory.
    with fabric.init_module(empty_init=True):
        model = SimpleModel(seq_len=64) # Ensure seq_len matches dataset

    # Setup model and optimizer with Fabric
    # This step wraps the model with XlaFullyShardedDataParallel
    model = fabric.setup_module(model)

    # NOTE: (IMPORTANT) create optimizer after setup_module and setup optimizers separately
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    optimizer = fabric.setup_optimizers(optimizer)

    # Prepare DataLoader
    train_dataset = DummyDataset(seq_len=64)
    # Lightning Fabric automatically handles DistributedSampler for TPUs [1, 8, 14]
    train_dataloader = DataLoader(train_dataset, batch_size=32) # Per-device batch size
    train_dataloader = fabric.setup_dataloaders(train_dataloader) # NOTE: add checkpoint reloading

    # Training Loop
    EPOCHS = 3
    for epoch in range(EPOCHS):
        model.train()
        for batch_idx, (data, target) in enumerate(train_dataloader):
            if batch_idx < 10:
                fabric.print(f"Batch {batch_idx} | Data: {data} | Local Rank: {fabric.local_rank} | Global Rank: {fabric.global_rank}")
            optimizer.zero_grad()
            output = model(data)
            # Ensure target shape matches output for loss calculation
            # For SimpleModel, output is (batch, vocab_size), target should be (batch) for CrossEntropyLoss
            # The DummyDataset and SimpleModel might need adjustment for a typical CrossEntropyLoss setup.
            # Assuming a simplified loss for illustration:
            loss = torch.nn.functional.mse_loss(output, output.detach() * 0.9) # Dummy loss
            # For a real scenario with CrossEntropyLoss:
            # loss = nn.CrossEntropyLoss()(output.view(-1, output.size(-1)), target.view(-1))


            fabric.backward(loss)
            optimizer.step()

            if batch_idx % 10 == 0:
                fabric.print(f"Epoch {epoch}/{EPOCHS} | Batch {batch_idx} | Loss: {loss.item():.4f}")

        fabric.barrier() # Ensure all processes complete epoch

        # Checkpointing (example)
        if epoch % 1 == 0: # Save every epoch
            checkpoint_dir = f"{checkpoint_base_dir}/epoch_{epoch}_shards"
            state = {"model": model, "optimizer": optimizer, "epoch": epoch, "dataloader": train_dataloader}
            fabric.print(f"Saving sharded checkpoint to {checkpoint_dir}...")
            fabric.save(checkpoint_dir, state=state)
            fabric.barrier()
            fabric.print("Sharded checkpoint saved.")

    fabric.print("Training finished.")

if __name__ == "__main__":
    # Ensure necessary environment variables like PJRT_DEVICE=TPU are set
        # Configuration (these would typically come from args or a config file)
    NUM_TPU_CORES_PER_HOST = "auto" # Standard for most TPU VMs
    # NUM_HOSTS should be dynamically determined or configured based on your TPU Pod slice
    # For example, if you have a v4-32 (32 chips), and each host has 4 chips, NUM_HOSTS = 8.
    # This value is critical for Fabric to initialize the distributed environment correctly.
    # It can often be inferred by XLA if the launch mechanism sets up the environment.
    # If fabric.launch() is used, num_nodes must be passed if > 1.
    # If an external launcher like xla_dist is used, Fabric might pick it up.
    # For this example, let's assume it's known or passed.
    NUM_HOSTS = 64 # Example: training on 2 hosts

    # Define auto_wrap_policy (example: wrap TransformerEncoderLayer)
    # This is a critical parameter for FSDP.
    auto_wrap_policy_config = {TransformerEncoderLayer} # NOTE: a set of Modules
    # activation_checkpointing_policy_config = {    } # Optional

    # Initialize Fabric
    # For multi-host, state_dict_type MUST be 'sharded'
    # XLAFSDPStrategy specific parameters
    strategy_params = {
        "auto_wrap_policy": auto_wrap_policy_config,
        # "activation_checkpointing_policy": activation_checkpointing_policy_config,
        "state_dict_type": 'sharded', # Crucial for multi-host TPU
        "sequential_save": False, # Set to True to reduce host RAM during checkpointing
    }

    fabric = Fabric(
        accelerator="tpu",
        devices=NUM_TPU_CORES_PER_HOST, # Number of TPU cores per host
        num_nodes=NUM_HOSTS,            # Number of hosts/nodes
        strategy=XLAFSDPStrategy(**strategy_params),
        precision="32-true"          # NOTE: ValueError: `precision='bf16-mixed')` is not supported in XLA. `precision` must be one of: ('32-true', '16-true', 'bf16-true').
    )
    fabric.launch(main_training_function) # Essential for initializing the distributed environment