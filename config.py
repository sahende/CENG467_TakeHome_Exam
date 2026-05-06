"""
Global configuration file for the CENG467 NLP project.
Contains all shared settings and paths.
"""
import os
import torch

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Random seed for reproducibility
SEED = 42

# Training hyperparameters (shared across questions)
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
NUM_EPOCHS = 5
MAX_SEQ_LENGTH = 512