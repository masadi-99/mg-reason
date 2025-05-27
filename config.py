"""Configuration settings for the medical reasoning evaluation system."""
import os
from typing import Dict, List

# API Configuration
def load_openai_api_key() -> str:
    """Load OpenAI API key from file."""
    try:
        with open('openai_api_key.txt', 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError("openai_api_key.txt not found. Please ensure your API key file exists.")

OPENAI_API_KEY = load_openai_api_key()

# Model configurations
OPENAI_MODELS = {
    "gpt-4o-mini": {
        "max_tokens": 4096,
        "temperature": 0.0,
        "top_p": 1.0,
    },
    "gpt-4": {
        "max_tokens": 8192,
        "temperature": 0.0,
        "top_p": 1.0,
    },
    "gpt-4-turbo": {
        "max_tokens": 4096,
        "temperature": 0.0,
        "top_p": 1.0,
    },
    "gpt-3.5-turbo": {
        "max_tokens": 4096,
        "temperature": 0.0,
        "top_p": 1.0,
    }
}

# Dataset paths
DATASET_PATHS = {
    "train": "S-MedQA_train.json",
    "validation": "S-MedQA_validation.json", 
    "test": "S-MedQA_test.json",
    "test_filtered_6": "S-MedQA_test_filtered_6specialties.json"
}

# BMJ Best Practice settings
BMJ_BASE_URL = "https://bestpractice.bmj.com"
BMJ_CARDIOLOGY_URL = "https://bestpractice.bmj.com/specialties/2/Cardiology"
PDF_DOWNLOAD_DIR = "cardiology_pdfs"

# Evaluation settings
EVALUATION_SETTINGS = {
    "sample_size": None,  # None for full dataset
    "reasoning_types": ["direct", "chain_of_thought", "self_consistency"],
    "num_trials": 1,
    "random_seed": 42
}

# Batch processing settings
BATCH_SETTINGS = {
    "enabled": True,  # Enable batch processing by default
    "min_batch_size": 10,  # Minimum samples to use batch processing
    "max_batch_size": 1000,  # Maximum samples per batch
    "poll_interval": 30,  # Seconds between status checks
    "max_wait_time": 86400,  # Maximum wait time (24 hours)
    "batch_dir": "batch_files",  # Directory for batch files
    "auto_cleanup": True,  # Clean up batch files after completion
    "demo_mode": False,  # Demo mode for testing (simulates batch processing)
    "fast_poll": False,  # Use faster polling for development (10s intervals)
} 