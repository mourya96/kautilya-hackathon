"""
Configuration module for semantic search engine.
Loads configuration from environment variables with fallback defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration settings loaded from environment variables."""

    # Embedding Model Configuration
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'all-mpnet-base-v2')

    # Device Configuration
    DEVICE = os.getenv('DEVICE', 'cuda')

    # FAISS Configuration
    USE_GPU = os.getenv('USE_GPU', 'true').lower() in ('true', '1', 'yes')

    # Search Configuration
    DEFAULT_TOP_K = int(os.getenv('DEFAULT_TOP_K', '5'))
    DEFAULT_OUTPUT_FORMAT = os.getenv('DEFAULT_OUTPUT_FORMAT', 'json')

    # Paths
    POSTMAN_COLLECTION_PATH = os.getenv(
        'POSTMAN_COLLECTION_PATH',
        'postman-twitter-api/Twitter API v2.postman_collection.json'
    )
    INDEX_DIR = os.getenv('INDEX_DIR', 'index')

    @classmethod
    def get_device(cls):
        """Get the device to use for computations."""
        if cls.DEVICE == 'auto':
            import torch
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return cls.DEVICE

    @classmethod
    def display_config(cls):
        """Display current configuration."""
        print("Current Configuration:")
        print(f"  Embedding Model: {cls.EMBEDDING_MODEL}")
        print(f"  Device: {cls.get_device()}")
        print(f"  Use GPU for FAISS: {cls.USE_GPU}")
        print(f"  Default Top-K: {cls.DEFAULT_TOP_K}")
        print(f"  Default Output Format: {cls.DEFAULT_OUTPUT_FORMAT}")
        print(f"  Collection Path: {cls.POSTMAN_COLLECTION_PATH}")
        print(f"  Index Directory: {cls.INDEX_DIR}")