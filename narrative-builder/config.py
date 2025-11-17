"""
Configuration Management - Load settings from environment file
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager with environment variable support"""
    
    # Default configuration values
    DEFAULTS = {
        # Data Loading
        'DEFAULT_DATA_PATH': 'news_dataset.json',
        'MIN_SOURCE_RATING': 0.0,  # Changed from 8.0 to 0.0 to be more inclusive by default
        'STREAMING_PARSER': True,
        
        # Embedding Model
        'EMBEDDING_MODEL': 'all-mpnet-base-v2',
        'EMBEDDING_BATCH_SIZE': 32,
        'EMBEDDING_MAX_LENGTH': 512,
        'NORMALIZE_EMBEDDINGS': True,
        
        # Vector Index
        'FAISS_INDEX_TYPE': 'auto',  # auto, flat, ivf
        'FAISS_IVF_NLIST': None,  # Auto-calculate if None
        'FAISS_IVF_NPROBE': 10,
        'FAISS_USE_GPU': False,
        
        # Search
        'DEFAULT_TOP_K': 50,
        'SIMILARITY_THRESHOLD': 0.0,
        
        # Clustering
        'CLUSTERING_METHOD': 'keyword',  # keyword, kmeans, dbscan
        'MIN_CLUSTER_SIZE': 2,
        'MAX_CLUSTERS': 10,
        
        # Narrative Generation
        'SUMMARY_MIN_SENTENCES': 5,
        'SUMMARY_MAX_SENTENCES': 10,
        'TIMELINE_SORT': 'date',  # date, relevance
        'INCLUDE_GRAPH': True,
        
        # Performance
        'CACHE_DIR': './cache',
        'ENABLE_CACHE': True,
        'LOG_LEVEL': 'INFO',
        'NUM_WORKERS': 4,
        
        # Output
        'OUTPUT_FORMAT': 'json',  # json, yaml
        'OUTPUT_INDENT': 2,
        'INCLUDE_METADATA': True,
    }
    
    def __init__(self, env_file: Optional[str] = '.env'):
        """
        Initialize configuration
        
        Args:
            env_file: Path to environment file (default: .env)
        """
        self.config: Dict[str, Any] = self.DEFAULTS.copy()
        self.env_file = env_file
        self._load_env_file()
        self._load_environment_variables()
        self._validate_config()
        
    def _load_env_file(self):
        """Load configuration from .env file"""
        if not self.env_file:
            return
            
        env_path = Path(self.env_file)
        if not env_path.exists():
            logger.debug(f"Environment file not found: {self.env_file}")
            return
        
        logger.info(f"Loading configuration from {self.env_file}")
        
        try:
            with open(env_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue
                    
                    # Parse KEY=VALUE
                    if '=' not in line:
                        logger.warning(f"Invalid line {line_num} in {self.env_file}: {line}")
                        continue
                    
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Remove quotes if present
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    
                    # Convert value to appropriate type
                    value = self._parse_value(value)
                    
                    if key in self.config:
                        self.config[key] = value
                        logger.debug(f"Set {key} = {value}")
                    else:
                        logger.warning(f"Unknown configuration key: {key}")
        
        except Exception as e:
            logger.error(f"Error loading {self.env_file}: {e}")
    
    def _load_environment_variables(self):
        """Load configuration from system environment variables"""
        # Environment variables take precedence over .env file
        for key in self.config.keys():
            env_key = f"NARRATIVE_{key}"
            if env_key in os.environ:
                value = self._parse_value(os.environ[env_key])
                self.config[key] = value
                logger.debug(f"Set {key} from environment: {value}")
    
    def _parse_value(self, value: str) -> Any:
        """
        Parse string value to appropriate type
        
        Args:
            value: String value to parse
            
        Returns:
            Parsed value
        """
        # Boolean
        if value.lower() in ('true', 'yes', '1', 'on'):
            return True
        if value.lower() in ('false', 'no', '0', 'off'):
            return False
        
        # None/Null
        if value.lower() in ('none', 'null', ''):
            return None
        
        # Integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Float
        try:
            return float(value)
        except ValueError:
            pass
        
        # String
        return value
    
    def _validate_config(self):
        """Validate configuration values"""
        # Validate numeric ranges
        if self.config['MIN_SOURCE_RATING'] < 0 or self.config['MIN_SOURCE_RATING'] > 10:
            logger.warning("MIN_SOURCE_RATING should be between 0 and 10")
        
        if self.config['EMBEDDING_BATCH_SIZE'] < 1:
            logger.warning("EMBEDDING_BATCH_SIZE must be positive")
            self.config['EMBEDDING_BATCH_SIZE'] = 32
        
        if self.config['DEFAULT_TOP_K'] < 1:
            logger.warning("DEFAULT_TOP_K must be positive")
            self.config['DEFAULT_TOP_K'] = 50
        
        # Validate paths
        cache_dir = Path(self.config['CACHE_DIR'])
        if self.config['ENABLE_CACHE']:
            cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate choices
        valid_index_types = ['auto', 'flat', 'ivf']
        if self.config['FAISS_INDEX_TYPE'] not in valid_index_types:
            logger.warning(f"Invalid FAISS_INDEX_TYPE, using 'auto'")
            self.config['FAISS_INDEX_TYPE'] = 'auto'
        
        valid_clustering = ['keyword', 'kmeans', 'dbscan']
        if self.config['CLUSTERING_METHOD'] not in valid_clustering:
            logger.warning(f"Invalid CLUSTERING_METHOD, using 'keyword'")
            self.config['CLUSTERING_METHOD'] = 'keyword'
        
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.config['LOG_LEVEL'].upper() not in valid_log_levels:
            logger.warning(f"Invalid LOG_LEVEL, using 'INFO'")
            self.config['LOG_LEVEL'] = 'INFO'
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        """
        Set configuration value
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        if key in self.config:
            self.config[key] = value
            logger.debug(f"Updated {key} = {value}")
        else:
            logger.warning(f"Unknown configuration key: {key}")
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values"""
        return self.config.copy()
    
    def print_config(self):
        """Print current configuration"""
        print("\n" + "="*60)
        print("Current Configuration")
        print("="*60)
        
        categories = {
            'Data Loading': ['DEFAULT_DATA_PATH', 'MIN_SOURCE_RATING', 'STREAMING_PARSER'],
            'Embedding Model': ['EMBEDDING_MODEL', 'EMBEDDING_BATCH_SIZE', 'EMBEDDING_MAX_LENGTH', 'NORMALIZE_EMBEDDINGS'],
            'Vector Index': ['FAISS_INDEX_TYPE', 'FAISS_IVF_NLIST', 'FAISS_IVF_NPROBE', 'FAISS_USE_GPU'],
            'Search': ['DEFAULT_TOP_K', 'SIMILARITY_THRESHOLD'],
            'Clustering': ['CLUSTERING_METHOD', 'MIN_CLUSTER_SIZE', 'MAX_CLUSTERS'],
            'Narrative': ['SUMMARY_MIN_SENTENCES', 'SUMMARY_MAX_SENTENCES', 'TIMELINE_SORT', 'INCLUDE_GRAPH'],
            'Performance': ['CACHE_DIR', 'ENABLE_CACHE', 'LOG_LEVEL', 'NUM_WORKERS'],
            'Output': ['OUTPUT_FORMAT', 'OUTPUT_INDENT', 'INCLUDE_METADATA']
        }
        
        for category, keys in categories.items():
            print(f"\n{category}:")
            for key in keys:
                value = self.config[key]
                print(f"  {key:30s} = {value}")
        
        print("\n" + "="*60 + "\n")
    
    def save_to_file(self, filepath: str = '.env.example'):
        """
        Save current configuration to file
        
        Args:
            filepath: Path to save configuration
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("# Narrative Builder Configuration\n")
                f.write("# Generated configuration file\n\n")
                
                categories = {
                    'Data Loading': ['DEFAULT_DATA_PATH', 'MIN_SOURCE_RATING', 'STREAMING_PARSER'],
                    'Embedding Model': ['EMBEDDING_MODEL', 'EMBEDDING_BATCH_SIZE', 'EMBEDDING_MAX_LENGTH', 'NORMALIZE_EMBEDDINGS'],
                    'Vector Index': ['FAISS_INDEX_TYPE', 'FAISS_IVF_NLIST', 'FAISS_IVF_NPROBE', 'FAISS_USE_GPU'],
                    'Search': ['DEFAULT_TOP_K', 'SIMILARITY_THRESHOLD'],
                    'Clustering': ['CLUSTERING_METHOD', 'MIN_CLUSTER_SIZE', 'MAX_CLUSTERS'],
                    'Narrative': ['SUMMARY_MIN_SENTENCES', 'SUMMARY_MAX_SENTENCES', 'TIMELINE_SORT', 'INCLUDE_GRAPH'],
                    'Performance': ['CACHE_DIR', 'ENABLE_CACHE', 'LOG_LEVEL', 'NUM_WORKERS'],
                    'Output': ['OUTPUT_FORMAT', 'OUTPUT_INDENT', 'INCLUDE_METADATA']
                }
                
                for category, keys in categories.items():
                    f.write(f"# {category}\n")
                    for key in keys:
                        value = self.config[key]
                        if isinstance(value, str):
                            f.write(f'{key}="{value}"\n')
                        else:
                            f.write(f'{key}={value}\n')
                    f.write('\n')
            
            logger.info(f"Configuration saved to {filepath}")
        
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")


# Global configuration instance
_config: Optional[Config] = None


def get_config(env_file: Optional[str] = '.env') -> Config:
    """
    Get global configuration instance
    
    Args:
        env_file: Path to environment file
        
    Returns:
        Configuration instance
    """
    global _config
    if _config is None:
        _config = Config(env_file)
    return _config


def reload_config(env_file: Optional[str] = '.env'):
    """
    Reload configuration from file
    
    Args:
        env_file: Path to environment file
    """
    global _config
    _config = Config(env_file)
    return _config
