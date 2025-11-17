"""
Utilities - Helper functions and configurations
"""

import json
import logging
import sys
from typing import Any, Dict
from datetime import datetime


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        level: Logging level
        
    Returns:
        Configured logger
    """
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    
    # Reduce noise from external libraries
    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    return root_logger


def format_json_output(data: Dict[str, Any], indent: int = 2) -> str:
    """
    Format data as pretty JSON string
    
    Args:
        data: Dictionary to format
        indent: Indentation level
        
    Returns:
        Formatted JSON string
    """
    return json.dumps(data, indent=indent, ensure_ascii=False)


def calculate_statistics(articles: list) -> Dict[str, Any]:
    """
    Calculate dataset statistics
    
    Args:
        articles: List of articles
        
    Returns:
        Statistics dictionary
    """
    if not articles:
        return {}
    
    stats = {
        'total_articles': len(articles),
        'sources': len(set(a.get('source', 'Unknown') for a in articles)),
        'date_range': get_date_range(articles),
        'avg_content_length': sum(len(a.get('content', '')) for a in articles) / len(articles)
    }
    
    return stats


def get_date_range(articles: list) -> tuple:
    """
    Get date range from articles
    
    Args:
        articles: List of articles
        
    Returns:
        Tuple of (earliest_date, latest_date)
    """
    dates = [a['date'] for a in articles if a.get('date')]
    if not dates:
        return None, None
    return min(dates), max(dates)


def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to maximum length
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."


class Timer:
    """Context manager for timing code blocks"""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
        
    def __enter__(self):
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, *args):
        self.end_time = datetime.now()
        elapsed = (self.end_time - self.start_time).total_seconds()
        print(f"{self.name} completed in {elapsed:.2f}s", file=sys.stderr)
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0


def validate_article(article: Dict[str, Any]) -> bool:
    """
    Validate article has required fields
    
    Args:
        article: Article dictionary
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = ['headline', 'content']
    return all(field in article and article[field] for field in required_fields)


def normalize_score(score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """
    Normalize score to range [min_val, max_val]
    
    Args:
        score: Score to normalize
        min_val: Minimum value
        max_val: Maximum value
        
    Returns:
        Normalized score
    """
    return max(min_val, min(max_val, score))
