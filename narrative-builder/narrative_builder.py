#!/usr/bin/env python3
"""
Narrative Builder - Main Entry Point
Processes large news datasets and generates dynamic narratives for any topic
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any
import time

from data_loader import DataLoader
from embedder import NewsEmbedder
from narrative_engine import NarrativeEngine
from utils import setup_logging, format_json_output
from config import get_config

# Load configuration
config = get_config()
logger = setup_logging(level=getattr(__import__('logging'), config.get('LOG_LEVEL', 'INFO')))


class NarrativeBuilder:
    """Main orchestrator for narrative generation"""
    
    def __init__(self, data_path: str = None, min_rating: float = None):
        """
        Initialize the narrative builder
        
        Args:
            data_path: Path to the JSON news dataset (uses config if None)
            min_rating: Minimum source rating to filter (uses config if None)
        """
        self.config = get_config()
        self.data_path = data_path or self.config.get('DEFAULT_DATA_PATH')
        self.min_rating = min_rating if min_rating is not None else self.config.get('MIN_SOURCE_RATING')
        self.data_loader = None
        self.embedder = None
        self.engine = None
        
    def load_and_filter(self) -> int:
        """Load dataset and filter by rating"""
        logger.info(f"Loading dataset from {self.data_path}...")
        start_time = time.time()
        
        self.data_loader = DataLoader(self.data_path)
        filtered_count = self.data_loader.load_and_filter(self.min_rating)
        
        elapsed = time.time() - start_time
        logger.info(f"Loaded {filtered_count} articles (rating > {self.min_rating}) in {elapsed:.2f}s")
        
        return filtered_count
    
    def initialize_embedder(self):
        """Initialize the embedding model and create vector index"""
        logger.info("Initializing embedding model...")
        start_time = time.time()
        
        model_name = self.config.get('EMBEDDING_MODEL')
        cache_dir = self.config.get('CACHE_DIR')
        
        self.embedder = NewsEmbedder(model_name=model_name, cache_dir=cache_dir)
        articles = self.data_loader.get_articles()
        
        batch_size = self.config.get('EMBEDDING_BATCH_SIZE')
        self.embedder.build_index(articles, batch_size=batch_size)
        
        elapsed = time.time() - start_time
        logger.info(f"Built vector index for {len(articles)} articles in {elapsed:.2f}s")
    
    def generate_narrative(self, topic: str, top_k: int = None) -> Dict[str, Any]:
        """
        Generate narrative for a given topic
        
        Args:
            topic: User-provided topic to generate narrative for
            top_k: Number of most relevant articles to retrieve (uses config if None)
            
        Returns:
            Complete narrative structure as dictionary
        """
        if top_k is None:
            top_k = self.config.get('DEFAULT_TOP_K')
            
        logger.info(f"Generating narrative for topic: '{topic}'")
        start_time = time.time()
        
        # Retrieve relevant articles using semantic search
        relevant_articles = self.embedder.search(topic, top_k=top_k)
        logger.info(f"Retrieved {len(relevant_articles)} relevant articles")
        
        # Initialize narrative engine
        self.engine = NarrativeEngine(relevant_articles, topic)
        
        # Generate all components
        narrative = {
            "topic": topic,
            "narrative_summary": self.engine.generate_summary(),
            "timeline": self.engine.build_timeline(),
            "clusters": self.engine.create_clusters(),
            "graph": self.engine.build_narrative_graph()
        }
        
        elapsed = time.time() - start_time
        logger.info(f"Narrative generation completed in {elapsed:.2f}s")
        
        return narrative


def main():
    """Main entry point"""
    # Load configuration first
    config = get_config()
    
    parser = argparse.ArgumentParser(
        description='Generate narratives from news data for any topic',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python narrative_builder.py --topic "Jubilee Hills elections"
  python narrative_builder.py --topic "Israel-Iran conflict" --data news_data.json
  python narrative_builder.py --topic "AI regulation" --top-k 100 --output narrative.json
  python narrative_builder.py --config  # Show current configuration
        """
    )
    
    parser.add_argument(
        '--topic',
        type=str,
        required=False,
        help='Topic to generate narrative for (e.g., "AI regulation")'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        default=None,
        help=f'Path to news dataset JSON file (default: from config or {config.get("DEFAULT_DATA_PATH")})'
    )
    
    parser.add_argument(
        '--rating',
        type=float,
        default=None,
        help=f'Minimum source rating threshold (default: from config or {config.get("MIN_SOURCE_RATING")})'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=None,
        help=f'Number of relevant articles to retrieve (default: from config or {config.get("DEFAULT_TOP_K")})'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (default: print to stdout)'
    )
    
    parser.add_argument(
        '--env',
        type=str,
        default='.env',
        help='Path to environment configuration file (default: .env)'
    )
    
    parser.add_argument(
        '--config',
        action='store_true',
        help='Show current configuration and exit'
    )
    
    parser.add_argument(
        '--save-config',
        type=str,
        default=None,
        metavar='FILE',
        help='Save current configuration to file and exit'
    )
    
    args = parser.parse_args()
    
    # Reload config if custom env file specified
    if args.env != '.env':
        from config import reload_config
        config = reload_config(args.env)
    
    # Handle config display/save commands
    if args.config:
        config.print_config()
        sys.exit(0)
    
    if args.save_config:
        config.save_to_file(args.save_config)
        print(f"Configuration saved to {args.save_config}")
        sys.exit(0)
    
    # Topic is required for narrative generation
    if not args.topic:
        parser.error("--topic is required (unless using --config or --save-config)")
    
    # Use command-line args or fall back to config
    data_path = args.data or config.get('DEFAULT_DATA_PATH')
    min_rating = args.rating if args.rating is not None else config.get('MIN_SOURCE_RATING')
    top_k = args.top_k if args.top_k is not None else config.get('DEFAULT_TOP_K')
    
    # Validate data file exists
    if not Path(data_path).exists():
        logger.error(f"Dataset file not found: {data_path}")
        sys.exit(1)
    
    try:
        # Initialize builder
        builder = NarrativeBuilder(data_path, min_rating)
        
        # Load and filter data
        article_count = builder.load_and_filter()
        if article_count == 0:
            logger.error("No articles found after filtering. Check your dataset and rating threshold.")
            sys.exit(1)
        
        # Build vector index
        builder.initialize_embedder()
        
        # Generate narrative
        narrative = builder.generate_narrative(args.topic, top_k)
        
        # Output results
        output_json = format_json_output(narrative)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(output_json)
            logger.info(f"Narrative saved to {args.output}")
        else:
            print(output_json)
        
        logger.info("Narrative generation completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Error during narrative generation: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
