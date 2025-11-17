"""
Main semantic search engine for Twitter API documentation.
Provides command-line interface for querying the documentation.
Features: Multiprocessing for embedding generation, caching for query results.
"""

import os
import json
import argparse
import hashlib
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
from functools import lru_cache
from multiprocessing import Pool, cpu_count
import numpy as np

from config import Config
from data_loader import load_twitter_api_docs
from chunker import create_documentation_chunks
from embedder import DocumentEmbedder
from vector_store import FAISSVectorStore


class QueryCache:
    """LRU cache for query results with disk persistence."""
    
    def __init__(self, cache_dir: str, max_memory_size: int = 100):
        """
        Initialize query cache.
        
        Args:
            cache_dir: Directory to store cache files
            max_memory_size: Maximum number of queries to keep in memory
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / 'query_cache.pkl'
        self.max_memory_size = max_memory_size
        self.memory_cache = {}
        self.access_order = []
        
        # Load existing cache from disk
        self._load_from_disk()
    
    def _get_cache_key(self, query: str, top_k: int, model_name: str) -> str:
        """Generate cache key from query parameters."""
        cache_string = f"{query}|{top_k}|{model_name}"
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def get(self, query: str, top_k: int, model_name: str) -> Optional[List[Dict[str, Any]]]:
        """Retrieve cached results if available."""
        cache_key = self._get_cache_key(query, top_k, model_name)
        
        if cache_key in self.memory_cache:
            # Update access order (LRU)
            self.access_order.remove(cache_key)
            self.access_order.append(cache_key)
            return self.memory_cache[cache_key]
        
        return None
    
    def set(self, query: str, top_k: int, model_name: str, results: List[Dict[str, Any]]):
        """Store results in cache."""
        cache_key = self._get_cache_key(query, top_k, model_name)
        
        # Add to memory cache
        self.memory_cache[cache_key] = results
        self.access_order.append(cache_key)
        
        # Evict oldest if over size limit
        while len(self.memory_cache) > self.max_memory_size:
            oldest_key = self.access_order.pop(0)
            del self.memory_cache[oldest_key]
        
        # Persist to disk
        self._save_to_disk()
    
    def _load_from_disk(self):
        """Load cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    data = pickle.load(f)
                    self.memory_cache = data.get('cache', {})
                    self.access_order = data.get('order', [])
                    
                    # Limit to max size
                    while len(self.memory_cache) > self.max_memory_size:
                        oldest_key = self.access_order.pop(0)
                        del self.memory_cache[oldest_key]
            except Exception as e:
                print(f"Warning: Could not load cache from disk: {e}")
    
    def _save_to_disk(self):
        """Save cache to disk."""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump({
                    'cache': self.memory_cache,
                    'order': self.access_order
                }, f)
        except Exception as e:
            print(f"Warning: Could not save cache to disk: {e}")
    
    def clear(self):
        """Clear all cached data."""
        self.memory_cache.clear()
        self.access_order.clear()
        if self.cache_file.exists():
            self.cache_file.unlink()
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            'total_entries': len(self.memory_cache),
            'max_size': self.max_memory_size
        }


def _embed_batch_worker(args):
    """Worker function for multiprocessing embedding generation."""
    batch_chunks, model_name, device = args
    embedder = DocumentEmbedder(model_name, device=device)
    return embedder.embed_chunks(batch_chunks)


class TwitterAPISemanticSearch:
    """Semantic search engine for Twitter API documentation."""

    def __init__(self,
                 collection_path: str = None,
                 model_name: str = None,
                 use_gpu: bool = None,
                 index_dir: str = None,
                 enable_cache: bool = True,
                 num_workers: int = None):
        """
        Initialize the semantic search engine.

        Args:
            collection_path: Path to Postman collection JSON (defaults to config)
            model_name: Sentence transformer model name (defaults to config)
            use_gpu: Whether to use GPU acceleration (defaults to config)
            index_dir: Directory to save/load index files (defaults to config)
            enable_cache: Enable query result caching
            num_workers: Number of worker processes for embedding (None = auto)
        """
        # Use config values as defaults
        self.collection_path = collection_path or Config.POSTMAN_COLLECTION_PATH
        self.model_name = model_name or Config.EMBEDDING_MODEL
        self.use_gpu = use_gpu if use_gpu is not None else Config.USE_GPU
        self.index_dir = Path(index_dir or Config.INDEX_DIR)
        self.index_dir.mkdir(exist_ok=True)
        
        # Multiprocessing settings
        self.num_workers = num_workers or max(1, cpu_count() - 1)
        
        # Caching
        self.enable_cache = enable_cache
        if enable_cache:
            cache_dir = self.index_dir / 'cache'
            self.query_cache = QueryCache(str(cache_dir))
        else:
            self.query_cache = None

        device = Config.get_device() if use_gpu is None else ('cuda' if use_gpu else 'cpu')
        self.embedder = DocumentEmbedder(self.model_name, device=device)
        self.vector_store = None
        self.chunks = None

    def _embed_chunks_parallel(self, chunks: List[Dict[str, Any]]) -> np.ndarray:
        """
        Generate embeddings using multiprocessing.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            Array of embeddings
        """
        # For small datasets or GPU, use single process
        if len(chunks) < 100 or self.use_gpu:
            return self.embedder.embed_chunks(chunks)
        
        print(f"Using {self.num_workers} workers for parallel embedding...")
        
        # Split chunks into batches for workers
        batch_size = len(chunks) // self.num_workers
        batches = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batches.append((batch, self.model_name, 'cpu'))  # Force CPU for workers
        
        # Process batches in parallel
        with Pool(processes=self.num_workers) as pool:
            results = pool.map(_embed_batch_worker, batches)
        
        # Concatenate results
        embeddings = np.vstack(results)
        return embeddings

    def build_index(self, force_rebuild: bool = False, use_parallel: bool = True):
        """
        Build the search index from documentation.

        Args:
            force_rebuild: Force rebuild even if index exists
            use_parallel: Use multiprocessing for embedding generation
        """
        index_path = self.index_dir / 'faiss.index'
        chunks_path = self.index_dir / 'chunks.pkl'

        # Check if index exists and load if not force rebuilding
        if not force_rebuild and index_path.exists() and chunks_path.exists():
            print("Loading existing index...")
            self.embedder.load_model()
            embedding_dim = self.embedder.get_embedding_dimension()
            self.vector_store = FAISSVectorStore(embedding_dim, self.use_gpu)
            self.vector_store.load(str(index_path), str(chunks_path))
            self.chunks = self.vector_store.chunks
            return

        print("Building index from scratch...")

        # Step 1: Load documentation
        print(f"Loading documentation from {self.collection_path}...")
        endpoints = load_twitter_api_docs(self.collection_path)
        print(f"Loaded {len(endpoints)} endpoints")

        # Step 2: Create chunks
        print("Creating documentation chunks...")
        self.chunks = create_documentation_chunks(endpoints)
        print(f"Created {len(self.chunks)} chunks")

        # Step 3: Generate embeddings
        print("Generating embeddings...")
        if use_parallel and not self.use_gpu:
            embeddings = self._embed_chunks_parallel(self.chunks)
        else:
            embeddings = self.embedder.embed_chunks(self.chunks)

        # Step 4: Build FAISS index
        print("Building FAISS index...")
        embedding_dim = embeddings.shape[1]
        self.vector_store = FAISSVectorStore(embedding_dim, self.use_gpu)
        self.vector_store.build_index(embeddings, self.chunks)

        # Step 5: Save index
        print("Saving index...")
        self.vector_store.save(str(index_path), str(chunks_path))

        print("Index built and saved successfully!")

    def search(self, query: str, top_k: int = 5, use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Search the documentation for a query.

        Args:
            query: Search query string
            top_k: Number of results to return
            use_cache: Use cached results if available

        Returns:
            List of search results with rankings and scores
        """
        if self.vector_store is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Check cache first
        if use_cache and self.query_cache:
            cached_results = self.query_cache.get(query, top_k, self.model_name)
            if cached_results is not None:
                print("(Using cached results)")
                return cached_results

        # Generate query embedding
        query_embedding = self.embedder.embed_query(query)

        # Search
        results = self.vector_store.search(query_embedding, top_k)
        
        # Store in cache
        if use_cache and self.query_cache:
            self.query_cache.set(query, top_k, self.model_name, results)

        return results

    def clear_cache(self):
        """Clear the query cache."""
        if self.query_cache:
            self.query_cache.clear()
            print("Cache cleared!")
    
    def get_cache_stats(self) -> Optional[Dict[str, int]]:
        """Get cache statistics."""
        if self.query_cache:
            return self.query_cache.get_stats()
        return None

    def format_results_json(self, results: List[Dict[str, Any]]) -> str:
        """
        Format search results as JSON.

        Args:
            results: List of search results

        Returns:
            JSON formatted string
        """
        return json.dumps(results, indent=2)

    def format_results_readable(self, results: List[Dict[str, Any]]) -> str:
        """
        Format search results in a human-readable format.

        Args:
            results: List of search results

        Returns:
            Formatted string
        """
        lines = []
        for result in results:
            lines.append(f"\n{'=' * 80}")
            lines.append(f"Rank: {result['rank']}")
            lines.append(f"Similarity Score: {result['similarity_score']:.4f}")
            lines.append(f"Endpoint: {result['metadata']['endpoint_name']}")
            lines.append(f"Category: {result['metadata']['category']}")
            lines.append(f"Method: {result['metadata']['method']}")
            lines.append(f"URL: {result['metadata']['url']}")
            lines.append(f"\n{result['text']}")

        lines.append(f"\n{'=' * 80}")
        return '\n'.join(lines)


def main():
    """Main entry point for command-line interface."""
    parser = argparse.ArgumentParser(
        description='Semantic search engine for Twitter API documentation',
        epilog='Configuration can be set via .env file or command-line arguments. '
               'Command-line arguments override .env settings.'
    )
    parser.add_argument(
        '--query',
        type=str,
        help='Search query string'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=None,
        help=f'Number of results to return (default: {Config.DEFAULT_TOP_K} from config)'
    )
    parser.add_argument(
        '--collection',
        type=str,
        default=None,
        help=f'Path to Postman collection JSON file (default: from config)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help=f'Sentence transformer model name (default: {Config.EMBEDDING_MODEL} from config)'
    )
    parser.add_argument(
        '--rebuild',
        action='store_true',
        help='Force rebuild the index'
    )
    parser.add_argument(
        '--output',
        type=str,
        choices=['json', 'readable'],
        default=None,
        help=f'Output format (default: {Config.DEFAULT_OUTPUT_FORMAT} from config)'
    )
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Disable GPU acceleration'
    )
    parser.add_argument(
        '--show-config',
        action='store_true',
        help='Display current configuration and exit'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Number of worker processes for embedding (default: auto)'
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable query result caching'
    )
    parser.add_argument(
        '--clear-cache',
        action='store_true',
        help='Clear the query cache and exit'
    )
    parser.add_argument(
        '--cache-stats',
        action='store_true',
        help='Display cache statistics and exit'
    )
    parser.add_argument(
        '--no-parallel',
        action='store_true',
        help='Disable parallel processing for embedding generation'
    )

    args = parser.parse_args()

    # Show config if requested
    if args.show_config:
        Config.display_config()
        return

    # Use config defaults for unspecified arguments
    top_k = args.top_k if args.top_k is not None else Config.DEFAULT_TOP_K
    output_format = args.output if args.output is not None else Config.DEFAULT_OUTPUT_FORMAT

    # Initialize search engine with config-aware defaults
    search_engine = TwitterAPISemanticSearch(
        collection_path=args.collection,
        model_name=args.model,
        use_gpu=None if not args.no_gpu else False,
        enable_cache=not args.no_cache,
        num_workers=args.workers
    )

    # Handle cache commands
    if args.clear_cache:
        search_engine.clear_cache()
        return
    
    if args.cache_stats:
        stats = search_engine.get_cache_stats()
        if stats:
            print("Cache Statistics:")
            print(f"  Total entries: {stats['total_entries']}")
            print(f"  Max size: {stats['max_size']}")
        else:
            print("Caching is disabled")
        return

    print(f"Using model: {search_engine.model_name}")
    print(f"Using device: {search_engine.embedder.device}")
    print(f"Workers: {search_engine.num_workers}")
    print(f"Caching: {'enabled' if search_engine.enable_cache else 'disabled'}")
    print()

    # Build or load index
    search_engine.build_index(
        force_rebuild=args.rebuild,
        use_parallel=not args.no_parallel
    )

    # If query provided, search
    if args.query:
        print(f"\nSearching for: '{args.query}'")
        print(f"Top {top_k} results:\n")

        results = search_engine.search(args.query, top_k=top_k)

        if output_format == 'json':
            print(search_engine.format_results_json(results))
        else:
            print(search_engine.format_results_readable(results))
    else:
        print("\nIndex ready. Use --query to search.")
        print("Example: python semantic_search.py --query 'How do I fetch tweets with expansions?'")
        print("\nTip: Use --show-config to see current configuration")
        print("Tip: Use --cache-stats to see cache statistics")
        print("Tip: Use --clear-cache to clear cached queries")


if __name__ == '__main__':
    main()