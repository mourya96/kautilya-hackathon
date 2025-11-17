"""
News Embedder - High-performance semantic embedding and vector search
"""

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict, Any, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor
import pickle
from pathlib import Path
from config import get_config

logger = logging.getLogger(__name__)


class NewsEmbedder:
    """Handles embedding generation and vector similarity search"""
    
    def __init__(self, model_name: str = None, cache_dir: str = None):
        """
        Initialize embedder with specified model
        
        Args:
            model_name: Name of sentence-transformers model (uses config if None)
            cache_dir: Directory to cache embeddings and index (uses config if None)
        """
        config = get_config()
        
        if model_name is None:
            model_name = config.get('EMBEDDING_MODEL')
        if cache_dir is None:
            cache_dir = config.get('CACHE_DIR')
        
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.articles = []
        self.embeddings = None
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.config = config
        
    def build_index(self, articles: List[Dict[str, Any]], batch_size: int = None):
        """
        Build FAISS index from articles
        
        Args:
            articles: List of article dictionaries
            batch_size: Batch size for embedding generation (uses config if None)
        """
        if batch_size is None:
            batch_size = self.config.get('EMBEDDING_BATCH_SIZE')
        
        self.articles = articles
        
        # Check for cached embeddings
        cache_file = self.cache_dir / 'embeddings.pkl'
        if self.config.get('ENABLE_CACHE') and cache_file.exists():
            logger.info("Loading cached embeddings...")
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    if len(cached_data['articles']) == len(articles):
                        self.embeddings = cached_data['embeddings']
                        logger.info("Loaded embeddings from cache")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        
        # Generate embeddings if not cached
        if self.embeddings is None:
            logger.info(f"Generating embeddings for {len(articles)} articles...")
            max_length = self.config.get('EMBEDDING_MAX_LENGTH')
            texts = [article['full_text'][:max_length] for article in articles]
            
            # Generate embeddings in batches for efficiency
            normalize = self.config.get('NORMALIZE_EMBEDDINGS')
            self.embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=normalize
            )
            
            # Cache embeddings
            if self.config.get('ENABLE_CACHE'):
                try:
                    with open(cache_file, 'wb') as f:
                        pickle.dump({
                            'articles': [a['id'] for a in articles],
                            'embeddings': self.embeddings
                        }, f)
                    logger.info("Cached embeddings for future use")
                except Exception as e:
                    logger.warning(f"Failed to cache embeddings: {e}")
        
        # Build FAISS index
        logger.info("Building FAISS index...")
        self._build_faiss_index()
        
    def _build_faiss_index(self):
        """Build optimized FAISS index"""
        n_articles = len(self.embeddings)
        index_type = self.config.get('FAISS_INDEX_TYPE')
        use_gpu = self.config.get('FAISS_USE_GPU')
        
        # Determine index type
        if index_type == 'auto':
            # Auto-select based on dataset size
            if n_articles < 100000:
                index_type = 'flat'
            else:
                index_type = 'ivf'
        
        # Build appropriate index
        if index_type == 'flat':
            # Use IndexFlatIP for cosine similarity (inner product with normalized vectors)
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        else:  # ivf
            # Use IVF for larger datasets
            nlist = self.config.get('FAISS_IVF_NLIST')
            if nlist is None:
                nlist = int(np.sqrt(n_articles))  # Auto-calculate
            
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
            self.index.train(self.embeddings)
            self.index.nprobe = self.config.get('FAISS_IVF_NPROBE')
        
        # Move to GPU if requested and available
        if use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                logger.info("Using GPU acceleration")
            except Exception as e:
                logger.warning(f"GPU not available, using CPU: {e}")
        
        self.index.add(self.embeddings)
        logger.info(f"FAISS index built with {self.index.ntotal} vectors (type: {index_type})")
    
    def search(self, query: str, top_k: int = 50) -> List[Dict[str, Any]]:
        """
        Search for articles relevant to query
        
        Args:
            query: Search query text
            top_k: Number of results to return
            
        Returns:
            List of relevant articles with similarity scores
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Generate query embedding
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Search index
        similarities, indices = self.index.search(query_embedding, top_k)
        
        # Prepare results
        results = []
        for idx, similarity in zip(indices[0], similarities[0]):
            if idx < len(self.articles):
                article = self.articles[idx].copy()
                article['similarity_score'] = float(similarity)
                results.append(article)
        
        return results
    
    def compute_similarity_matrix(self, articles: List[Dict[str, Any]]) -> np.ndarray:
        """
        Compute pairwise similarity matrix for a set of articles
        
        Args:
            articles: List of articles to compute similarities for
            
        Returns:
            NxN similarity matrix
        """
        # Get embeddings for these articles
        article_ids = [a['id'] for a in articles]
        indices = [i for i, a in enumerate(self.articles) if a['id'] in article_ids]
        
        if not indices:
            return np.array([[]])
        
        # Extract embeddings
        article_embeddings = self.embeddings[indices]
        
        # Compute cosine similarity matrix
        similarity_matrix = np.dot(article_embeddings, article_embeddings.T)
        
        return similarity_matrix
    
    def find_similar_articles(self, article: Dict[str, Any], top_k: int = 10) -> List[Tuple[Dict[str, Any], float]]:
        """
        Find articles similar to a given article
        
        Args:
            article: Reference article
            top_k: Number of similar articles to return
            
        Returns:
            List of (article, similarity) tuples
        """
        # Find article index
        try:
            idx = next(i for i, a in enumerate(self.articles) if a['id'] == article['id'])
        except StopIteration:
            return []
        
        # Get embedding
        article_embedding = self.embeddings[idx:idx+1]
        
        # Search
        similarities, indices = self.index.search(article_embedding, top_k + 1)
        
        # Skip first result (the article itself)
        results = []
        for idx, similarity in zip(indices[0][1:], similarities[0][1:]):
            if idx < len(self.articles):
                results.append((self.articles[idx], float(similarity)))
        
        return results
    
    def batch_encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode a batch of texts
        
        Args:
            texts: List of text strings
            
        Returns:
            Array of embeddings
        """
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
