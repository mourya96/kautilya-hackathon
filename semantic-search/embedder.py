"""
Embedding generation module using sentence-transformers.
Converts documentation chunks into dense vector representations.
"""

import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer


class DocumentEmbedder:
    """Generates embeddings for documentation chunks using sentence transformers."""

    def __init__(self, model_name: str = 'all-mpnet-base-v2', device: str = None):
        """
        Initialize the embedder with a sentence transformer model.

        Args:
            model_name: Name of the sentence-transformers model to use
                       Options: 'all-MiniLM-L6-v2' (fast), 'all-mpnet-base-v2' (quality)
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        self.model_name = model_name
        self.device = device
        self.model = None

    def load_model(self):
        """Load the sentence transformer model."""
        print(f"Loading embedding model: {self.model_name}...")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        print(f"Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")

    def embed_chunks(self, chunks: List[Dict[str, Any]],
                    batch_size: int = 32,
                    show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for a list of documentation chunks.

        Args:
            chunks: List of chunk dictionaries with 'text' field
            batch_size: Batch size for embedding generation
            show_progress: Whether to show progress bar

        Returns:
            NumPy array of embeddings with shape (n_chunks, embedding_dim)
        """
        if self.model is None:
            self.load_model()

        # Extract text from chunks
        texts = [chunk['text'] for chunk in chunks]

        # Generate embeddings
        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalize for cosine similarity
        )

        print(f"Embeddings generated. Shape: {embeddings.shape}")
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a search query.

        Args:
            query: Search query string

        Returns:
            NumPy array of query embedding with shape (embedding_dim,)
        """
        if self.model is None:
            self.load_model()

        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embeddings.

        Returns:
            Embedding dimension size
        """
        if self.model is None:
            self.load_model()
        return self.model.get_sentence_embedding_dimension()


def generate_embeddings(chunks: List[Dict[str, Any]],
                       model_name: str = 'all-mpnet-base-v2',
                       device: str = None) -> np.ndarray:
    """
    Convenience function to generate embeddings for documentation chunks.

    Args:
        chunks: List of chunk dictionaries
        model_name: Sentence transformer model name
        device: Device to use for embeddings

    Returns:
        NumPy array of embeddings
    """
    embedder = DocumentEmbedder(model_name, device)
    return embedder.embed_chunks(chunks)
