"""
Vector store module using FAISS for efficient similarity search.
Manages the vector index and performs semantic search queries.
"""

import os
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple
import faiss


class FAISSVectorStore:
    """FAISS-based vector store for semantic search."""

    def __init__(self, embedding_dim: int, use_gpu: bool = True):
        """
        Initialize the FAISS vector store.

        Args:
            embedding_dim: Dimension of the embeddings
            use_gpu: Whether to use GPU acceleration (if available)
        """
        self.embedding_dim = embedding_dim
        self.use_gpu = use_gpu
        self.index = None
        self.chunks = None
        self.gpu_index = None

    def build_index(self, embeddings: np.ndarray, chunks: List[Dict[str, Any]]):
        """
        Build the FAISS index from embeddings.

        Args:
            embeddings: NumPy array of embeddings (n_chunks, embedding_dim)
            chunks: List of chunk dictionaries corresponding to embeddings
        """
        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                f"Number of embeddings ({embeddings.shape[0]}) must match "
                f"number of chunks ({len(chunks)})"
            )

        print(f"Building FAISS index with {embeddings.shape[0]} vectors...")

        # Create flat L2 index (for cosine similarity with normalized embeddings)
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner Product for normalized vectors

        # Try to use GPU if available and requested
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.gpu_index = faiss.index_cpu_to_gpu(res, 0, self.index)
                self.gpu_index.add(embeddings.astype(np.float32))
                print("FAISS index built successfully on GPU")
            except Exception as e:
                print(f"GPU not available, falling back to CPU: {e}")
                self.index.add(embeddings.astype(np.float32))
                print("FAISS index built successfully on CPU")
        else:
            self.index.add(embeddings.astype(np.float32))
            print("FAISS index built successfully on CPU")

        # Store chunks for retrieval
        self.chunks = chunks

    def search(self, query_embedding: np.ndarray,
              top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents using query embedding.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return

        Returns:
            List of result dictionaries with text, metadata, and similarity scores
        """
        if self.chunks is None:
            raise ValueError("Index not built. Call build_index first.")

        # Ensure query embedding is 2D and float32
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)

        # Search using GPU index if available, otherwise CPU
        search_index = self.gpu_index if self.gpu_index is not None else self.index
        distances, indices = search_index.search(query_embedding, top_k)

        # Prepare results
        results = []
        for i, (idx, score) in enumerate(zip(indices[0], distances[0])):
            if idx < len(self.chunks):  # Valid index
                chunk = self.chunks[idx]
                results.append({
                    'rank': i + 1,
                    'similarity_score': float(score),
                    'text': chunk['text'],
                    'metadata': chunk['metadata']
                })

        return results

    def save(self, index_path: str, chunks_path: str):
        """
        Save the FAISS index and chunks to disk.

        Args:
            index_path: Path to save the FAISS index
            chunks_path: Path to save the chunks (pickle file)
        """
        # Save index (use CPU index for saving)
        cpu_index = faiss.index_gpu_to_cpu(self.gpu_index) if self.gpu_index is not None else self.index
        faiss.write_index(cpu_index, index_path)

        # Save chunks
        with open(chunks_path, 'wb') as f:
            pickle.dump(self.chunks, f)

        print(f"Index saved to {index_path}")
        print(f"Chunks saved to {chunks_path}")

    def load(self, index_path: str, chunks_path: str):
        """
        Load the FAISS index and chunks from disk.

        Args:
            index_path: Path to the saved FAISS index
            chunks_path: Path to the saved chunks (pickle file)
        """
        # Load index
        self.index = faiss.read_index(index_path)

        # Move to GPU if requested
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.gpu_index = faiss.index_cpu_to_gpu(res, 0, self.index)
                print("Index loaded successfully on GPU")
            except Exception as e:
                print(f"GPU not available, using CPU: {e}")

        # Load chunks
        with open(chunks_path, 'rb') as f:
            self.chunks = pickle.load(f)

        print(f"Index loaded from {index_path}")
        print(f"Chunks loaded from {chunks_path} ({len(self.chunks)} chunks)")

    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the index.

        Returns:
            Dictionary with index statistics
        """
        if self.index is None:
            return {'status': 'not_built'}

        return {
            'status': 'built',
            'total_vectors': self.index.ntotal,
            'embedding_dim': self.embedding_dim,
            'using_gpu': self.gpu_index is not None,
            'total_chunks': len(self.chunks) if self.chunks else 0
        }


def create_vector_store(embeddings: np.ndarray,
                       chunks: List[Dict[str, Any]],
                       use_gpu: bool = True) -> FAISSVectorStore:
    """
    Convenience function to create and build a vector store.

    Args:
        embeddings: NumPy array of embeddings
        chunks: List of chunk dictionaries
        use_gpu: Whether to use GPU acceleration

    Returns:
        Built FAISSVectorStore instance
    """
    embedding_dim = embeddings.shape[1]
    store = FAISSVectorStore(embedding_dim, use_gpu)
    store.build_index(embeddings, chunks)
    return store
