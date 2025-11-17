"""Quick test script to build the index with progress tracking"""

from config import Config
from data_loader import load_twitter_api_docs
from chunker import create_documentation_chunks
from embedder import DocumentEmbedder
from vector_store import FAISSVectorStore
import time

print("Configuration:")
Config.display_config()
print("\n" + "="*80 + "\n")

print("Step 1: Loading documentation...")
start = time.time()
endpoints = load_twitter_api_docs(Config.POSTMAN_COLLECTION_PATH)
print(f"✓ Loaded {len(endpoints)} endpoints in {time.time()-start:.2f}s")

print("\nStep 2: Creating chunks...")
start = time.time()
chunks = create_documentation_chunks(endpoints)
print(f"✓ Created {len(chunks)} chunks in {time.time()-start:.2f}s")

print("\nStep 3: Loading embedding model...")
start = time.time()
embedder = DocumentEmbedder(model_name=Config.EMBEDDING_MODEL, device=Config.get_device())
embedder.load_model()
print(f"✓ Model loaded in {time.time()-start:.2f}s")

print("\nStep 4: Generating embeddings...", flush=True)
start = time.time()
embeddings = embedder.embed_chunks(chunks)
print(f"✓ Generated embeddings in {time.time()-start:.2f}s", flush=True)
print(f"  Embeddings shape: {embeddings.shape}", flush=True)

print("\nStep 5: Building FAISS index...")
start = time.time()
embedding_dim = embeddings.shape[1]
vector_store = FAISSVectorStore(embedding_dim, use_gpu=Config.USE_GPU)
vector_store.build_index(embeddings, chunks)
print(f"✓ Index built in {time.time()-start:.2f}s")

print("\nStep 6: Saving index...")
start = time.time()
vector_store.save(f'{Config.INDEX_DIR}/faiss.index', f'{Config.INDEX_DIR}/chunks.pkl')
print(f"✓ Index saved in {time.time()-start:.2f}s")

print("\n✅ All done!")
