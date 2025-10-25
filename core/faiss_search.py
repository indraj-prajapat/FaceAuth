"""
FAISS Search Module
Manages face embedding database and similarity search
"""
import numpy as np
import faiss
from config import FAISS_CONFIG

class FAISSSearcher:
    def __init__(self, embedding_dim=512):
        self.embedding_dim = embedding_dim
        self.indices = {}
        self.db_ids = []

        # Initialize indices for each backbone
        for model_name in ['arcface', 'adaface', 'elastic']:
            self.indices[model_name] = faiss.IndexFlatIP(embedding_dim)

    def add_to_database(self, embeddings, db_id):
        """
        Add embeddings to database

        Args:
            embeddings: Dict with embeddings from each model
            db_id: Database ID for this entry
        """
        for model_name in ['arcface', 'adaface', 'elastic']:
            emb = embeddings[model_name].reshape(1, -1).astype('float32')
            # L2 normalize
            faiss.normalize_L2(emb)
            self.indices[model_name].add(emb)

        self.db_ids.append(db_id)

    def search(self, probe_embeddings, k=10):
        """
        Search for similar faces in database

        Args:
            probe_embeddings: Dict with probe embeddings from each model
            k: Number of nearest neighbors to return

        Returns:
            dict: Search results for each model
        """
        results = {}

        for model_name in ['arcface', 'adaface', 'elastic']:
            emb = probe_embeddings[model_name].reshape(1, -1).astype('float32')
            # L2 normalize
            faiss.normalize_L2(emb)

            # Search
            distances, indices = self.indices[model_name].search(emb, k)

            # Convert to similarity scores (cosine similarity)
            similarities = distances[0]
            db_indices = indices[0]

            results[model_name] = (similarities, db_indices)

        return results

    def get_db_size(self):
        """Return size of database"""
        return len(self.db_ids)
    def get_db_id_by_index(self, idx: int):
        if 0 <= idx < len(self.db_ids):
            return self.db_ids[idx]
        return None

    def save_index(self, path):
        """Save FAISS indices to disk"""
        for model_name, index in self.indices.items():
            faiss.write_index(index, f"{path}/{model_name}.index")

    def load_index(self, path):
        """Load FAISS indices from disk"""
        for model_name in ['arcface', 'adaface', 'elastic']:
            self.indices[model_name] = faiss.read_index(f"{path}/{model_name}.index")
