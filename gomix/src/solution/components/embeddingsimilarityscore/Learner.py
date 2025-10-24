import faiss
import os
import numpy as np
from uuid import uuid4
import sys
from pathlib import Path
import json

root = next(p for p in Path(__file__).resolve().parents if (p / "src").exists())
sys.path.insert(0, str(root))
from src.utils.ProteinEmbeddingLoader import ProteinEmbeddingLoader

THIS_DIR = os.path.dirname(os.path.realpath(__file__))

# Path to pre-built FAISS index
PREBUILT_INDEX_DIR = os.path.join(root, 'src', 'demo_utils', 'faiss_index')
PREBUILT_INDEX_PATH = os.path.join(PREBUILT_INDEX_DIR, 'embeddings_faiss_index.faiss')
PREBUILT_PROTEIN_IDS_PATH = os.path.join(PREBUILT_INDEX_DIR, 'protein_ids_order.json')
PREBUILT_METADATA_PATH = os.path.join(PREBUILT_INDEX_DIR, 'faiss_index_metadata.json')


class Learner:
    _TEMP_VECTORS_DB_FILE_PATH = os.path.join(THIS_DIR, f'../../../data/temp_vectors_db_{uuid4()}')

    def __init__(self, train_annotations: dict, prot_embedding_loader: ProteinEmbeddingLoader):
        self._train_annotations = train_annotations
        self._protein_embedding_loader = prot_embedding_loader
        self._use_prebuilt_index = False
        self._training_vectors_db = None
        
        if not self._try_load_prebuilt_index():
            print("⚠️  No pre-built index found. Building FAISS index from scratch...")
            print("    This will take 30-40 minutes. To avoid this delay, run:")
            print("    ./scripts/build-faiss-index.sh")
            self._init_training_vectors_db()

    def _try_load_prebuilt_index(self) -> bool:
        """
        Try to load pre-built FAISS index.
        Returns True if successful, False otherwise.
        """
        # Check if all required files exist
        if not os.path.exists(PREBUILT_INDEX_PATH):
            return False
        if not os.path.exists(PREBUILT_PROTEIN_IDS_PATH):
            return False
        if not os.path.exists(PREBUILT_METADATA_PATH):
            return False
        
        try:
            # print(f"Loading FAISS index from: {PREBUILT_INDEX_DIR}")
            
            # Load metadata
            with open(PREBUILT_METADATA_PATH, 'r') as f:
                metadata = json.load(f)
            
            # Verify metadata matches current configuration
            expected_size = self._protein_embedding_loader.get_embedding_size()
            if metadata.get('embedding_size') != expected_size:
                print(f"   ⚠️  Embedding size mismatch: expected {expected_size}, got {metadata.get('embedding_size')}")
                return False
            
            # Load protein IDs order
            with open(PREBUILT_PROTEIN_IDS_PATH, 'r') as f:
                self._prebuilt_protein_ids = json.load(f)
            
            # Verify protein IDs match training annotations
            prebuilt_set = set(self._prebuilt_protein_ids)
            train_set = set(self._train_annotations.keys())
            
            if prebuilt_set != train_set:
                missing_in_index = train_set - prebuilt_set
                extra_in_index = prebuilt_set - train_set
                
                if missing_in_index:
                    print(f"   ⚠️  {len(missing_in_index)} proteins in training set not found in index")
                if extra_in_index:
                    print(f"   ⚠️  {len(extra_in_index)} proteins in index not found in training set")
                
                # Allow small mismatches but warn user
                if len(missing_in_index) > 100 or len(extra_in_index) > 100:
                    print(f"   ❌ Too many mismatches. Index may be outdated.")
                    return False
            
            # Load FAISS index
            self._training_vectors_db = _PrebuiltVectorDatabase(PREBUILT_INDEX_PATH)
            self._use_prebuilt_index = True
                        
            return True
            
        except Exception as e:
            print(f"   ❌ Failed to load pre-built index: {e}")
            return False

    def _init_training_vectors_db(self):
        if os.path.exists(self._TEMP_VECTORS_DB_FILE_PATH):
            os.remove(self._TEMP_VECTORS_DB_FILE_PATH)

        self._training_vectors_db = _VectorDatabase(
            embedding_size=self._protein_embedding_loader.get_embedding_size(),
            db_file_path=self._TEMP_VECTORS_DB_FILE_PATH
        )
        # The order of add_to_index() calls is important, as later in predict() we rely on that.
        for prot_id in self._train_annotations:
            embedding = self._protein_embedding_loader.load(prot_id)
            embedding = embedding.numpy().reshape(1, -1)
            self._training_vectors_db.add_to_index(embedding=embedding)
        self._training_vectors_db.save_index()

    def __del__(self):
        if os.path.exists(self._TEMP_VECTORS_DB_FILE_PATH):
            os.remove(self._TEMP_VECTORS_DB_FILE_PATH)

    def predict(self, prot_id) -> dict:
        prot_embedding = self._protein_embedding_loader.load(prot_id)
        prot_embedding = prot_embedding.numpy().reshape(1, -1)
        D, I = self._training_vectors_db.search(prot_embedding, k=60)

        # Get protein IDs based on index type
        if self._use_prebuilt_index:
            # Use pre-built protein IDs order
            cosine_similarities = {self._prebuilt_protein_ids[idx]: score for idx, score in zip(I[0], D[0])}
        else:
            # Use train_annotations keys order (original behavior)
            cosine_similarities = {list(self._train_annotations.keys())[idx]: score for idx, score in zip(I[0], D[0])}
        
        sorted_similarities = sorted(cosine_similarities.items(), key=lambda x: x[1], reverse=True)
        similar_proteins_weighted_scores = {prot: score / (rank + 1) for rank, (prot, score) in enumerate(sorted_similarities)}

        go_terms_scores = {}
        for similar_prot_id, score in similar_proteins_weighted_scores.items():
            for go_term in self._train_annotations[similar_prot_id]:
                go_terms_scores[go_term] = go_terms_scores.get(go_term, 0) + score

        # Normalize GO terms' scores.
        total_score = sum(similar_proteins_weighted_scores.values())
        for go_term in go_terms_scores:
            go_terms_scores[go_term] /= total_score

        return go_terms_scores


class _VectorDatabase:
    def __init__(self, embedding_size: int, db_file_path: str):
        self._db_path = db_file_path
        if not os.path.exists(self._db_path):
            self._index = faiss.index_factory(embedding_size, "Flat", faiss.METRIC_INNER_PRODUCT)
        else:
            self._load_index()

    def add_to_index(self, embedding: np.array):
        embedding = embedding.astype('float32')
        faiss.normalize_L2(embedding)
        self._index.add(embedding)

    def save_index(self):
        if os.path.exists(self._db_path):
            raise ValueError(f"Index at {self._db_path} already exists. Avoiding overwrite.")
        faiss.write_index(self._index, self._db_path)

    def search(self, embedding: np.array, k):
        embedding = embedding.astype('float32')
        faiss.normalize_L2(embedding)
        D, I = self._index.search(embedding, k)
        return D, I

    def _load_index(self):
        self._index = faiss.read_index(self._db_path)


class _PrebuiltVectorDatabase:
    """
    Wrapper for pre-built FAISS index that provides the same interface as _VectorDatabase.
    This class only supports loading and searching, not building.
    """
    def __init__(self, index_path: str):
        self._index = faiss.read_index(index_path)
    
    def search(self, embedding: np.array, k):
        embedding = embedding.astype('float32')
        faiss.normalize_L2(embedding)
        D, I = self._index.search(embedding, k)
        return D, I
