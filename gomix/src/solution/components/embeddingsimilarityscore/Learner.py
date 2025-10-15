import faiss
import os
import numpy as np
from uuid import uuid4
import sys
from pathlib import Path

root = next(p for p in Path(__file__).resolve().parents if (p / "src").exists())
sys.path.insert(0, str(root))
from src.utils.ProteinEmbeddingLoader import ProteinEmbeddingLoader

THIS_DIR = os.path.dirname(os.path.realpath(__file__))


class Learner:
    _TEMP_VECTORS_DB_FILE_PATH = os.path.join(THIS_DIR, f'../../../data/temp_vectors_db_{uuid4()}')

    def __init__(self, train_annotations: dict, prot_embedding_loader: ProteinEmbeddingLoader):
        self._train_annotations = train_annotations
        self._protein_embedding_loader = prot_embedding_loader
        self._init_training_vectors_db()

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
