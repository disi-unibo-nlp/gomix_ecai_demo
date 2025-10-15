import os
import torch
import pickle
from sklearn.decomposition import TruncatedSVD
from typing import List, Optional

TASK_DATASET_PATH = os.environ["TASK_DATASET_PATH"]
assert TASK_DATASET_PATH, 'Environment variable \'TASK_DATASET_PATH\' must be declared.'

ALL_PROTEIN_SEQUENCE_EMBEDDINGS_DIR = os.path.join(TASK_DATASET_PATH, 'all_protein_sequence_embeddings/esm2_t48_15B_UR50D')
ALL_PROTEIN_TEXT_FEATURES_EMBEDDINGS_DIR = os.path.join(TASK_DATASET_PATH, 'all_protein_text_features_OpenAI_embeddings')
ALL_PROTEIN_INTERPRO_FEATURES_FILE = os.path.join(TASK_DATASET_PATH, 'all_protein_interpro_features.pickle')


class ProteinEmbeddingLoader:
    _AVAILABLE_TYPES = {'sequence', 'text', 'interpro'}

    _PROT_SEQUENCE_EMBEDDING_SIZE = 5120  # `2560` for esm2-3B embeddings, `5120` for esm2-15B embeddings
    _PROT_TEXT_EMBEDDING_SIZE = 1536  # Number of elements in a single protein text features embedding vector
    _PROT_INTERPRO_EMBEDDING_SIZE = 512

    def __init__(self, types: Optional[List[str]] = None):
        if not types:
            types = ['sequence']  # Default.

        # Validate types.
        assert types, 'You have to specify at least one type.'
        assert all(t in self._AVAILABLE_TYPES for t in types), f'Unknown type in {types}'
        assert len(set(types)) == len(types), f'You cannot use the same type twice: {types}'

        self._types = types

        if 'interpro' in self._types:
            self._interpro_embeddings = self._preload_all_interpro_embeddings()

    def load(self, prot_id: str) -> torch.Tensor:
        result = torch.Tensor()
        for type in self._types:
            method = getattr(self, f'_load_{type}_embedding')
            result = torch.cat((result, method(prot_id)))
        assert result.shape == (self.get_embedding_size(),)
        return result

    def get_embedding_size(self):
        size = 0
        for type in self._types:
            if type == 'sequence':
                size += self._PROT_SEQUENCE_EMBEDDING_SIZE
            elif type == 'text':
                size += self._PROT_TEXT_EMBEDDING_SIZE
            elif type == 'interpro':
                size += self._PROT_INTERPRO_EMBEDDING_SIZE
            else:
                raise ValueError(f'Unknown type: {type}')
        return size

    def _preload_all_interpro_embeddings(self) -> dict:
        with open(ALL_PROTEIN_INTERPRO_FEATURES_FILE, 'rb') as f:
            d = pickle.load(f)

        svd = TruncatedSVD(n_components=self._PROT_INTERPRO_EMBEDDING_SIZE)
        reduced_matrix = svd.fit_transform(d['coo_matrix'])

        result = {prot_id: embedding for prot_id, embedding in zip(d['ordered_protein_ids'], reduced_matrix)}
        assert len(result) == len(d['ordered_protein_ids']), 'Wrong number of embeddings.'
        assert all(len(embedding) == self._PROT_INTERPRO_EMBEDDING_SIZE for embedding in result.values()), 'Wrong embedding size.'
        return result

    def _load_sequence_embedding(self, prot_id) -> torch.Tensor:
        d = torch.load(f'{ALL_PROTEIN_SEQUENCE_EMBEDDINGS_DIR}/{prot_id}.pt')['mean_representations']
        d = d[max(d, key=int)]
        assert type(d) == torch.Tensor and d.shape == (self._PROT_SEQUENCE_EMBEDDING_SIZE,)
        return d

    def _load_text_embedding(self, prot_id) -> torch.Tensor:
        pickle_file_path = os.path.join(ALL_PROTEIN_TEXT_FEATURES_EMBEDDINGS_DIR, f'{prot_id}.pickle')
        if not os.path.exists(pickle_file_path):
            return torch.zeros(self._PROT_TEXT_EMBEDDING_SIZE)

        with open(pickle_file_path, 'rb') as f:
            embedding = pickle.load(f)
        assert len(embedding) == self._PROT_TEXT_EMBEDDING_SIZE
        return torch.Tensor(embedding)

    def _load_interpro_embedding(self, prot_id) -> torch.Tensor:
        if prot_id not in self._interpro_embeddings:
            return torch.zeros(self._PROT_INTERPRO_EMBEDDING_SIZE)

        return torch.Tensor(self._interpro_embeddings[prot_id])
