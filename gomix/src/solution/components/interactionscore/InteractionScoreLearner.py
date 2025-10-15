import numpy as np
import json


class InteractionScoreLearner:
    def __init__(self, train_annotations: dict, ppi_file_path: str):
        self._train_annotations = train_annotations
        self._interaction_scores = self._read_interaction_scores(ppi_file_path)

    def predict(self, prot_id) -> dict:
        if prot_id in self._interaction_scores:
            interacting_proteins = self._interaction_scores[prot_id]
        else:
            return {}

        # Keep only the interacting proteins present in the training set.
        interacting_proteins = {int_prot_id: score for int_prot_id, score in interacting_proteins.items() if int_prot_id in self._train_annotations}

        go_terms = set()
        total_score = 0.0
        for interacting_prot_id, score in interacting_proteins.items():
            go_terms |= set(self._train_annotations[interacting_prot_id])
            total_score += score

        go_terms = list(sorted(go_terms))
        go_terms_scores = np.zeros(len(go_terms), dtype=np.float32)
        for i, go_term in enumerate(go_terms):
            for interacting_prot_id, score in interacting_proteins.items():
                if go_term in self._train_annotations[interacting_prot_id]:
                    go_terms_scores[i] += score
            go_terms_scores[i] /= total_score

        return {go_term: go_terms_scores[i] for i, go_term in enumerate(go_terms)}

    @staticmethod
    def _read_interaction_scores(ppi_file_path: str) -> dict:
        with open(ppi_file_path) as f:
            scores = json.load(f)  # dict: {prot_id: {interacting_prot_id: score, ...}, ...}

        # Keep only the top-k scores for each protein.
        top_k = 30
        return {
            prot_id: dict(sorted(prot_scores.items(), key=lambda x: x[1], reverse=True)[:top_k])
            for prot_id, prot_scores in scores.items()
        }
