import numpy as np


class DiamondScoreLearner:
    def __init__(self, train_annotations: dict, diamond_scores_file_path: str):
        self._train_annotations = train_annotations
        self._diamond_scores = self._read_diamond_scores(diamond_scores_file_path)

    def predict(self, prot_id) -> dict:
        if prot_id in self._diamond_scores:
            similar_proteins = self._diamond_scores[prot_id]
        else:
            return {}

        # Keep only the similar proteins present training annotations.
        similar_proteins = {sim_prot_id: score for sim_prot_id, score in similar_proteins.items() if sim_prot_id in self._train_annotations}

        go_terms = set()
        total_score = 0.0
        for sim_prot_id, score in similar_proteins.items():
            go_terms |= set(self._train_annotations[sim_prot_id])
            total_score += score

        go_terms = list(sorted(go_terms))
        go_terms_scores = np.zeros(len(go_terms), dtype=np.float32)
        for i, go_term in enumerate(go_terms):
            for sim_prot_id, score in similar_proteins.items():
                if go_term in self._train_annotations[sim_prot_id]:
                    go_terms_scores[i] += score
            go_terms_scores[i] /= total_score

        return {go_term: go_terms_scores[i] for i, go_term in enumerate(go_terms)}

    @staticmethod
    def _read_diamond_scores(diamond_scores_file_path: str) -> dict:
        scores = {}
        with open(diamond_scores_file_path) as f:
            for line in f:
                prot_a, prot_b, score = line.strip().split()
                if prot_a not in scores:
                    scores[prot_a] = {}
                scores[prot_a][prot_b] = float(score)
        return scores
