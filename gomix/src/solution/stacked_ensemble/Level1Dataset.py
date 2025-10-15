import numpy as np
from typing import List


class Level1Dataset:
    def __init__(self, go_terms_vocabulary: List[str], base_predictions: List[dict], ground_truth: dict = None):
        self.go_term_to_index = {go_term: i for i, go_term in enumerate(go_terms_vocabulary)}
        self._init_prot_id_to_index(base_predictions)

        """
        Prediction is {prot_id: [(go_term, score), ...]}
        Ground truth is {prot_id: [go_term, ...]}
        """
        self.base_predictions = base_predictions  # list of predictions by base models
        self.ground_truth = ground_truth

        if self.ground_truth is not None:
            self._verify_predictions_and_ground_truth_refer_to_same_proteins()

        print(f'Created a level-1 dataset with {len(self.prot_id_to_index)} proteins and {len(self.go_term_to_index)} GO terms.')

    def get_base_scores_array(self) -> np.ndarray:
        n_base_models = len(self.base_predictions)
        base_scores = np.zeros((len(self.prot_id_to_index), len(self.go_term_to_index), n_base_models))
        for model_idx, pred in enumerate(self.base_predictions):
            for prot_id, go_terms_scores in pred.items():
                prot_idx = self.prot_id_to_index[prot_id]
                for go_term, score in go_terms_scores:
                    if go_term in self.go_term_to_index:
                        go_term_idx = self.go_term_to_index[go_term]
                        base_scores[prot_idx, go_term_idx, model_idx] = score

        return base_scores

    def get_labels_array(self) -> np.ndarray:
        assert self.ground_truth is not None
        labels = np.zeros((len(self.prot_id_to_index), len(self.go_term_to_index)))
        for prot_id, go_terms in self.ground_truth.items():
            prot_idx = self.prot_id_to_index[prot_id]
            for go_term in go_terms:
                go_term_idx = self.go_term_to_index[go_term]
                labels[prot_idx, go_term_idx] = 1

        return labels

    def convert_predictions_array_to_dict(self, predictions: np.ndarray) -> dict:
        # Convert predictions from ndarray to dict
        # Predictions is ndarray of shape (n_prot, n_go_terms)
        # Return dict of {prot_id: [(go_term, score), ...]}
        prot_id_to_predictions = {}
        for prot_id, prot_idx in self.prot_id_to_index.items():
            prot_id_to_predictions[prot_id] = []
            for go_term, go_term_idx in self.go_term_to_index.items():
                score = predictions[prot_idx, go_term_idx]
                prot_id_to_predictions[prot_id].append((go_term, score))

        return prot_id_to_predictions

    def _verify_predictions_and_ground_truth_refer_to_same_proteins(self) -> None:
        gt_proteins = sorted(self.ground_truth.keys())
        assert all([sorted(pred.keys()) == gt_proteins for pred in self.base_predictions])

    def _init_prot_id_to_index(self, base_predictions: List[dict]) -> None:
        prot_ids = set()
        for pred in base_predictions:
            for prot_id in pred:
                prot_ids.add(prot_id)

        self.prot_id_to_index = {prot_id: i for i, prot_id in enumerate(prot_ids)}
