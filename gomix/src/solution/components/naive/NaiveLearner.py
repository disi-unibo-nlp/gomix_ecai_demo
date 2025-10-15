from collections import defaultdict


class NaiveLearner:
    def __init__(self, train_annotations: dict):
        self.go_term_freq = defaultdict(int)
        for prot_id, go_terms in train_annotations.items():
            for go_term in set(go_terms):
                self.go_term_freq[go_term] += 1
        total_proteins = len(train_annotations)
        for go_term in self.go_term_freq.keys():
            self.go_term_freq[go_term] /= total_proteins

    def predict(self):
        # It's naive, so it predicts the same for all proteins.
        top_k_predictions = {k: v for k, v in sorted(self.go_term_freq.items(), key=lambda item: item[1], reverse=True)[:300]}
        return top_k_predictions
