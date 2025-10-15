import sys
from pathlib import Path
import json
import os
sys.path.append(str(Path(__file__).resolve().parents[4]))
from src.solution.components.interactionscore.InteractionScoreLearner import InteractionScoreLearner
from src.utils.predictions_evaluation.evaluate import evaluate_with_deepgoplus_evaluator

TASK_DATASET_PATH = os.environ.get("TASK_DATASET_PATH")
assert TASK_DATASET_PATH, 'Environment variable \'TASK_DATASET_PATH\' must be declared.'


"""
Example usage:
TASK_DATASET_PATH=data/processed/task_datasets/2016 python src/solution/components/interactionscore/demo.py
"""
def main():
    train_annotations_file_path = os.path.join(TASK_DATASET_PATH, 'propagated_annotations', 'train.json')
    test_annotations_file_path = os.path.join(TASK_DATASET_PATH, 'annotations', 'test.json')
    gene_ontology_file_path = os.path.join(TASK_DATASET_PATH, 'go.obo')
    ppi_file_path = os.path.join(TASK_DATASET_PATH, 'all_proteins_STRING_interactions.json')

    with open(train_annotations_file_path, 'r') as f:
        train_annotations = json.load(f)  # dict: prot ID -> list of GO terms

    with open(test_annotations_file_path, 'r') as f:
        test_annotations = json.load(f)  # dict: prot ID -> list of GO terms

    learner = InteractionScoreLearner(train_annotations, ppi_file_path)

    predictions = {
        prot_id: [(go_term, score) for go_term, score in learner.predict(prot_id).items()]
        for prot_id in test_annotations.keys()
    }

    print('Evaluating InteractionScore predictions...')
    evaluate_with_deepgoplus_evaluator(
        gene_ontology_file_path=gene_ontology_file_path,
        predictions=predictions,
        ground_truth=test_annotations
    )


if __name__ == '__main__':
    main()
