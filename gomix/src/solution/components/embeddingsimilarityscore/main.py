import sys
from pathlib import Path
import json
import random
import os
from tqdm import tqdm
sys.path.append(str(Path(__file__).resolve().parents[4]))
from src.utils.ProteinEmbeddingLoader import ProteinEmbeddingLoader
from src.solution.components.embeddingsimilarityscore.Learner import Learner
from src.utils.predictions_evaluation.evaluate import evaluate_with_deepgoplus_evaluator

EMBEDDING_TYPES = ['sequence']


"""
To run the demo:
TASK_DATASET_PATH=data/processed/task_datasets/2016 python src/solution/components/embeddingsimilarityscore/main.py
"""
if __name__ == '__main__':
    random.seed(0)

    TASK_DATASET_PATH = os.environ.get("TASK_DATASET_PATH")
    assert TASK_DATASET_PATH, 'Environment variable \'TASK_DATASET_PATH\' must be declared.'

    train_annotations_file_path = os.path.join(TASK_DATASET_PATH, 'propagated_annotations', 'train.json')
    test_annotations_file_path = os.path.join(TASK_DATASET_PATH, 'annotations', 'test.json')
    gene_ontology_file_path = os.path.join(TASK_DATASET_PATH, 'go.obo')

    with open(train_annotations_file_path, 'r') as f:
        train_annotations = json.load(f)  # dict: prot ID -> list of GO terms

    with open(test_annotations_file_path, 'r') as f:
        test_annotations = json.load(f)  # dict: prot ID -> list of GO terms

    learner = Learner(
        train_annotations=train_annotations,
        prot_embedding_loader=ProteinEmbeddingLoader(types=EMBEDDING_TYPES)
    )

    print('Generating predictions...')
    predictions = {
        prot_id: [(go_term, score) for go_term, score in learner.predict(prot_id).items()]
        for prot_id in tqdm(test_annotations.keys())
    }

    print(f'Evaluating EmbeddingSimilarityScore predictions (with embedding type = {EMBEDDING_TYPES})...')
    evaluate_with_deepgoplus_evaluator(
        gene_ontology_file_path=gene_ontology_file_path,
        predictions=predictions,
        ground_truth=test_annotations
    )
