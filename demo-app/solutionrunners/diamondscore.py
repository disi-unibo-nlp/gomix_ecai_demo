import sys
import os
from pathlib import Path
import json
from typing import List

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE / "../../"))
from gomix.src.solution.components.diamondscore.DiamondScoreLearner import DiamondScoreLearner

TASK_DATASET_PATH = os.path.join(HERE, "../../gomix/src/data/processed/task_datasets/2016")
TRAIN_ANNOTATIONS_FILE_PATH = os.path.join(TASK_DATASET_PATH, 'propagated_annotations', 'train.json')
DIAMOND_SCORES_FILE_PATH = os.path.join(TASK_DATASET_PATH, 'all_proteins_diamond.res')

def predict(test_protein_id) -> List[tuple]:
    with open(TRAIN_ANNOTATIONS_FILE_PATH, 'r') as f:
        train_annotations = json.load(f)  # dict: prot ID -> list of GO terms
    diamond_scores_file_path = os.path.join(TASK_DATASET_PATH, 'all_proteins_diamond.res')

    learner = DiamondScoreLearner(train_annotations, diamond_scores_file_path)

    return [(go_term, score) for go_term, score in learner.predict(test_protein_id).items()]
