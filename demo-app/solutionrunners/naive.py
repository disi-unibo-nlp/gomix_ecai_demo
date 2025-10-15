import sys
import os
from pathlib import Path
import json
from typing import List

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE / "../../"))
from gomix.src.solution.components.naive.NaiveLearner import NaiveLearner

TASK_DATASET_PATH = os.path.join(HERE, "../../gomix/src/data/processed/task_datasets/2016")
TRAIN_ANNOTATIONS_FILE_PATH = os.path.join(TASK_DATASET_PATH, 'propagated_annotations', 'train.json')

def predict() -> List[tuple]:
    with open(TRAIN_ANNOTATIONS_FILE_PATH, 'r') as f:
        train_annotations = json.load(f)  # dict: prot ID -> list of GO terms

    learner = NaiveLearner(train_annotations)

    return [(go_term, score) for go_term, score in learner.predict().items()]
