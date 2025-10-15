import sys
import os
from pathlib import Path
import json
from typing import List

HERE = Path(__file__).resolve().parent
TASK_DATASET_PATH = os.path.join(HERE, "../../gomix/src/data/processed/task_datasets/2016")
TRAIN_ANNOTATIONS_FILE_PATH = os.path.join(TASK_DATASET_PATH, 'propagated_annotations', 'train.json')

os.environ["TASK_DATASET_PATH"] = TASK_DATASET_PATH

sys.path.append(str(HERE / "../../"))
from gomix.src.solution.components.embeddingsimilarityscore.Learner import Learner
from gomix.src.utils.ProteinEmbeddingLoader import ProteinEmbeddingLoader

def predict(test_protein_id) -> List[tuple]:
    with open(TRAIN_ANNOTATIONS_FILE_PATH, 'r') as f:
        train_annotations = json.load(f)  # dict: prot ID -> list of GO terms

    learner = Learner(
        train_annotations=train_annotations,
        prot_embedding_loader=ProteinEmbeddingLoader(types=['sequence'])
    )

    return [(go_term, score) for go_term, score in learner.predict(test_protein_id).items()]
