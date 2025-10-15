import torch
import json
import os
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
import numpy as np
import sys
from typing import List
from pathlib import Path
import random
sys.path.append(str(Path(__file__).resolve().parents[4]))
from src.solution.components.FC_on_embeddings.ProteinToGOModel import ProteinToGOModel
from src.utils.EmbeddedProteinsDataset import EmbeddedProteinsDataset
from src.utils.predictions_evaluation.evaluate import evaluate_with_deepgoplus_evaluator
from src.utils.ProteinEmbeddingLoader import ProteinEmbeddingLoader

TASK_DATASET_PATH = os.environ["TASK_DATASET_PATH"]
assert TASK_DATASET_PATH, 'Environment variable \'TASK_DATASET_PATH\' must be declared.'

PROPAGATED_TRAIN_ANNOTS_FILE_PATH = os.path.join(TASK_DATASET_PATH, 'propagated_annotations/train.json')
OFFICIAL_TEST_ANNOTS_FILE_PATH = os.path.join(TASK_DATASET_PATH, 'annotations/test.json')
GENE_ONTOLOGY_FILE_PATH = os.path.join(TASK_DATASET_PATH, 'go.obo')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps')

PROT_EMBEDDING_LOADER = ProteinEmbeddingLoader()


def main():
    random.seed(0)
    torch.manual_seed(0)

    train_dataset = _make_training_dataset()

    model = make_and_train_model_on(train_dataset)

    print('Training finished. Let\'s now evaluate on test set (with the official criteria).')
    _evaluate_for_testing_with_official_criteria(model, go_term_to_index=train_dataset.go_term_to_index)


def make_and_train_model_on(dataset) -> ProteinToGOModel:
    train_set, val_set = train_test_split(dataset, test_size=0.2)
    train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_set, batch_size=32)

    print(f"Training using device: {DEVICE}")

    model = make_model_on_device(dataset.go_term_to_index)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best_val_f_max = -np.inf
    best_epoch = 0
    MAX_EPOCHS = 11
    for epoch in range(1, MAX_EPOCHS+1):
        print(f"Epoch {epoch}: Learning rate = {optimizer.param_groups[0]['lr']}")
        model.train()
        train_loss = 0.0
        for i, (prot_embeddings, targets) in enumerate(train_dataloader):
            prot_embeddings = prot_embeddings.to(DEVICE)
            targets = targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(prot_embeddings)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if i % 200 == 199:  # print every 200 mini-batches
                print(f'[{epoch}, {i + 1:5d}] loss: {train_loss / 200}')
                train_loss = 0.0

        val_loss, performances_by_threshold = _evaluate_for_validation(model, val_dataloader, loss_fn)
        print(f'[{epoch}, validation] val_loss: {val_loss:.4f}')

        f_max = 0
        opt_threshold = 0
        for threshold, (precision, recall) in performances_by_threshold.items():
            if precision + recall > 0:  # Avoid division by zero
                f1_score = 2 * precision * recall / (precision + recall)
                if f1_score > f_max:
                    f_max = f1_score
                    opt_threshold = threshold
        print(f'[{epoch}, validation] F_max: {f_max:.4f} (at optimal threshold t={opt_threshold})')

        if f_max > best_val_f_max:
            best_val_f_max = f_max
            best_epoch = epoch
        elif epoch - best_epoch > 2:  # Early stopping.
            print(f'Early stopping. Best F_max score on validation set was {best_val_f_max:.4f} at epoch {best_epoch}')
            break

        print('——')
        scheduler.step()

    return model


def make_model_on_device(go_term_to_index: dict):
    return ProteinToGOModel(
        protein_embedding_size=PROT_EMBEDDING_LOADER.get_embedding_size(),
        output_size=len(go_term_to_index)
    ).to(DEVICE)


def make_training_dataset_with_annotations(annots) -> EmbeddedProteinsDataset:
    return EmbeddedProteinsDataset(annotations=annots, prot_embedding_loader=PROT_EMBEDDING_LOADER)


def _make_training_dataset():
    with open(PROPAGATED_TRAIN_ANNOTS_FILE_PATH, 'r') as f:
        train_annotations = json.load(f)

    return make_training_dataset_with_annotations(train_annotations)


def _evaluate_for_validation(model, dataloader, loss_fn):
    model.to(DEVICE)
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for prot_embeddings, targets in dataloader:
            prot_embeddings = prot_embeddings.to(DEVICE)
            targets = targets.to(DEVICE)
            outputs = model(prot_embeddings)
            running_loss += loss_fn(outputs, targets).item()
            all_preds.append(torch.sigmoid(outputs))
            all_targets.append(targets)
    running_loss /= len(dataloader)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    performances_by_threshold = {}

    for threshold in np.round(np.arange(0.01, 0.9, 0.01), 2):
        polarized_preds = (all_preds >= threshold).float()
        true_positives = (polarized_preds * all_targets).sum(dim=1)
        false_positives = (polarized_preds * (1 - all_targets)).sum(dim=1)

        precision = true_positives.sum() / (true_positives.sum() + false_positives.sum())
        recall = true_positives.sum() / all_targets.sum()

        performances_by_threshold[threshold] = (precision, recall)

    return running_loss, performances_by_threshold


def _evaluate_for_testing_with_official_criteria(model, go_term_to_index: dict):
    with open(OFFICIAL_TEST_ANNOTS_FILE_PATH, 'r') as f:
        test_annotations = json.load(f)
    prot_ids = list(test_annotations.keys())

    predictions = predict_and_transform_predictions_to_dict(model, prot_ids, go_term_to_index)

    evaluate_with_deepgoplus_evaluator(
        gene_ontology_file_path=GENE_ONTOLOGY_FILE_PATH,
        predictions=predictions,
        ground_truth=test_annotations
    )


def predict_and_transform_predictions_to_dict(model: ProteinToGOModel, prot_ids: List[str], go_term_to_index: dict) -> dict:
    index_to_go_term = {v: k for k, v in go_term_to_index.items()}

    prev_device = _get_model_device(model)
    model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        all_predictions = {}
        batch_size = 256
        for i in range(0, len(prot_ids), batch_size):
            batch_prot_ids = prot_ids[i:i + batch_size]
            batch_prot_embeddings = torch.stack([PROT_EMBEDDING_LOADER.load(prot_id) for prot_id in batch_prot_ids])

            preds = model.predict(batch_prot_embeddings.to(DEVICE))
            top_scores, top_indices = torch.topk(preds, 140)  # Get the top k scores along with their indices

            for prot_id, scores, indices in zip(batch_prot_ids, top_scores, top_indices):
                all_predictions[prot_id] = [(index_to_go_term[idx.item()], score.item()) for score, idx in zip(scores, indices)]

    model.to(prev_device)

    return all_predictions


def _get_model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


if __name__ == '__main__':
    main()
