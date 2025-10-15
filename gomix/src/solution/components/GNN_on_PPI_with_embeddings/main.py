import pickle
import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch_geometric.data import Data as GeometricData
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm
import random
import json
from typing import List
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[4]))
from src.solution.components.GNN_on_PPI_with_embeddings.ProteinGraphBuilder import ProteinGraphBuilder
from src.solution.components.GNN_on_PPI_with_embeddings.Net import Net
from src.utils.predictions_evaluation.evaluate import evaluate_with_deepgoplus_evaluator
from src.utils.ProteinEmbeddingLoader import ProteinEmbeddingLoader

TASK_DATASET_PATH = os.environ['TASK_DATASET_PATH']
assert TASK_DATASET_PATH, 'Environment variable \'TASK_DATASET_PATH\' must be declared.'

PPI_FILE_PATH = os.path.join(TASK_DATASET_PATH, 'all_proteins_STRING_interactions.json')
TRAIN_ANNOTATIONS_FILE_PATH = os.path.join(TASK_DATASET_PATH, 'propagated_annotations/train.json')
OFFICIAL_TEST_ANNOTS_FILE_PATH = os.path.join(TASK_DATASET_PATH, 'annotations/test.json')
GENE_ONTOLOGY_FILE_PATH = os.path.join(TASK_DATASET_PATH, 'go.obo')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))

PROT_EMBEDDING_LOADER = ProteinEmbeddingLoader()


def main():
    random.seed(0)
    torch.manual_seed(0)

    graph, graph_ctx = _build_or_load_whole_graph()
    print('Protein graph:', graph)
    print(f'It contains {graph.num_nodes} nodes (proteins). Average edges per node: {_get_average_degree(graph):.2f}. {_get_percentage_nodes_lte_k(graph, k=0):.2f}% have 0 edges.')

    model = make_and_train_model_on(graph=graph, graph_ctx=graph_ctx)
    print('Training finished. Let\'s now evaluate on test set (with the official criteria).')

    _evaluate_for_testing_with_official_criteria(model, graph, graph_ctx)


def _build_or_load_whole_graph():
    pickle_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../../data/temp_cache/GNN_on_PPI_with_embeddings/all_proteins_ppi_graph.pickle')
    if os.path.exists(pickle_file_path):
        print("Loading graph from pickle cache file.")
        with open(pickle_file_path, 'rb') as file:
            return pickle.load(file)

    with open(TRAIN_ANNOTATIONS_FILE_PATH, 'r') as f:
        train_annotations = json.load(f)

    result = build_whole_graph_from_scratch(train_annotations)

    with open(pickle_file_path, 'wb') as file:
        pickle.dump(result, file)

    return result


def build_whole_graph_from_scratch(train_annotations: dict) -> tuple:
    # Note: the nodes in PPI file are not only the proteins in train+val set but also those in test set.
    graph_builder = ProteinGraphBuilder(ppi_file_path=PPI_FILE_PATH, prot_embedding_loader=PROT_EMBEDDING_LOADER)
    graph_builder.set_targets(train_annotations)
    graph: GeometricData = graph_builder.build()
    graph.validate(raise_on_error=True)

    graph_ctx = {
        'prot_id_to_node_idx': graph_builder.prot_id_to_node_idx,
        'go_term_to_class_idx': graph_builder.go_term_to_class_idx,
    }
    return graph, graph_ctx


def make_and_train_model_on(graph: GeometricData, graph_ctx: dict) -> Net:
    train_mask, val_mask = _make_train_val_masks(graph, graph_ctx)
    train_loader = NeighborLoader(graph, num_neighbors=[4, 2], batch_size=10, input_nodes=train_mask)
    val_loader = NeighborLoader(graph, num_neighbors=[4, 2], batch_size=20, input_nodes=val_mask)

    print('Train-val split: {} - {}'.format(train_mask.sum(), val_mask.sum()))

    print(f'\nUsing device: {DEVICE}')
    model = make_model_on_device(graph_ctx)
    optimizer = optim.Adam(model.parameters(), lr=4e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.62)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    print('Training...')
    best_val_f_max = -np.inf
    best_epoch = 0
    MAX_EPOCHS = 13
    for epoch in range(1, MAX_EPOCHS+1):
        print(f"Epoch {epoch}: lr={optimizer.param_groups[0]['lr']}")
        model.train()

        pbar = tqdm(total=int(len(train_loader.dataset)))
        pbar.set_description(f'Epoch {epoch}')

        total_loss = total_examples = 0
        for batch in train_loader:
            batch.to(DEVICE)

            optimizer.zero_grad()
            targets = batch.y[:batch.batch_size]
            preds = model(batch)[:batch.batch_size]
            loss = loss_fn(preds, targets)
            loss.backward()
            optimizer.step()

            total_loss += float(loss) * batch.batch_size
            total_examples += batch.batch_size
            pbar.update(batch.batch_size)
        pbar.close()

        loss = total_loss / total_examples
        print(f'Epoch {epoch}, Loss: {loss:.4f}')

        val_loss, performances_by_threshold = _validate_model(model, val_loader, loss_fn)
        print(f'Epoch {epoch}, Validation Loss: {val_loss:.4f}')

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


def make_model_on_device(graph_ctx: dict) -> Net:
    return Net(
        prot_embedding_size=PROT_EMBEDDING_LOADER.get_embedding_size(),
        num_classes=len(graph_ctx['go_term_to_class_idx'])
    ).to(DEVICE)


@torch.no_grad()
def predict_and_transform_predictions_to_dict(model, prot_ids: List[str], graph: GeometricData, graph_ctx: dict) -> dict:
    node_idx_to_prot_id = {node_idx: prot_id for prot_id, node_idx in graph_ctx['prot_id_to_node_idx'].items()}
    class_idx_to_go_term = {class_idx: go_term for go_term, class_idx in graph_ctx['go_term_to_class_idx'].items()}

    mask = _make_graph_mask_with_prot_ids(prot_ids, graph, graph_ctx)
    if mask.sum() != len(prot_ids):
        print(f'Note: the test graph mask has {mask.sum()} nodes, while the number of test proteins was {len(prot_ids)}.')
    loader = NeighborLoader(graph, num_neighbors=[8, 4], batch_size=64, input_nodes=mask)

    prev_device = _get_model_device(model)
    model.to(DEVICE)
    model.eval()
    all_predictions = {prot_id: [] for prot_id in prot_ids}  # Filling default values for proteins with no predictions (i.e., proteins missing from the graph).
    for batch in loader:
        batch.to(DEVICE)
        preds = model.predict(batch)[:batch.batch_size]
        top_scores, top_indices = torch.topk(preds, 140)  # Get the top k scores along with their indices

        batch_node_indices = batch.n_id[:batch.batch_size]
        batch_prot_ids = [node_idx_to_prot_id[idx.item()] for idx in batch_node_indices]
        for prot_id, scores, class_indices in zip(batch_prot_ids, top_scores, top_indices):
            all_predictions[prot_id] = [(class_idx_to_go_term[class_idx.item()], score.item()) for score, class_idx in zip(scores, class_indices)]

    model.to(prev_device)

    return all_predictions


def _get_average_degree(data: GeometricData):
    num_edges = data.edge_index.shape[1] / 2
    num_nodes = data.x.shape[0]
    average_degree = num_edges / num_nodes
    return average_degree


def _get_percentage_nodes_lte_k(data: GeometricData, k: int):
    # Constructing a set of unique edges considering the graph as undirected
    edge_set = set(tuple(sorted(edge)) for edge in data.edge_index.t().tolist())

    # Counting the degree for each node
    degrees = torch.zeros(data.x.shape[0], dtype=torch.long)
    for edge in edge_set:
        degrees[edge[0]] += 1
        degrees[edge[1]] += 1

    num_nodes_lte_k = (degrees <= k).sum().item()
    perc_nodes_less_than_k = num_nodes_lte_k / len(degrees) * 100
    return perc_nodes_less_than_k


def _make_train_val_masks(graph: GeometricData, graph_ctx: dict):
    with open(TRAIN_ANNOTATIONS_FILE_PATH, 'r') as f:
        prot_ids = list(json.load(f).keys())

    random.shuffle(prot_ids)
    split_idx = int(len(prot_ids) * 0.8)
    train_prot_ids, val_prot_ids = prot_ids[:split_idx], prot_ids[split_idx:]

    train_mask = _make_graph_mask_with_prot_ids(train_prot_ids, graph, graph_ctx)
    val_mask = _make_graph_mask_with_prot_ids(val_prot_ids, graph, graph_ctx)

    return train_mask, val_mask


def _make_graph_mask_with_prot_ids(prot_ids: list, graph: GeometricData, graph_ctx: dict):
    prot_id_to_node_idx = graph_ctx['prot_id_to_node_idx']
    mask = torch.zeros(graph.num_nodes, dtype=torch.bool)
    mask[[prot_id_to_node_idx[prot_id] for prot_id in prot_ids if prot_id in prot_id_to_node_idx]] = True
    return mask


@torch.no_grad()
def _validate_model(model, val_loader, loss_fn: torch.nn.Module):
    pbar = tqdm(total=len(val_loader.dataset))
    pbar.set_description('Evaluating')

    model.to(DEVICE)
    model.eval()
    total_loss = total_examples = 0
    all_preds = []
    all_targets = []
    for batch in val_loader:
        batch.to(DEVICE)

        targets = batch.y[:batch.batch_size]
        preds = model(batch)[:batch.batch_size]
        loss = loss_fn(preds, targets)
        total_loss += float(loss) * batch.batch_size
        total_examples += batch.batch_size

        all_preds.append(torch.sigmoid(preds).cpu())
        all_targets.append(targets.cpu())

        pbar.update(batch.batch_size)
    pbar.close()

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

    loss = total_loss / total_examples
    return loss, performances_by_threshold


def _evaluate_for_testing_with_official_criteria(model: torch.nn.Module, graph: GeometricData, graph_ctx: dict):
    with open(OFFICIAL_TEST_ANNOTS_FILE_PATH, 'r') as f:
        test_annotations = json.load(f)
    test_prot_ids = list(test_annotations.keys())

    predictions = predict_and_transform_predictions_to_dict(model, test_prot_ids, graph, graph_ctx)

    print('Test results:')
    evaluate_with_deepgoplus_evaluator(
        gene_ontology_file_path=GENE_ONTOLOGY_FILE_PATH,
        predictions=predictions,
        ground_truth=test_annotations
    )


def _get_model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


if __name__ == '__main__':
    main()
