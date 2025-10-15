import math
import numpy as np
import json
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[0]))
from GeneOntology import GeneOntology

TASK_DATASET_PATH = os.environ["TASK_DATASET_PATH"]
assert TASK_DATASET_PATH, 'Environment variable \'TASK_DATASET_PATH\' must be declared.'

# TODO: It may be necessary to use propagated annotations (also for test set). Double-check by seeing how the S_min mesaure is computed.
PROPAGATED_ANNOTATIONS_DIR = os.path.join(TASK_DATASET_PATH, 'propagated_annotations')

# Code adapted from https://github.com/bio-ontology-research-group/deepgoplus

ONT_ROOTS = {
    'MFO': 'GO:0003674',
    'BPO': 'GO:0008150',
    'CCO': 'GO:0005575'
}
NAMESPACES = {
    'MFO': 'molecular_function',
    'BPO': 'biological_process',
    'CCO': 'cellular_component'
}


def evaluate(
    gene_ontology_file_path: str,
    predictions: dict,  # dict: Prot ID -> list of (GO term, score)
    ground_truth: dict,  # dict: Prot ID -> list of GO terms
):
    go_rels = GeneOntology(gene_ontology_file_path, with_rels=True)
    _calculate_ontology_ic(go_rels)
    for ont in NAMESPACES.keys():
        go_set = go_rels.get_namespace_terms(NAMESPACES[ont])
        go_set.remove(ONT_ROOTS[ont])

        fmax = 0.0
        optimal_f_threshold = 0.0
        smin = 1000.0
        precisions = []
        recalls = []
        for t in range(101):
            threshold = t / 100.0

            all_gt_labels = []
            all_preds = []
            for prot_id, gt_labels in ground_truth.items():
                gt_labels = set([term for term in gt_labels if term in go_set])
                all_gt_labels.append(gt_labels)

                preds = set([term for term, score in predictions[prot_id] if score >= threshold])
                for go_term in preds.copy():
                    preds |= go_rels.get_ancestors(go_term)
                preds &= go_set  # Very important: it removes all terms that are not in the ontology we're considering.
                all_preds.append(preds)

            fscore, prec, rec, s = _evaluate_annots(go=go_rels, real_annots=all_gt_labels, pred_annots=all_preds)
            precisions.append(prec)
            recalls.append(rec)
            if fmax < fscore:
                fmax = fscore
                optimal_f_threshold = threshold
            if smin > s:
                smin = s

        # Compute AUPR
        precisions = np.array(precisions)
        recalls = np.array(recalls)
        sorted_index = np.argsort(recalls)
        recalls = recalls[sorted_index]
        precisions = precisions[sorted_index]
        aupr = np.trapz(precisions, recalls)

        print('\n' + ont)
        print(f'F_max: {fmax:.3f} (optimal threshold={optimal_f_threshold:.2f})')
        print(f'S_min: {smin:.3f}')
        print(f'AUPR: {aupr:.3f}')


def _calculate_ontology_ic(go_rels):
    annotations = []
    for file_name in ['train.json', 'test.json']:
        with open(os.path.join(PROPAGATED_ANNOTATIONS_DIR, file_name)) as f:
            annotations_subgroup = json.load(f).values()  # list of lists of GO terms
            annotations_subgroup = [set(x) for x in annotations_subgroup]
            annotations.extend(annotations_subgroup)
    go_rels.calculate_ic(annotations)  # Pass list of sets of GO terms


def _evaluate_annots(go, real_annots, pred_annots):
    total = 0
    p = 0.0
    r = 0.0
    p_total = 0
    ru = 0.0
    mi = 0.0
    for i in range(len(real_annots)):
        if len(real_annots[i]) == 0:
            continue
        tp = real_annots[i].intersection(pred_annots[i])
        fp = pred_annots[i] - tp
        fn = real_annots[i] - tp
        for go_id in fp:
            mi += go.get_ic(go_id)
        for go_id in fn:
            ru += go.get_ic(go_id)
        tpn = len(tp)
        fpn = len(fp)
        fnn = len(fn)
        total += 1
        recall = tpn / (1.0 * (tpn + fnn))
        r += recall
        if len(pred_annots[i]) > 0:
            p_total += 1
            precision = tpn / (1.0 * (tpn + fpn))
            p += precision
    ru /= total
    mi /= total
    r /= total
    if p_total > 0:
        p /= p_total
    f = 0.0
    if p + r > 0:
        f = 2 * p * r / (p + r)
    s = math.sqrt(ru * ru + mi * mi)
    return f, p, r, s
