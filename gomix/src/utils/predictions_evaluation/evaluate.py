import os
import subprocess
import csv
import shutil
from src.utils.predictions_evaluation.deepgoplus_evaluator.evaluate import evaluate as _evaluate_with_deepgoplus_method

THIS_DIR = os.path.dirname(os.path.realpath(__file__))


def evaluate_with_deepgoplus_evaluator(
    gene_ontology_file_path: str,
    predictions: dict,  # dict: Prot ID -> list of (GO term, score)
    ground_truth: dict,  # dict: Prot ID -> list of GO terms
):
    _evaluate_with_deepgoplus_method(gene_ontology_file_path, predictions, ground_truth)


# It generates an `evaluation_results` sub-folder in dest_dir_path.
def evaluate_with_CAFA_evaluator(
    dest_dir_path: str,
    gene_ontology_file_path: str,
    predictions: dict,  # dict: Prot ID -> list of (GO term, score)
    ground_truth: dict,  # dict: Prot ID -> list of GO terms
):
    # Prepare temporary predictions file.
    tmp_predictions_dir = os.path.join(dest_dir_path, 'tmp_predictions')
    os.makedirs(tmp_predictions_dir)
    tmp_predictions_file_path = os.path.join(tmp_predictions_dir, 'tmp_predictions.tsv')
    with open(tmp_predictions_file_path, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        for prot_id, preds in predictions.items():
            for go_term, score in preds:
                writer.writerow((prot_id, go_term, score))

    # Prepare temporary ground truth file.
    tmp_ground_truth_file_path = os.path.join(dest_dir_path, 'tmp_eval_ground_truth.tsv')
    with open(tmp_ground_truth_file_path, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        for prot_id, go_terms in ground_truth.items():
            for go_term in go_terms:
                writer.writerow((prot_id, go_term))

    subprocess.run([
        'python',
        os.path.join(THIS_DIR, 'CAFA_evaluator/src/main.py'),
        gene_ontology_file_path,
        tmp_predictions_dir,
        tmp_ground_truth_file_path,
        '-out_dir',
        os.path.join(dest_dir_path, 'evaluation_results'),
    ])

    # Clean up.
    shutil.rmtree(tmp_predictions_dir)
    os.remove(tmp_ground_truth_file_path)
