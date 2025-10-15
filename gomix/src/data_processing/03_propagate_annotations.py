import sys
from pathlib import Path
import argparse
import json
import glob
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.predictions_evaluation.deepgoplus_evaluator.GeneOntology import GeneOntology
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('annotations_dir', help="Directory containing annotation files")
    parser.add_argument("output_dir", help="Path to output directory")
    parser.add_argument("gene_ontology_file_path", help="Path to gene ontology file")
    args = parser.parse_args()

    gene_ontology = GeneOntology(args.gene_ontology_file_path, with_rels=True)

    # Loop over each JSON file in the directory
    for annotations_file_path in glob.glob(os.path.join(args.annotations_dir, "*.json")):
        with open(annotations_file_path, 'r') as json_file:
            prot_to_go_terms = json.load(json_file)

        for prot, go_terms in prot_to_go_terms.items():
            new_go_terms = set()
            for go_term in go_terms:
                new_go_terms.add(go_term)
                new_go_terms = new_go_terms.union(gene_ontology.get_ancestors(go_term))
            prot_to_go_terms[prot] = list(new_go_terms)

        output_file_path = os.path.join(args.output_dir, os.path.basename(annotations_file_path))
        with open(output_file_path, 'w') as f:
            json.dump(prot_to_go_terms, f)

        print(f"Processed file {annotations_file_path} and wrote the result to {output_file_path}")

    print("All files have been processed.")


if __name__ == '__main__':
    main()
