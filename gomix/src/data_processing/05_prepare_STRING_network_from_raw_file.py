import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.parse_fasta_file import parse_fasta_file

UNIPROT_AC = True  # If True, use UniProt ACs as protein IDs, otherwise use UniProt IDs

"""
Example usage:

python src/data_processing/05_prepare_STRING_network_from_raw_file.py \
--all-proteins-fasta-file data/processed/task_datasets/2016/all_proteins.fasta \
--raw-network-file data/raw/STRING/dataset_specific/2016dataset_filtered_protein.links.v11.0.txt \
--prot-id-to-string-accession-mapping-file data/raw/STRING/all_organisms.uniprot_2_string.2018.tsv \
--out-file data/processed/task_datasets/2016/all_proteins_STRING_interactions.json

The raw network file has to be produced starting from the STRING "protein.links" file,
by filtering it to only contain the proteins from the dataset of interest.
"""


def generate_string_accession_to_prot_id_mapping(reverse_mapping_file_path: str):
    mapping = dict()
    with open(reverse_mapping_file_path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            entries = line.strip().split('\t')
            prot_id = entries[1].split('|')[0 if UNIPROT_AC else 1]
            string_ac = entries[2]
            mapping[string_ac] = prot_id
    return mapping


def get_string_network(path_to_network: str, prot_ids_to_keep: set, string_ac_to_prot_id: dict) -> dict:
    results = defaultdict(dict)
    num_lines = count_lines_in_file(path_to_network)
    with open(path_to_network, 'r') as fp:
        for line in tqdm(fp, total=num_lines, unit='line', desc="Processing network file"):
            if line.startswith("protein1"):
                continue

            string_ac1, string_ac2, score = line.strip().split()
            if string_ac1 in string_ac_to_prot_id and string_ac2 in string_ac_to_prot_id:
                print(f'WARNING: Skipping line: {line}', file=sys.stderr)
                continue

            protein1, protein2 = string_ac_to_prot_id[string_ac1], string_ac_to_prot_id[string_ac2]
            if protein1 in prot_ids_to_keep and protein2 in prot_ids_to_keep:
                score = float(score) / 1000
                results[protein1][protein2] = results[protein2][protein1] = score

    return results


def count_lines_in_file(path):
    num_lines = 0
    with open(path, 'r') as fp:
        for _ in tqdm(fp, desc="Counting lines", unit="line"):
            num_lines += 1
            if '_' in globals():
                del _
    return num_lines


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--all-proteins-fasta-file')
    parser.add_argument('--raw-network-file')
    parser.add_argument('--prot-id-to-string-accession-mapping-file')
    parser.add_argument('--out-file')
    args = parser.parse_args()

    protein_ids_to_keep = set([r['seq_id'] for r in parse_fasta_file(args.all_proteins_fasta_file)])

    network = get_string_network(
        path_to_network=args.raw_network_file,
        prot_ids_to_keep=protein_ids_to_keep,
        string_ac_to_prot_id=generate_string_accession_to_prot_id_mapping(args.prot_id_to_string_accession_mapping_file)
    )

    with open(args.out_file, 'w') as f:
        json.dump(network, f)

    print(f"Number of proteins in the final adjacency matrix: {len(network.keys())}")
