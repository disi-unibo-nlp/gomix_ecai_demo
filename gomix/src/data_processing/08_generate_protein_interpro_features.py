import xml.etree.ElementTree as ET
import argparse
from scipy.sparse import coo_matrix
import pickle
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.parse_fasta_file import parse_fasta_file

"""
Example usage:

python src/data_processing/08_generate_protein_interpro_features.py \
data/raw/InterPro/58.0__match_complete.xml \
data/processed/task_datasets/2016/all_proteins.fasta \
data/processed/task_datasets/2016/all_protein_interpro_features.pickle
"""


def main(interpro_matches_xml_file, fasta_file, output_file):
    prot_ids_from_fasta = set([r['seq_id'] for r in parse_fasta_file(fasta_file)])
    row_indices = []
    col_indices = []
    ipr_to_col_index = {}
    next_col_index = 0
    ordered_prot_ids = []

    for protein_id, iprs in _parse_interpro_xml(interpro_matches_xml_file):
        if protein_id not in prot_ids_from_fasta:
            continue

        ordered_prot_ids.append(protein_id)
        row_index = len(ordered_prot_ids) - 1

        for ipr in iprs:
            if ipr not in ipr_to_col_index:
                ipr_to_col_index[ipr] = next_col_index  # Assign a column number to this InterPro ID
                next_col_index += 1

            col_index = ipr_to_col_index[ipr]
            row_indices.append(row_index)
            col_indices.append(col_index)

        if len(ordered_prot_ids) % 100 == 0:
            print(f"Number of proteins matched so far: {len(ordered_prot_ids)}")

    # Create coo_matrix using an expression for the data_values
    coo = coo_matrix(([1] * len(row_indices), (row_indices, col_indices)), shape=(len(ordered_prot_ids), next_col_index))
    with open(output_file, 'wb') as f_out:
        pickle.dump({'coo_matrix': coo, 'ordered_protein_ids': ordered_prot_ids}, f_out)

    print(f"\nDone. Features stored for a total of {len(ordered_prot_ids)} proteins. COO matrix shape: {coo.shape}")


def _parse_interpro_xml(interpro_xml_file):
    context = ET.iterparse(interpro_xml_file)
    for event, elem in context:
        if event == "end" and elem.tag == 'protein':
            protein_id = elem.attrib['name']
            iprs = [ipr.attrib['id'] for match in elem.findall('match') for ipr in match.findall('ipr')]
            yield protein_id, iprs
            elem.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('interpro_matches_xml_file')
    parser.add_argument('fasta_file')
    parser.add_argument('output_file')

    args = parser.parse_args()
    main(
        interpro_matches_xml_file=args.interpro_matches_xml_file,
        fasta_file=args.fasta_file,
        output_file=args.output_file
    )
