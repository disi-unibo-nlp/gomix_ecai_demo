import os
import sys
from pathlib import Path
import argparse
import pandas as pd
import json
from glob import glob
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.predictions_evaluation.deepgoplus_evaluator.GeneOntology import GeneOntology


def process_tsv_file(input_file, gene_ontology):
    df = pd.read_csv(input_file, sep='\t', header=None, names=['protein_id', 'GO_term', 'aspect', 'ref'])
    df_grouped = df.groupby('protein_id')['GO_term'].agg(list).reset_index()
    df_grouped['GO_terms'] = df_grouped['GO_term'].apply(lambda terms: [term for term in terms if gene_ontology.has_term(term)])
    df_grouped = df_grouped[df_grouped['GO_terms'].apply(lambda terms: len(terms) > 0)]
    df_grouped.drop(columns=['GO_term'], inplace=True)  # Optional: remove the original 'GO_term' column after using it.
    return df_grouped


def process_pkl_file(input_file, gene_ontology):
    df = pd.read_pickle(input_file)
    df['annotations'] = df['annotations'].apply(lambda terms: [term for term in terms if gene_ontology.has_term(term)])
    df = df[df['annotations'].apply(lambda terms: len(terms) > 0)]
    df.rename(columns={'proteins': 'protein_id', 'annotations': 'GO_terms'}, inplace=True)
    return df[['protein_id', 'GO_terms']]


def main(input_dir, output_dir):
    gene_ontology_file_path = glob(os.path.join(input_dir, '*.obo'))[0]  # Assuming there is only one .obo file in the directory
    gene_ontology = GeneOntology(gene_ontology_file_path, with_rels=True)

    pkl_files = glob(os.path.join(input_dir, '*_data.pkl'))
    txt_files = glob(os.path.join(input_dir, '*_data.tsv'))

    for input_file in pkl_files + txt_files:
        if input_file.endswith('.pkl'):
            df = process_pkl_file(input_file, gene_ontology)
        elif input_file.endswith('.tsv'):
            df = process_tsv_file(input_file, gene_ontology)
        else:
            raise ValueError('Unsupported file format.')

        data_dict = dict(zip(df['protein_id'], df['GO_terms']))

        basename = os.path.basename(input_file)
        filename, _ = os.path.splitext(basename)
        filename = filename.replace('_data', '')  # remove '_data' from filename
        output_filename = filename + '.json'
        output_file_path = os.path.join(output_dir, output_filename)

        with open(output_file_path, 'w') as json_file:
            json.dump(data_dict, json_file)

        print(f'File saved at {output_file_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', help='Path to directory containing data files and .obo file')
    parser.add_argument('output_dir', help='Path to output directory')
    args = parser.parse_args()

    main(args.input_dir, args.output_dir)
