import os
import pandas as pd
import argparse


def process_pickle_files(input_dir, output_dir):
    all_prot_id_to_seq = {}
    for file in os.listdir(input_dir):
        if file.endswith("_data.pkl"):
            df = pd.read_pickle(os.path.join(input_dir, file))
            for idx, row in df.iterrows():
                protein_id = row['proteins']
                sequence = row['sequences']
                all_prot_id_to_seq[protein_id] = sequence

    with open(os.path.join(output_dir, 'all_proteins.fasta'), 'w') as fasta_file:
        for protein_id, sequence in all_prot_id_to_seq.items():
            fasta_file.write(f">{protein_id}\n{sequence}\n")


def main():
    parser = argparse.ArgumentParser(description='Process .pkl files and generate a fasta file.')
    parser.add_argument('--input_dir', type=str, help='The directory containing the .pkl files.')
    parser.add_argument('--output_dir', type=str, help='The directory where the fasta file will be stored.')

    args = parser.parse_args()

    process_pickle_files(args.input_dir, args.output_dir)


if __name__ == '__main__':
    main()
