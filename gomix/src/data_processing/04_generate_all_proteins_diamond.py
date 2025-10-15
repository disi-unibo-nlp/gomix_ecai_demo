#!/usr/bin/env python
import argparse
import pandas as pd
import subprocess
import tempfile

"""
Example usage:

python src/data_processing/04_generate_all_proteins_diamond.py \
--all-proteins-fasta-file data/processed/task_datasets/2016/all_proteins.fasta \
--out-file data/processed/task_datasets/2016/all_proteins_diamond.res
"""
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all-proteins-fasta-file')
    parser.add_argument('--out-file')
    args = parser.parse_args()

    with tempfile.NamedTemporaryFile(suffix=".dmnd", delete=True) as blast_db_file:
        print('Generating DIAMOND db...')
        generate_diamond_db(args.all_proteins_fasta_file, blast_db_file.name)

        print('Running BLAST...')
        run_diamond_blastp(blast_db_file.name, args.all_proteins_fasta_file, args.out_file)

    print(f'Finished generating the BLAST results. Stored into {args.out_file}')
    print_results_summary(args.out_file)


def generate_diamond_db(fasta_file_path, out_file_path):
    subprocess.run(["diamond", "makedb", "--in", fasta_file_path, "-d", out_file_path])


def run_diamond_blastp(train_db_path, query_path, result_path):
    subprocess.run(["diamond", "blastp", "-d", train_db_path, "-q", query_path, "--outfmt", "6", "qseqid", "sseqid", "bitscore", "-o", result_path])


def print_results_summary(result_path):
    df = pd.read_csv(result_path, sep='\t', names=["qseqid", "sseqid", "bitscore"])
    print(f"Total lines in result file: {df.shape[0]}")
    print(f"Unique proteins in first column of result file: {df.qseqid.nunique()}")


if __name__ == '__main__':
    main()
