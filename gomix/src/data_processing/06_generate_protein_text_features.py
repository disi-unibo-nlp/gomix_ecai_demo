import sys
import argparse
from pathlib import Path
import re
import random
from typing import List
from xml.etree import ElementTree
import requests
import os
import time
from tenacity import retry, stop_after_attempt, wait_random_exponential
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.parse_fasta_file import parse_fasta_file

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
UNIPROT_ENTRIES_FILE_PATH = os.path.join(THIS_DIR, '../../data/raw/Uniprot/uniprot_swiss_entries.dat')

FILTER_PAPERS_BEFORE_YEAR = 2016


"""
Example usage:

python src/data_processing/06_generate_protein_text_features.py \
--all-proteins-fasta-file data/processed/task_datasets/2016/all_proteins.fasta \
--out-dir data/processed/task_datasets/2016/all_protein_text_features
"""
def main():
    random.seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--all-proteins-fasta-file')
    parser.add_argument('--out-dir')
    args = parser.parse_args()

    all_prot_ids = set(r['seq_id'] for r in parse_fasta_file(args.all_proteins_fasta_file))
    uniprot_referenced_pmids = read_referenced_PMIDs_in_uniprot_entries()
    print('Uniprot entries loaded.')
    Path(args.out_dir).mkdir(exist_ok=True)

    already_done_prot_ids = set(os.path.splitext(f)[0] for f in os.listdir(args.out_dir))

    chunks = divide_list_in_chunks(list(all_prot_ids - already_done_prot_ids), chunk_size=100)
    for chunk_idx, prot_ids_batch in enumerate(chunks):
        batch_pmids = []
        for prot_id in prot_ids_batch:
            batch_pmids.extend(uniprot_referenced_pmids.get(prot_id, set()))
        batch_referenced_papers = resolve_pmids(batch_pmids)

        proteins_with_usable_papers_count = 0

        for prot_id in prot_ids_batch:
            pmids = uniprot_referenced_pmids.get(prot_id, set())
            papers = [batch_referenced_papers.get(str(pmid), None) for pmid in pmids]
            papers = [p for p in papers if p is not None and is_usable_paper(p)]
            if papers:
                papers_string = get_papers_str_representation(papers)
                with open(os.path.join(args.out_dir, f"{prot_id}.txt"), 'w') as f:
                    f.write(papers_string)
                proteins_with_usable_papers_count += 1

        print(f'Mini-batch {chunk_idx+1}/{len(chunks)} done. Stored papers for {proteins_with_usable_papers_count} proteins out of {len(prot_ids_batch)}')

    print('Done.')


def read_referenced_PMIDs_in_uniprot_entries() -> dict:
    results = {}
    with open(UNIPROT_ENTRIES_FILE_PATH, 'r') as f:
        current_prot_id = None
        for line in f:
            if line.startswith("ID   "):
                current_prot_id = line.split()[1]
                if current_prot_id not in results:
                    results[current_prot_id] = set()

            if line.startswith("RX   ") and current_prot_id is not None:
                regexp_result = re.search(r'PubMed=(\d+)', line)
                if regexp_result:
                    results[current_prot_id].add(regexp_result.group(1))

    return results


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def resolve_pmids(pmids: List[str]) -> dict:
    time.sleep(1)
    base = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
    db = 'pubmed'

    epost_url = f"{base}epost.fcgi"
    response = requests.post(epost_url, data={'db': db, 'id': ','.join(map(str, pmids))})
    content = response.text

    web_env = None
    query_key = None

    if "<WebEnv>" in content:
        web_env = content.split("<WebEnv>")[1].split("</WebEnv>")[0]
    if "<QueryKey>" in content:
        query_key = content.split("<QueryKey>")[1].split("</QueryKey>")[0]

    if not web_env or not query_key:
        return {}

    efetch_url = f"{base}efetch.fcgi"
    data = requests.post(efetch_url, data={'db': db, 'query_key': query_key, 'WebEnv': web_env, 'rettype': 'xml'}).text

    root = ElementTree.fromstring(data)
    articles = {}
    for article in root.findall(".//PubmedArticle"):
        pmid = article.find(".//PMID").text
        title = article.find(".//ArticleTitle").text

        abstract = article.find(".//AbstractText")
        abstract_text = abstract.text if abstract is not None else None

        pub_date = article.find(".//PubDate")
        year = int(pub_date.find("Year").text) if pub_date.find("Year") is not None else None

        articles[pmid] = {
            "title": title,
            "abstract": abstract_text,
            "publication_year": year
        }

    return articles


def is_usable_paper(p: dict) -> bool:
    title_is_ok = p['title'] is not None and len(p['title']) > 0
    abstract_is_ok = p['abstract'] is not None and len(p['abstract']) > 0
    pub_date_is_ok = p['publication_year'] is not None and int(p['publication_year']) < FILTER_PAPERS_BEFORE_YEAR
    return title_is_ok and abstract_is_ok and pub_date_is_ok


def get_papers_str_representation(papers: List[dict]) -> str:
    result = ''
    for p in papers:
        result += p['title'] + '\n' + p['abstract'] + '\n\n'
    return result


def divide_list_in_chunks(lst, chunk_size):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


if __name__ == '__main__':
    main()
