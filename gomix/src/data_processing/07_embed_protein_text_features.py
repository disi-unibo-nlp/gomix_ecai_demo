import sys
import argparse
from pathlib import Path
import pickle
from typing import List
from tqdm import tqdm
import os
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.calc_openai_embedding import calc_embedding as calc_openai_embedding
from src.utils.TfidfVectorizer import TfidfVectorizer

EMBED_METHOD = 'tfidf'  # 'openai' or 'tfidf'


"""
Example usage:

OPENAI_API_KEY=the_key \
python src/data_processing/07_embed_protein_text_features.py \
--in-dir data/processed/task_datasets/2016/all_protein_text_features \
--out-dir data/processed/task_datasets/2016/all_protein_text_features_OpenAI_embeddings

python src/data_processing/07_embed_protein_text_features.py \
--in-dir data/processed/task_datasets/2016/all_protein_text_features \
--out-dir data/processed/task_datasets/2016/all_protein_text_features_TFIDF_embeddings
"""
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-dir')
    parser.add_argument('--out-dir')
    args = parser.parse_args()

    Path(args.out_dir).mkdir(exist_ok=True)

    all_prot_ids = set(os.path.splitext(f)[0] for f in os.listdir(args.in_dir))
    already_embedded_prot_ids = set(os.path.splitext(f)[0] for f in os.listdir(args.out_dir))

    embed_method = init_embed_method(all_prot_ids=all_prot_ids, text_features_dir=args.in_dir)

    for prot_id, papers_string in tqdm(get_protein_text_features(all_prot_ids - already_embedded_prot_ids, args.in_dir)):
        embedding = embed_method(papers_string)
        if embedding:
            with open(os.path.join(args.out_dir, f"{prot_id}.pickle"), 'wb') as f:
                pickle.dump(embedding, f)

    print('Done.')


def get_protein_text_features(prot_ids: set, text_features_dir: str):
    for prot_id in prot_ids:
        with open(os.path.join(text_features_dir, f"{prot_id}.txt"), 'r') as f:
            papers_string = f.read()

        yield prot_id, papers_string


def init_embed_method(all_prot_ids: set, text_features_dir: str):
    if EMBED_METHOD == 'openai':
        assert os.environ.get('OPENAI_API_KEY'), f'You have to set OPENAI_API_KEY environment variable if you want to use OpenAI embedding method.'
        return embed_protein_text_features_with_openai
    elif EMBED_METHOD == 'tfidf':
        def _find_training_docs():
            for el in get_protein_text_features(prot_ids=all_prot_ids, text_features_dir=text_features_dir):
                yield el[1]
        tfidf_vectorizer = TfidfVectorizer(find_training_docs=_find_training_docs)
        return lambda text: embed_protein_text_features_with_tfidf(text=text, tfidf_vectorizer=tfidf_vectorizer)
    else:
        raise ValueError(f'Unknown embedding method: {EMBED_METHOD}')


def embed_protein_text_features_with_openai(text: str) -> List[float]:
    # Make sure the text is not too long for OpenAI API.
    while len(text) > 24000:
        substr = text[:text.rfind('\n\n')]
        text = substr if len(substr) < len(text) else ''

    text = text.strip()
    return calc_openai_embedding(text) if text else None


def embed_protein_text_features_with_tfidf(text: str, tfidf_vectorizer: TfidfVectorizer) -> List[float]:
    return tfidf_vectorizer.vectorize(text)


if __name__ == '__main__':
    main()
