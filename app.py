# gomix_app.py
"""
GOMix — Gradio demo for protein function prediction (placeholders).
Run: pip install gradio pandas numpy matplotlib
Then: python gomix_app.py
"""

import hashlib
import io
import re
from typing import Dict, List, Tuple

import gradio as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

APP_NAME = "GOMix"

AA_VALID = set("ACDEFGHIKLMNPQRSTVWYBXZJUO")  # permissive
METHODS = [
    "Naive",
    "DiamondScore",
    "InteractionScore",
    "EmbeddingSimilarityScore",
    "FC on embeddings",
    "GNN on PPI & embeddings",
    "Stacked Ensemble",
]
ONTOLOGIES: Dict[str, List[Tuple[str, str]]] = {
    "MFO": [
        ("GO:0003674", "molecular_function"),
        ("GO:0003824", "catalytic activity"),
        ("GO:0005515", "protein binding"),
        ("GO:0000166", "nucleotide binding"),
        ("GO:0004871", "signal transducer activity"),
        ("GO:0016491", "oxidoreductase activity"),
        ("GO:0003700", "DNA-binding TF activity"),
        ("GO:0005215", "transporter activity"),
        ("GO:0016787", "hydrolase activity"),
        ("GO:0005524", "ATP binding"),
    ],
    "BPO": [
        ("GO:0008150", "biological_process"),
        ("GO:0009987", "cellular process"),
        ("GO:0008152", "metabolic process"),
        ("GO:0007165", "signal transduction"),
        ("GO:0006355", "regulation of transcription, DNA-templated"),
        ("GO:0006950", "response to stress"),
        ("GO:0006412", "translation"),
        ("GO:0055114", "oxidation-reduction process"),
        ("GO:0005975", "carbohydrate metabolic process"),
        ("GO:0006468", "protein phosphorylation"),
    ],
    "CCO": [
        ("GO:0005575", "cellular_component"),
        ("GO:0005737", "cytoplasm"),
        ("GO:0005634", "nucleus"),
        ("GO:0005886", "plasma membrane"),
        ("GO:0005739", "mitochondrion"),
        ("GO:0005829", "cytosol"),
        ("GO:0005794", "Golgi apparatus"),
        ("GO:0005783", "endoplasmic reticulum"),
        ("GO:0005615", "extracellular space"),
        ("GO:0005840", "ribosome"),
    ],
}

METHOD_SEED_OFFSET = {
    "Naive": 11,
    "DiamondScore": 23,
    "InteractionScore": 37,
    "EmbeddingSimilarityScore": 41,
    "FC on embeddings": 53,
    "GNN on PPI & embeddings": 67,
}

FASTA_RE = re.compile(r"^>.*$", re.MULTILINE)


def clean_sequence(text: str) -> str:
    if not text:
        return ""
    seq = FASTA_RE.sub("", text).replace("\n", "").replace("\r", "").replace(" ", "")
    return seq.upper()


def validate_sequence(seq: str) -> Tuple[bool, str]:
    if not seq:
        return False, "Empty sequence."
    invalid = sorted(set(ch for ch in seq if ch not in AA_VALID))
    if invalid:
        return (
            False,
            f"Invalid characters: {''.join(invalid)}. Allowed: {''.join(sorted(AA_VALID))}.",
        )
    if len(seq) < 20:
        return False, "Sequence is very short (<20 aa)."
    return True, ""


def seq_method_seed(seq: str, method: str, ontology: str, offset: int = 0) -> int:
    h = hashlib.sha256(f"{seq}|{method}|{ontology}".encode()).hexdigest()[:8]
    base = int(h, 16)
    return (base + offset) & 0xFFFFFFFF


def score_with_seed(seq: str, method: str, ontology: str) -> np.ndarray:
    if method == "Stacked Ensemble":
        parts = [m for m in METHODS if m != "Stacked Ensemble"]
        mats = []
        for m in parts:
            seed = seq_method_seed(seq, m, ontology, METHOD_SEED_OFFSET[m])
            rng = np.random.default_rng(seed)
            raw = rng.random(len(ONTOLOGIES[ontology]))
            mats.append(raw)
        arr = np.vstack(mats).mean(axis=0)
        return arr
    else:
        seed = seq_method_seed(seq, method, ontology, METHOD_SEED_OFFSET[method])
        rng = np.random.default_rng(seed)
        return rng.random(len(ONTOLOGIES[ontology]))


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    s = scores - scores.min()
    if s.max() > 0:
        s = s / s.max()
    return s


def predict(seq_text: str, method: str, ontology: str, topk: int):
    seq = clean_sequence(seq_text)
    ok, msg = validate_sequence(seq)
    if not ok:
        return (
            pd.DataFrame(columns=["GO ID", "Name", "Score"]),
            gr.Info(msg),
            None,
            "",
        )

    go_list = ONTOLOGIES[ontology]
    raw = score_with_seed(seq, method, ontology)
    scores = normalize_scores(raw)

    df = pd.DataFrame(
        {
            "GO ID": [go for go, _ in go_list],
            "Name": [name for _, name in go_list],
            "Score": scores,
        }
    ).sort_values("Score", ascending=False)

    df_top = df.head(int(topk)).reset_index(drop=True)

    fig = plt.figure()
    plt.barh(df_top["GO ID"], df_top["Score"])
    plt.gca().invert_yaxis()
    plt.xlabel("Score")
    plt.title(f"Top-{topk} predictions ({ontology})")

    meta = f"Sequence length: {len(seq)} | Method: {method} | Ontology: {ontology}"
    return df_top, None, fig, meta


INTRO_MD = f"""
# {APP_NAME}: Protein Function Prediction (Demo)

This is a demo UI for the paper **_Predicting Protein Functions with Ensemble Deep Learning and Protein Language Models_**.
Paste an amino acid sequence (FASTA or raw), choose a method, choose an ontology (MFO/BPO/CCO), and get mock predictions.
Methods are **placeholders**; plug in your real models later.

- Sequence encoder in the paper: ESM2
- Methods included in the ensemble: Naive, DiamondScore, InteractionScore, EmbeddingSimilarityScore, FC on embeddings, GNN on PPI & embeddings, and the Stacked Ensemble.

**Note:** Outputs here are synthetic and deterministic. They are not biological predictions.
"""


def build_app():
    with gr.Blocks(title=f"{APP_NAME} — Protein Function Prediction") as demo:
        gr.Markdown(INTRO_MD)

        with gr.Row():
            seq = gr.Textbox(
                label="Protein sequence (AA; FASTA or raw)",
                placeholder=">sp|P01009|...\nMKWVTFISLLFLFSSAYSRGVFRRDTHKSEIAHRFKDLGE...",
                lines=8,
                autofocus=True,
            )
            with gr.Column():
                method = gr.Dropdown(
                    METHODS,
                    value="Stacked Ensemble",
                    label="Method",
                )
                ontology = gr.Radio(
                    list(ONTOLOGIES.keys()), value="MFO", label="Ontology"
                )
                topk = gr.Slider(3, 10, value=5, step=1, label="Top-K")
                run_btn = gr.Button("Predict", variant="primary")
                clear_btn = gr.Button("Clear")

        meta = gr.Markdown("")
        out_df = gr.Dataframe(
            headers=["GO ID", "Name", "Score"],
            datatype=["str", "str", "number"],
            label="Predictions",
            interactive=False,
            wrap=True,
            row_count=(0, "dynamic"),
        )
        out_plot = gr.Plot(label="Scores")

        def _predict(seq_text, method_sel, ont_sel, k):
            df, info, fig, meta_txt = predict(seq_text, method_sel, ont_sel, k)
            if info:
                gr.Info(info)
            return df, fig, f"**{meta_txt}**"

        run_btn.click(
            _predict,
            inputs=[seq, method, ontology, topk],
            outputs=[out_df, out_plot, meta],
        )

        clear_btn.click(
            lambda: ("", "Stacked Ensemble", "MFO", 5, "", pd.DataFrame(), plt.figure()),
            inputs=None,
            outputs=[seq, method, ontology, topk, meta, out_df, out_plot],
        )

        gr.Examples(
            examples=[
                [
                    ">example\nMKWVTFISLLFLFSSAYS",
                    "Naive",
                    "MFO",
                    5,
                ],
                [
                    "MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFPDWQ",
                    "EmbeddingSimilarityScore",
                    "BPO",
                    5,
                ],
                [
                    "MDYKDDDDKMGSSHHHHHHSSGLVPRGSHMASMTGGQQMGRGSEF",
                    "Stacked Ensemble",
                    "CCO",
                    7,
                ],
            ],
            inputs=[seq, method, ontology, topk],
            label="Examples",
        )

        gr.Markdown(
            """
**Disclaimer:** This UI is for demonstration only. Replace scoring stubs with your actual components:
- Plug Diamond results, PPI-derived scores, and KNN on embeddings into the per-method branches.
- The ensemble can combine calibrated outputs or meta-learner logits from your stacker.
"""
        )
    return demo


if __name__ == "__main__":
    build_app().launch()
