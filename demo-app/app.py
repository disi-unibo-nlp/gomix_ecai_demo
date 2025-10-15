import sys
from pathlib import Path
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import logging
import os
import json

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "demo-app"))

from solutionrunners.naive import predict as naive_predict
from solutionrunners.diamondscore import predict as diamondscore_predict
from solutionrunners.interactionscore import predict as interactionscore_predict
from solutionrunners.embeddingsimilarityscore import predict as embeddingsimilarityscore_predict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[logging.StreamHandler()])

APP_NAME = "GOMix"

METHODS_NAMES = [
    "Naive",
    "DiamondScore",
    "InteractionScore",
    "EmbeddingSimilarityScore",
    "FC on embeddings",
    "GNN on PPI & embeddings",
    "Stacked Ensemble",
]

TASK_DATASET_PATH = os.path.join(ROOT, "../gomix/src/data/processed/task_datasets/2016")
TEST_ANNOTATIONS_FILE_PATH = os.path.join(TASK_DATASET_PATH, 'annotations', 'test.json')


def predict(protein_id: str, method: str, topk: int):
    if method == 'Naive':
        predictions = naive_predict()
    elif method == "DiamondScore":
        predictions = diamondscore_predict(protein_id)
    elif method == "InteractionScore":
        predictions = interactionscore_predict(protein_id)
    elif method == "EmbeddingSimilarityScore":
        predictions = embeddingsimilarityscore_predict(protein_id)
    elif method in ("FC on embeddings", "GNN on PPI & embeddings", "Stacked Ensemble"):
        # Temporary placeholder. Need to implement.
        predictions = embeddingsimilarityscore_predict(protein_id)
        import random
        random.shuffle(predictions)
    else:
        raise 'Invalid method.'

    df = pd.DataFrame(predictions, columns=["GO ID", "Score"]).sort_values("Score", ascending=False)
    df_top = df.head(int(topk)).reset_index(drop=True)

    fig = plt.figure()
    plt.barh(df_top["GO ID"], df_top["Score"])
    plt.gca().invert_yaxis()
    plt.xlabel("Score")
    plt.title(f"Top-{topk} predictions")

    meta = f"Protein ID: {protein_id} | Method: {method}"
    return df_top, None, fig, meta


# --- START OF PRESENTATION ---

theme = gr.themes.Soft(
    primary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"],
)

custom_css = """
.gradio-container { max-width: 980px; margin: auto; }
:root, :host {
  --color-accent: #2563eb;                 /* primary blue */
  --color-accent-soft: rgba(37,99,235,.12);
  --body-background-fill: #f8fafc;         /* very light slate */
}
#intro {
  background: linear-gradient(180deg, var(--color-accent-soft), transparent 65%);
  border: 1px solid rgba(37,99,235,.18);
  border-radius: 16px;
  padding: 16px 20px;
}
h1, h2, h3 { letter-spacing: .2px; }
button.primary { box-shadow: 0 6px 18px rgba(37,99,235,.25); }
"""

INTRO_MD = f"""
# {APP_NAME}: Protein Function Prediction (Demo)

This is a demo UI for the paper **_Predicting Protein Functions with Ensemble Deep Learning and Protein Language Models_**.
Select a protein ID from a predefined set, choose a method, and get predictions.

- Sequence encoder in the paper: ESM2
- Methods included in the ensemble: Naive, DiamondScore, InteractionScore, EmbeddingSimilarityScore, FC on embeddings, GNN on PPI & embeddings, and the Stacked Ensemble.

**Note:** The following methods are implemented with actual components: Naive, DiamondScore, InteractionScore, and EmbeddingSimilarityScore. Other methods still use placeholder implementations. If the required data files are not found, the app will fall back to using placeholder implementations for all methods.
"""


def build_app():
    with open(TEST_ANNOTATIONS_FILE_PATH, 'r') as f:
        test_annotations = json.load(f)  # dict: prot ID -> list of GO terms
    test_prot_ids = [prot_id for prot_id in test_annotations]

    with gr.Blocks(title=f"{APP_NAME} — Protein Function Prediction", theme=theme, css=custom_css) as demo:
        gr.Markdown(INTRO_MD, elem_id="intro")

        with gr.Row():
            protein_sel = gr.Dropdown(
                choices=test_prot_ids,
                value=test_prot_ids[0],
                label="Protein ID",
            )
            with gr.Column():
                method = gr.Dropdown(
                    METHODS_NAMES,
                    value="Stacked Ensemble",
                    label="Method",
                )
                topk = gr.Slider(3, 10, value=5, step=1, label="Top-K")
                run_btn = gr.Button("Predict", variant="primary")
                clear_btn = gr.Button("Clear")

        meta = gr.Markdown("")
        out_df = gr.Dataframe(
            headers=["GO ID", "Score"],
            datatype=["str", "number"],
            label="Predictions",
            interactive=False,
            wrap=True,
            row_count=(0, "dynamic"),
        )
        out_plot = gr.Plot(label="Scores")

        out_truth = gr.Dataframe(
            headers=["GO ID"],
            datatype=["str"],
            label="True GO Terms",
            interactive=False,
            wrap=True,
            row_count=(0, "dynamic"),
        )

        def _predict(protein_id, method_sel, k):
            df_pred, info, fig, meta_txt = predict(protein_id, method_sel, k)
            truth_terms = test_annotations.get(protein_id, [])
            df_truth = pd.DataFrame({"GO ID": truth_terms})
            if info:
                gr.Info(info)
            return df_pred, fig, f"**{meta_txt}**", df_truth

        run_btn.click(
            _predict,
            inputs=[protein_sel, method, topk],
            outputs=[out_df, out_plot, meta, out_truth],
        )

        clear_btn.click(
            lambda: (test_prot_ids[0], "Stacked Ensemble", 5, "", pd.DataFrame(), plt.figure(), pd.DataFrame()),
            inputs=None,
            outputs=[protein_sel, method, topk, meta, out_df, out_plot, out_truth],
        )

        gr.Examples(
            examples=[
                [
                    test_prot_ids[0],
                    "Naive",
                    7,
                ],
            ],
            inputs=[protein_sel, method, topk],
            label="Examples",
        )

        gr.Markdown(
            """
**Disclaimer:** This UI is for demonstration purposes. Some methods are implemented with actual components:
- Naive: Uses frequency of GO terms in the training data
- DiamondScore: Uses BLAST to find similar proteins and their GO terms
- InteractionScore: Uses PPI network to find interacting proteins and their GO terms
- EmbeddingSimilarityScore: Uses cosine similarity between protein sequence embeddings

Other methods (FC on embeddings, GNN on PPI & embeddings, Stacked Ensemble) still use placeholder implementations.
"""
        )
    return demo


if __name__ == "__main__":
    build_app().launch()
