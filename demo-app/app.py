import sys
from pathlib import Path
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import logging
import os
import json
import re

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

# TASK_DATASET_PATH = os.path.join(ROOT, "../gomix/src/data/processed/task_datasets/2016")
# TEST_ANNOTATIONS_FILE_PATH = os.path.join(TASK_DATASET_PATH, 'annotations', 'test.json')


DEMO_UTILS = os.path.join(ROOT, "../gomix/src/demo_utils")
TEST_ANNOTATIONS_FILE_PATH = os.path.join(DEMO_UTILS, 'annotations', 'test.json')
TEST_ANNOTATIONS_INFO_FILE_PATH = os.path.join(DEMO_UTILS, 'test_uniprotid_info_formatted.txt')
PROTEIN_IMGS_PATH = os.path.join(DEMO_UTILS, 'imgs') 



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

    df = pd.DataFrame(predictions, columns=["GO_ID", "Score"]).sort_values("Score", ascending=False)
    df_top = df.head(int(topk)).reset_index(drop=True)
    
    # Batch fetch names and descriptions for all GO codes
    go_codes = df_top["GO_ID"].tolist()
    names_map, descriptions_map = find_multiple_gocode_descriptions(go_codes)
    df_top["Name"] = df_top["GO_ID"].map(names_map)
    df_top["Description"] = df_top["GO_ID"].map(descriptions_map)

    meta = f"Protein ID: {protein_id} | Method: {method}"
    return df_top, None, meta


# --- START OF PRESENTATION ---

theme = gr.themes.Soft(
    primary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"],
)

custom_css = """
.gradio-container { max-width: 1100px; margin: auto; }
:root, :host {
  --color-accent: #2563eb;                 /* primary blue */
  --color-accent-soft: rgba(37,99,235,.12);
  --body-background-fill: #f8fafc;         /* very light slate */
}
#intro {
  background: linear-gradient(180deg, var(--color-accent-soft), transparent 65%);
  border: 1px solid rgba(37,99,235,.18);
  border-radius: 16px;
  padding: 20px 24px;
  margin-bottom: 16px;
}
#intro .image-container img {
  border-radius: 12px;
  object-fit: contain;
}
h1, h2, h3 { letter-spacing: .2px; }
button.primary { box-shadow: 0 6px 18px rgba(37,99,235,.25); }

/* Protein info table - adapt to light/dark mode */
.protein-info-table {
  color: var(--body-text-color);
}
.dark .protein-info-table {
  color: white;
}
.light .protein-info-table, body:not(.dark) .protein-info-table {
  color: black;
}

/* Protein description - adapt to light/dark mode */
.protein-description {
  color: var(--body-text-color);
}
.dark .protein-description {
  color: white;
}
.light .protein-description, body:not(.dark) .protein-description {
  color: black;
}
"""

LOGO_PATH = os.path.join(DEMO_UTILS, 'imgs', 'gomix_icon.png')

def create_intro_section():
    """
    Creates the intro section with logo, title, and tabbed content.
    """
    with gr.Row(elem_id="intro"):
        # Logo column
        with gr.Column(scale=1, min_width=60):
            gr.Image(
                value=LOGO_PATH,
                show_label=False,
                show_download_button=False,
                container=False,
                height=60,
                width=60
            )

        # Title and content column
        with gr.Column(scale=9):
            gr.Markdown(f"# {APP_NAME}: Protein Function Prediction")

            with gr.Tabs():
                with gr.Tab("Context"):
                    gr.Markdown("""
**Understanding proteins is key to understanding life itself.**

Proteins are essential molecules that perform a wide range of functions in all living organisms. They are large molecules composed of 20 types of building blocks called **amino acids**, which are linked sequentially to form the **protein sequence**. This sequence determines the protein's 3D structure, dynamics, and ultimately its biological function.

With every breath, meal, and workout, your body uses complex protein interactions. Nearly every biological process—like transporting oxygen, building muscle, and fighting disease—relies on proteins. However, we still don't fully understand what many proteins do.

**The Challenge:** Due to ongoing genome sequencing projects, we have vast amounts of sequence data from thousands of species. Yet accurately assigning biological function to these proteins remains difficult because:
- Many proteins have multiple functions
- Proteins interact with multiple partners
- Experimental validation is time-consuming and expensive

**This Demo:** We predict **GO (Gene Ontology) terms** for proteins based on their amino acid sequences. GO terms describe:
- What the protein does (molecular function)
- Which biological processes it participates in
- Where in the cell it operates (cellular component)

A well-performing model can help scientists narrow down protein roles faster, prioritize lab experiments, and spot connections that might otherwise go unnoticed.
                    """)

                with gr.Tab("How does GOMix work?"):
                    gr.Markdown("""
**GOMix** is an open-source ensemble framework that combines multiple prediction strategies to achieve robust protein function prediction.

**Key Components:**

1. **Protein Encoding:** Uses **ESM2**, a state-of-the-art transformer-based protein language model, to convert amino acid sequences into rich numerical representations (embeddings).

2. **Ensemble of Six Methods:**
   - **Naive:** Baseline using frequency of GO terms in training data
   - **DiamondScore:** Sequence alignment-based (BLAST-like) similarity
   - **InteractionScore:** Leverages protein-protein interaction (PPI) networks
   - **EmbeddingSimilarityScore:** Cosine similarity between protein embeddings
   - **FC on embeddings:** Feedforward neural network classifier on embeddings
   - **GNN on PPI & embeddings:** Graph neural network combining PPI structure and embeddings

3. **Stacked Ensemble:** Combines predictions from all six methods to produce final, more reliable predictions.

This multi-strategy approach captures different aspects of protein function—from sequence similarity to network context to deep learned patterns—making predictions more robust and accurate than any single method alone.
                    """)

                with gr.Tab("Evaluation & Performance"):
                    gr.Markdown("""
**Dataset:** GOMix was evaluated on the 2016 CAFA-3 compliant dataset:
- **Training:** Proteins with experimental annotations before January 2016
- **Test:** Proteins with annotations obtained between January-October 2016

**Evaluation Metrics:**
- **F_max:** Maximum F1 score achievable (balances precision and recall)
- **S_min:** Minimum semantic distance between predicted and actual GO labels
- **AUPR:** Area under the precision-recall curve

**Key Considerations:**
- All annotations follow the **True Path Rule**: if a GO term is annotated, all its ancestor terms are also annotated
- Evaluation sweeps confidence thresholds from 0 to 1 to find optimal operating points
- **Open-world assumption:** Some predictions may be correct but not yet experimentally validated

**Note:** In this demo, green circles (🟢) indicate correct predictions (matching ground truth), while red circles (🔴) indicate predictions not in the current ground truth. Remember that red predictions aren't necessarily "wrong"—they might be valid but not yet experimentally confirmed.
                    """)

                with gr.Tab("About"):
                    gr.Markdown("""
This is a demo application for the paper **_"Predicting Protein Functions with Ensemble Deep Learning and Protein Language Models"_**.

**Purpose:** This interface demonstrates how ensemble learning can be applied to automated Protein Function Prediction (PFP), bridging traditional bioinformatics approaches with modern deep learning techniques.

**Implementation Status:**
- Fully implemented methods: Naive, DiamondScore, InteractionScore, EmbeddingSimilarityScore
- Other methods (FC, GNN, Stacked Ensemble) use placeholder implementations in this demo

**Learn More:**
- Select a protein from the dropdown to view its metadata and structure
- Choose a prediction method and adjust the number of top predictions (Top-K)
- Compare predictions with ground truth annotations
- Explore how different methods perform on various proteins

Understanding protein function is a small step for AI, but a giant leap for medicine and biology!
                    """)

#**Note:** The following methods are implemented with actual components: Naive, DiamondScore, InteractionScore, and EmbeddingSimilarityScore. Other methods still use placeholder implementations. If the required data files are not found, the app will fall back to using placeholder implementations for all methods.


TOOLTIPS = {
    "GeneName": "This subsection indicates the name(s) of the gene(s) that code for the protein sequence(s) described in the entry.",
    "Organism": "This subsection provides information on the name(s) of the organism that is the source of the protein sequence.",
    "ProteinExistence": "This indicates the type of evidence that supports the existence of the protein. Note that the 'protein existence' evidence does not give information on the accuracy or correctness of the sequence(s) displayed (e.g., experimental, predicted).",
    "AminoAcids": "This subsection provides the number of amino acids of the given UniProt ID (length of the protein sequence).",
    "AnnotationScore": "The annotation score provides a heuristic measure of the annotation content of a UniProtKB entry or proteome. This score cannot be used as a measure of the accuracy of the annotation as we cannot define the 'correct annotation' for any given protein.",
}

TABLE_MARKDOWN = f"""
<div align="center" class="protein-info-table">

  <h2><strong>{{{{Protein Name}}}}</strong></h2>
  <h3>
    <label>Gene<sup style="font-size:0.6em;">
      <span title="{TOOLTIPS['GeneName']}"> ⓘ</span>
    </sup></label>: {{{{Gene Name}}}}
  </h3>

  <br/>

  <table border="0" style="margin:auto; text-align:center;">
    <tr>
      <td>
        <div style="text-align:center;">
          <label><strong>Organism:</strong> <sup style="font-size:0.6em;">
            <span title="{TOOLTIPS['Organism']}">ⓘ</span>
          </sup></label>
        </div>
        <br/>{{{{Organism}}}}
      </td>
      <td>
        <div style="text-align:center;">
          <label><strong>Protein existence:</strong> <sup style="font-size:0.6em;">
            <span title="{TOOLTIPS['ProteinExistence']}">ⓘ</span>
          </sup></label>
        </div>
        <br/>{{{{ProteinExistence}}}}
      </td>
    </tr>
    <tr>
      <td>
        <div style="text-align:center;">
          <label><strong>Amino acids:</strong> <sup style="font-size:0.6em;">
            <span title="{TOOLTIPS['AminoAcids']}">ⓘ</span>
          </sup></label>
        </div>
        <br/>{{{{AminoAcids}}}}
      </td>
      <td>
        <div style="text-align:center;">
          <label><strong>Annotation score:</strong> <sup style="font-size:0.6em;">
            <span title="{TOOLTIPS['AnnotationScore']}">ⓘ</span>
          </sup></label>
        </div>
        <br/>{{{{AnnotationScore}}}}
      </td>
    </tr>
  </table>
</div>
"""


DESCRIPTION_MARKDOWN = """
<b>Description:</b><br/>
<div class="protein-description" style="
    height:95px;
    overflow:auto;
    border-radius:8px;
    border: none;
    padding:12px;
    background-color: transparent;
    font-size:0.95em;
">
{{Description}}
</div>
"""

######################
# NEW CUSTOM METHODS #
######################

def render_template(template: str, values: dict) -> str:
    """
    Replace all {{Placeholders}} in the template with corresponding values from the dictionary.
    Keeps placeholders intact if the key is missing.
    """
    pattern = r'{{\s*([^{}]+?)\s*}}'  # match everything inside {{...}}
    
    def replacer(match):
        key = match.group(1)  
        return str(values.get(key, match.group(0)))  
    
    return re.sub(pattern, replacer, template)


def render_markdown_from_key(protein_key, annotations_dict):
    """
    Renders the TABLE_MARKDOWN for the given protein_key
    using the information from annotations_dict.
    """
    # Get the protein info dict
    info = annotations_dict.get(protein_key, {})
    
    # Map values to the placeholders
    values = {
        "Protein Name": info.get("initial_name", ""),
        "Gene Name": info.get("gene", ""),
        "Organism": info.get("organism", ""),
        "ProteinExistence": info.get("protein_existence", ""),
        "AminoAcids": str(info.get("amino_acids", "")),
        "AnnotationScore": str(info.get("annotation_score", "")),
    }
    
    return render_template(TABLE_MARKDOWN, values)

def render_desc_from_key(protein_key, annotations_dict):
    """
    Renders the TABLE_MARKDOWN for the given protein_key
    using the information from annotations_dict.
    """
    # Get the protein info dict
    info = annotations_dict.get(protein_key, {})
    
    # Map values to the placeholders
    values = {
        "Description": info.get("description", "")
    }

    return render_template(DESCRIPTION_MARKDOWN, values)


def get_protein_image_path(protein_key):
    """
    Returns the path to the protein image.
    If the protein-specific image doesn't exist, returns a random image.
    """
    import random
    
    # Try to find the specific protein image
    protein_img_path = os.path.join(PROTEIN_IMGS_PATH, f"{protein_key}.png")
    
    if os.path.exists(protein_img_path):
        return protein_img_path
    
    # If not found, choose a random image
    random_imgs = [f"rand_{i}.png" for i in range(7)]  # rand_0.png to rand_6.png
    random_img = random.choice(random_imgs)
    return os.path.join(PROTEIN_IMGS_PATH, random_img)


def find_gocode_description(go_code):
    import requests

    def replace_colon(s):
        return s.replace(":", "%3A")

    requestURL = f"https://www.ebi.ac.uk/QuickGO/services/ontology/go/search?query={replace_colon(go_code)}&limit=1&page=1"
    r = requests.get(requestURL, headers={ "Accept" : "application/json"})
    if r.status_code != 200:
        return "Description not found."

    data = r.json()
    results = data.get("results", [])
    if not results:
        return "Description not found."

    definition = results[0].get("definition", {})
    result_name = results[0].get("name", "")
    definition_text = definition.get("text", "Description not found.")
    return result_name, definition_text


def find_multiple_gocode_descriptions(go_codes):
    """
    Fetch descriptions for multiple GO codes in a single API call.
    Returns two dictionaries: one mapping GO code to name, another to description.
    """
    import requests
    
    if not go_codes:
        return {}, {}
    
    # Join GO codes with OR for batch query
    query = " OR ".join(go_codes)
    
    requestURL = f"https://www.ebi.ac.uk/QuickGO/services/ontology/go/search"
    params = {
        "query": query,
        "limit": len(go_codes),
        "page": 1
    }
    
    try:
        r = requests.get(requestURL, headers={"Accept": "application/json"}, params=params)
        if r.status_code != 200:
            # Fallback to empty dicts
            return {code: "Name not found." for code in go_codes}, {code: "Description not found." for code in go_codes}
        
        data = r.json()
        results = data.get("results", [])
        
        # Create mappings from GO ID to name and description
        names = {}
        descriptions = {}
        for result in results:
            go_id = result.get("id", "")
            result_name = result.get("name", "Name not found.")
            definition = result.get("definition", {})
            definition_text = definition.get("text", "Description not found.")
            names[go_id] = result_name
            descriptions[go_id] = definition_text
        
        # Fill in any missing codes
        for code in go_codes:
            if code not in names:
                names[code] = "Name not found."
            if code not in descriptions:
                descriptions[code] = "Description not found."
        
        return names, descriptions
    
    except Exception as e:
        # Fallback in case of error
        return {code: "Name not found." for code in go_codes}, {code: "Description not found." for code in go_codes}

############
# MAIN APP #
############

def build_app():
    with open(TEST_ANNOTATIONS_INFO_FILE_PATH, 'r') as f:
        test_annotations = json.load(f)  # dict: prot ID -> list of GO terms
    keys_list = list(test_annotations.keys())

    with gr.Blocks(title=f"{APP_NAME} — Protein Function Prediction", theme=theme, css=custom_css) as demo:
        create_intro_section()

        gr.Markdown("---")
        gr.Markdown("")

        with gr.Row():
            with gr.Column():
                protein_sel = gr.Dropdown(
                    choices=keys_list,
                    value=keys_list[0] if keys_list else None,
                    label="Protein ID",
                )
                gr.Markdown("")
                with gr.Column():
                    method = gr.Dropdown(
                        METHODS_NAMES,
                        value=METHODS_NAMES[0] if METHODS_NAMES else None,
                        label="Method",
                    )
                    topk = gr.Slider(3, 10, value=5, step=1, label="Top-K")
                    run_btn = gr.Button("Predict", variant="primary")
                    clear_btn = gr.Button("Clear")
            with gr.Column():
                # PROTEIN INFO MARKDOWN
                protein_md = gr.Markdown("", label="Protein Description", elem_id="protein_info")
                def update_markdown(selected_protein):
                    return render_markdown_from_key(selected_protein, test_annotations)
                protein_sel.change(update_markdown, inputs=[protein_sel], outputs=[protein_md])

                # DESCRIPTION MARKDOWN
                description_md = gr.Markdown("", elem_id="description_info")
                def update_description(selected_protein):
                    return render_desc_from_key(selected_protein, test_annotations)
                protein_sel.change(update_description, inputs=[protein_sel], outputs=[description_md])
                
        # PROTEIN IMAGE
        protein_img = gr.Image(label="Protein Structure", type="filepath", height=300)        
        
        def update_protein_image(selected_protein):
            return get_protein_image_path(selected_protein)
        
        protein_sel.change(update_protein_image, inputs=[protein_sel], outputs=[protein_img])
        
        # UniProt link button
        uniprot_link = gr.Button("🔗 View on UniProt", variant="secondary", size="sm")
        uniprot_link.click(
            lambda selected_protein: None,
            inputs=[protein_sel],
            outputs=None,
            js="""(selected_protein) => {
                const code = selected_protein.split(' - ')[0].trim();
                const url = `https://www.uniprot.org/uniprotkb/${code}/entry`;
                window.open(url, '_blank');
            }"""
        )

        gr.Markdown("")
        gr.Markdown("---")
        gr.Markdown("")

        meta = gr.Markdown("")
        out_df = gr.Dataframe(
            headers=["", "GO_ID", "Score", "Name", "Description"],
            datatype=["str", "str", "number", "str", "str"],
            label="Predictions",
            interactive=False,
            wrap=True,
            row_count=(0, "dynamic"),
            column_widths=["5%", "12%", "20%", "15%", "48%"],
        )

        truth_stats = gr.Markdown("")
        out_truth = gr.Dataframe(
            headers=["GO_ID"],
            datatype=["str"],
            label="True GO Terms",
            interactive=False,
            wrap=True,
            row_count=(5, "dynamic"),
            max_height=300,
        )

        def _predict(protein_id, method_sel, k):
            df_pred, info, meta_txt = predict(protein_id, method_sel, k)
            # Load true annotations from test.json
            with open(TEST_ANNOTATIONS_FILE_PATH, 'r') as f:
                true_annotations = json.load(f)
            truth_terms = true_annotations.get(protein_id, [])
            df_truth = pd.DataFrame({"GO_ID": truth_terms})
            
            # Get predicted GO codes for highlighting
            predicted_go_codes = set(df_pred["GO_ID"].tolist())
            truth_go_codes = set(truth_terms)
            
            # Add Prediction column with good bad emoji 
            df_pred.insert(0, "", df_pred["GO_ID"].apply(lambda x: "🟢" if x in truth_go_codes else "🔴"))
            
            # Calculate statistics
            matches = predicted_go_codes.intersection(truth_go_codes)
            num_matches = len(matches)
            num_predicted = len(predicted_go_codes)
            num_truth = len(truth_go_codes)
            precision = (num_matches / num_predicted * 100) if num_predicted > 0 else 0
            
            stats_text = f"**Matches: {num_matches}/{num_predicted} predicted** ({precision:.1f}% precision) | **Total true GO terms: {num_truth}**"
            
            # Apply styling to highlight matching GO codes
            def highlight_matches(styler):
                def highlight_row(row):
                    if row["GO_ID"] in predicted_go_codes:
                        return ['background-color: #005451'] * len(row)  # Light blue
                    return [''] * len(row)
                
                styler.apply(highlight_row, axis=1)
                return styler
            
            df_truth_styled = df_truth.style.pipe(highlight_matches)
            
            if info:
                gr.Info(info)
            return df_pred, f"**{meta_txt}**", stats_text, df_truth_styled

        run_btn.click(
            _predict,
            inputs=[protein_sel, method, topk],
            outputs=[out_df, meta, truth_stats, out_truth],
        )

        clear_btn.click(
            lambda: (keys_list[0], "Stacked Ensemble", 5, "", pd.DataFrame(), "", pd.DataFrame()),
            inputs=None,
            outputs=[protein_sel, method, topk, meta, out_df, truth_stats, out_truth],
        )

        # gr.Examples(
        #     examples=[
        #         [
        #             test_prot_ids[0],
        #             "Naive",
        #             7,
        #         ],
        #     ],
        #     inputs=[protein_sel, method, topk],
        #     label="Examples",
        # )

#         gr.Markdown(
#             """
# **Disclaimer:** This UI is for demonstration purposes. Some methods are implemented with actual components:
# - Naive: Uses frequency of GO terms in the training data
# - DiamondScore: Uses BLAST to find similar proteins and their GO terms
# - InteractionScore: Uses PPI network to find interacting proteins and their GO terms
# - EmbeddingSimilarityScore: Uses cosine similarity between protein sequence embeddings

# Other methods (FC on embeddings, GNN on PPI & embeddings, Stacked Ensemble) still use placeholder implementations.
# """
#         )
    return demo


if __name__ == "__main__":
    build_app().launch()
