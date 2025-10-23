# GOMix: Protein Function Prediction with Ensemble Deep Learning

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3119/)

**Demo application and code for ECAI 2025 (Demo Track)**

## Paper Reference

**Title:** Predicting Protein Functions with Ensemble Deep Learning and Protein Language Models

**Authors:** Giacomo Frisoni, Marcello Fuschi, Gianluca Moro

**Conference:** ECAI 2025 (Demo Track)

## Project Description

Understanding protein functions is essential for deciphering how living organisms work at a cellular level and for advancing healthcare outcomes—from diagnosing diseases to developing targeted therapies. However, the discovery of new proteins far outpaces our ability to experimentally verify their functions. As of April 2023, the UniProtKB/Swiss-Prot database contains approximately 600,000 manually annotated proteins, yet over 200 million protein sequences have been discovered. This gap makes computational approaches for automated protein function prediction (PFP) increasingly critical.

**GOMix** is an ensemble learning method designed to predict the functions of newly discovered proteins. The system predicts which Gene Ontology (GO) terms apply to a given protein sequence.

### Key Features

GOMix distinguishes itself through several important characteristics:

1. **Ensemble approach**: Combines seven complementary prediction strategies, from traditional sequence alignment to modern protein language models
2. **Competitive performance**: Achieves state-of-the-art results on the CAFA-3 challenge benchmark
3. **Open-source**: Entirely free and accessible to the research community
4. **Modular design**: Components can be used independently or combined; new predictors can be easily integrated
5. **Low computational requirements**: Unlike many deep learning approaches, GOMix can run without expensive GPU infrastructure
6. **User-friendly interface**: Packaged as an easy-to-use web application for rapid protein function determination

The system fuses predictions from base learners including sequence homology methods (BLAST-based similarity), protein-protein interaction networks, protein language model embeddings (ESM2), fully-connected neural networks, and graph neural networks. This diversity enables GOMix to capture complementary aspects of protein function.

## Repository Structure

The repository is organized into three main areas: the **core prediction logic** (`gomix/`), the **web demonstration interface** (`demo-app/`), and **automation scripts** (`scripts/`).

```
.
├── gomix/                         # CORE: All prediction algorithms and data processing
│   ├── src/
│   │   ├── data_processing/      # Scripts to transform raw protein data into model inputs
│   │   ├── solution/             # All prediction methods (base learners + ensemble)
│   │   ├── utils/                # Shared tools for embeddings, evaluation, data loading
│   │   └── demo_utils/           # Pre-loaded test data and images for the demo app
│   ├── requirements.txt          # Python dependencies for core algorithms
│   └── README.md                 # Detailed technical documentation
│
├── demo-app/                      # INTERFACE: Web application for demonstrations
│   ├── app.py                    # Main Gradio UI (layout, interactions, API calls)
│   └── solutionrunners/          # Adapters that call gomix/ prediction methods
│       ├── naive.py              # Wrapper for frequency-based baseline
│       ├── diamondscore.py       # Wrapper for BLAST-based predictor
│       ├── interactionscore.py   # Wrapper for PPI network predictor
│       └── embeddingsimilarityscore.py  # Wrapper for ESM2 embedding predictor
│
├── scripts/                       # AUTOMATION: Setup and execution helpers
│   ├── setup.sh                  # Installs all Python dependencies
│   └── run-app.sh                # Launches the Gradio web application
│
├── app.py                        # Alternative entry point (delegates to demo-app/app.py)
├── test.ipynb                    # Jupyter notebook for experimentation
└── README.md                     # This file: project overview and setup instructions
```

### Role Distinctions

#### `gomix/` — Core Prediction Engine

This directory contains all scientific code for protein function prediction. **Modify files here to change prediction algorithms, training procedures, or evaluation metrics.**

**Sub-directories:**

- **`src/data_processing/`** — Eight numbered preprocessing scripts (01–08) that convert raw biological databases into model-ready formats:
  - `01_generate_all_proteins_fasta.py`: Extracts protein sequences into FASTA format
  - `02_generate_annotations_json.py`: Converts GO annotations to JSON
  - `03_propagate_annotations.py`: Propagates GO terms through the ontology hierarchy
  - `04_generate_all_proteins_diamond.py`: Runs DIAMOND/BLAST for sequence similarity
  - `05_prepare_STRING_network_from_raw_file.py`: Processes protein-protein interaction networks
  - `06_generate_protein_text_features.py`: Generates textual descriptions of proteins
  - `07_embed_protein_text_features.py`: Creates text embeddings using language models
  - `08_generate_protein_interpro_features.py`: Extracts protein domain features

  **When to modify**: Change these scripts if you need to incorporate new data sources or modify how raw data is transformed.

- **`src/solution/components/`** — Individual prediction methods (base learners). Each subdirectory represents one approach:

  - **`naive/`**: Baseline predictor that assigns GO terms based on training set frequency
    - `NaiveLearner.py`: Implementation of the frequency-based predictor
    - `demo.py`: Standalone script to evaluate this method
    - **Modify** `NaiveLearner.py` to change the baseline prediction strategy

  - **`diamondscore/`**: BLAST-based sequence homology predictor
    - `DiamondScoreLearner.py`: Uses DIAMOND to find similar proteins and transfer their GO terms
    - **Modify** to adjust similarity thresholds or scoring functions

  - **`interactionscore/`**: PPI network-based predictor
    - `InteractionScoreLearner.py`: Leverages protein interaction partners' functions
    - **Modify** to change how interaction scores are weighted or combined

  - **`embeddingsimilarityscore/`**: ESM2 embedding similarity predictor
    - `Learner.py`: Finds proteins with similar ESM2 embeddings and transfers functions
    - **Modify** to adjust cosine similarity thresholds or re-weighting strategies

  - **`FC_on_embeddings/`**: Neural network trained on protein embeddings
    - `ProteinToGOModel.py`: Fully-connected network architecture
    - `main.py`: Training and evaluation script
    - **Modify** `ProteinToGOModel.py` to change network architecture (layers, neurons, activation functions)

  - **`GNN_on_PPI_with_embeddings/`**: Graph neural network using PPI topology
    - `Net.py`: Graph convolutional network architecture
    - `ProteinGraphBuilder.py`: Constructs protein interaction graphs
    - `main.py`: Training and evaluation script
    - **Modify** `Net.py` to experiment with different GNN architectures (GAT, GraphSAINT, SIGN)

- **`src/solution/stacked_ensemble/`** — Meta-learner that combines all base predictors
  - `StackingMetaLearner.py`: Ensemble model that learns optimal weights for base predictions
  - `Level1Dataset.py`: Prepares base learner outputs as features for the meta-learner
  - `demo.py`: Runs the full ensemble and evaluates performance
  - **Modify** `StackingMetaLearner.py` to change ensemble strategy (e.g., weighted averaging, gradient boosting)

- **`src/utils/`** — Shared utilities used across all methods:
  - `embed_proteins_from_fasta.py`: Generates ESM2 embeddings from protein sequences
  - `ProteinEmbeddingLoader.py`: Loads pre-computed embeddings efficiently
  - `EmbeddedProteinsDataset.py`: PyTorch dataset wrapper for protein embeddings
  - `predictions_evaluation/`: CAFA and DeepGOPlus evaluation metric implementations
  - **Modify** these files to add new embedding models or evaluation metrics

- **`src/demo_utils/`** — Pre-loaded test data for the web application
  - `test_uniprotid_info_formatted.txt`: Metadata for demonstration proteins (names, descriptions)
  - `imgs/`: Protein structure visualizations displayed in the UI
  - **Modify** to add new demonstration proteins or update metadata

#### `demo-app/` — Web Application Interface

This directory contains only presentation logic. **Modify files here to change UI appearance, user interactions, or how results are displayed.**

- **`app.py`** — Main Gradio application implementing:
  - UI layout (dropdowns, buttons, tables, protein visualization)
  - User interaction handlers (predict button, protein selection)
  - GO term description fetching from QuickGO API
  - Result formatting and comparison with ground truth
  - **Modify** to change UI design, add new controls, or alter how predictions are displayed

- **`solutionrunners/`** — Thin wrapper modules that adapt `gomix/` methods for the web interface:
  - Each `.py` file imports the corresponding learner from `gomix/src/solution/components/`
  - Provides a simple `predict(protein_id)` function called by the Gradio UI
  - **Modify** these only if you need to change how data flows between the UI and prediction algorithms

**Important**: `demo-app/` contains no scientific logic. It only calls methods from `gomix/`. To add a new prediction method:
1. Implement it in `gomix/src/solution/components/your_method/`
2. Create a wrapper in `demo-app/solutionrunners/your_method.py`
3. Register it in `demo-app/app.py` (add to `METHODS_NAMES` list and `predict()` function)

#### `scripts/` — Automation Scripts

Shell scripts for environment setup and application launch. **Run these for installation and deployment.**

- **`setup.sh`** — Installs all dependencies:
  - Verifies Python 3.11 is available
  - Installs PyTorch 2.0.1 with CUDA 11.8 support
  - Installs PyTorch Geometric extensions (torch-scatter, torch-sparse)
  - Installs Gradio and scientific computing libraries
  - **Modify** if you need different PyTorch versions or additional packages

- **`run-app.sh`** — Launches the Gradio web application
  - Sets up environment paths
  - Starts the server on `http://127.0.0.1:7860`
  - **Modify** to change host/port or add command-line arguments

## Installation Guide

### Prerequisites

- **Python 3.11**: This project requires Python 3.11.X specifically. Download from [python.org](https://www.python.org/downloads/release/python-3119/)
- **Git Bash**: Required for running shell scripts on Windows systems
- **CUDA 11.8** (optional): For GPU acceleration of neural network training

### Setup Instructions

Execute these commands in a Git Bash shell:

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd gomix_ecai_demo
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv gomix-env
   source gomix-env/Scripts/activate
   ```

3. **Install all dependencies**:
   ```bash
   ./scripts/setup.sh
   ```

   This script installs:
   - PyTorch 2.0.1 with CUDA support
   - PyTorch Geometric and extensions (torch-scatter, torch-sparse)
   - Gradio for the web interface
   - NumPy, Pandas, Scikit-learn, Matplotlib
   - All dependencies from `gomix/requirements.txt`

### Running the Application

Launch the Gradio demonstration interface:

```bash
./scripts/run-app.sh
```

The application will start a local web server at `http://127.0.0.1:7860`. Open this URL in your browser to interact with GOMix.

## Usage

### Web Interface

1. **Select a protein**: Choose a UniProt ID from the dropdown menu
2. **Choose prediction method**: Select from Naive, DiamondScore, InteractionScore, EmbeddingSimilarityScore, or ensemble methods
3. **Set Top-K**: Specify how many top predictions to display (3–10)
4. **Click "Predict"**: Generate GO term predictions with confidence scores
5. **Review results**: Compare predictions against experimentally verified GO terms (ground truth)

The interface displays:
- Protein metadata (name, organism, sequence length)
- Predicted GO terms with confidence scores and descriptions
- Visual indicators showing which predictions match the ground truth
- Precision statistics

### Command-Line Execution

Individual components can be run independently for evaluation:

```bash
# Set the dataset path
export TASK_DATASET_PATH=gomix/src/data/processed/task_datasets/2016

# Run individual base learners
python gomix/src/solution/components/naive/demo.py
python gomix/src/solution/components/diamondscore/demo.py
python gomix/src/solution/components/interactionscore/demo.py
python gomix/src/solution/components/embeddingsimilarityscore/main.py
python gomix/src/solution/components/FC_on_embeddings/main.py
python gomix/src/solution/components/GNN_on_PPI_with_embeddings/main.py

# Run full stacked ensemble
python gomix/src/solution/stacked_ensemble/demo.py
```

## Technical Documentation

For detailed technical information including:
- Data preprocessing pipeline details
- Model architectures and hyperparameters
- Training procedures and optimization strategies
- Evaluation metrics and protocols
- Ablation studies and future improvement directions

Please refer to [`gomix/README.md`](gomix/README.md).

## Citation

If you use GOMix in your research, please cite:

```bibtex
@inproceedings{frisoni2025gomix,
  title={Predicting Protein Functions with Ensemble Deep Learning and Protein Language Models},
  author={Frisoni, Giacomo and Fuschi, Marcello and Moro, Gianluca},
  booktitle={Proceedings of the European Conference on Artificial Intelligence (ECAI)},
  year={2025},
  note={Demo Track}
}
```

## Acknowledgments

We thank Stefano Fantazzini for his valuable help in realizing the demonstration application.

This work builds upon:
- [ESM2](https://github.com/facebookresearch/esm) protein language models by Meta AI
- [DeepGOPlus](https://github.com/bio-ontology-research-group/deepgoplus) evaluation framework
- [CAFA challenge](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-019-1835-8) benchmarks
- STRING protein-protein interaction database
- Gene Ontology Consortium

## License

MIT License.