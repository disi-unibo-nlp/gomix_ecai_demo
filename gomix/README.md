# Protein function prediction

In this research work, we tackle the problem of predicting the GO terms associated to a protein, based on the protein's amino-acidic sequence and the way it interacts with other proteins as specified by PPI networks (Protein-Protein Interaction networks), like STRING.

This task is the topic of the [CAFA](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-019-1835-8) challenge.

## Dataset

- `data/raw/task_datasets/CAFA3` comes from https://github.com/bio-ontology-research-group/deepgoplus
- `data/raw/Uniprot/uniprot_swiss_entries.dat` comes from https://www.uniprot.org/help/downloads
- `data/raw/InterPro/58.0__match_complete.xml` comes from https://ftp.ebi.ac.uk/pub/databases/interpro/releases/58.0/

All around the code, when we refer to "TASK_DATASET_PATH" we usually mean one of the subfolders of `data/processed/task_datasets`.

The `data/processed`directory should contain the output of processing made on the `data/raw` files using the scripts in `src/data_processing`. It should also contain protein embeddings generated using the script provided by Meta (described later).

**[Here](https://liveunibo-my.sharepoint.com/:u:/g/personal/marcello_fuschi_studio_unibo_it/EWfa8C1OC15DoL76bw0PP98BLYxh7oXkVTqfonEZ3drLfg?e=ouotUi "Here") you can download the already-processed data for the "2016" dataset, so that you can run the experiments immediately instead of having to process the raw data again. Instructions on how to run the experiments are provided below in this file.**

Once you have downloaded the processed dataset from the above paragraph, paste it into the `data/processed/task_datasets` directory.

### "2016" dataset

In `data/raw/task_datasets/2016`. Downloaded from https://github.com/bio-ontology-research-group/deepgoplus.
In the DeepGOPlus paper, it was the dataset that's used to compare DeepGOPlus with GOLabeler and DeepText2GO.

This same dataset is also used by other good-performing papers:
- PANDA2
- DeepGOPlus
- DeepText2GO (https://www.sciencedirect.com/science/article/pii/S1046202318300021#s0030)
- GoLabeler

The .pkl files are pickle files containing a pandas dataframe.

### NetGO2 dataset

Taken from [DeepGOZero](https://academic.oup.com/bioinformatics/article/38/Supplement_1/i238/6617515). Our train is their train+valid, and the testing set is the same as testing set in both DeepGOZero and NetGO2 papers. Currently, the `data/processed/task_datasets/NetGO2/all_proteins.fasta` file is incomplete because not all proteins in the dataset were found by the Uniprot mapping tool. This may be fixable if we find some other source.

## Methods

To encode proteins, we'll sometimes use **ESM2**, a protein language model by Facebook. It first embeds amino acids, then you may take the average between them as the protein embedding.

Meta provided a script for embedding the proteins from a FASTA file, that I copied into `src/utils/embed_proteins_from_fasta`. Usage is described [here](https://github.com/facebookresearch/esm). 

Example (to embed proteins): `python src/utils/embed_proteins_from_fasta.py esm2_t33_650M_UR50D data/raw/CAFA3_training_data/uniprot_sprot_exp.fasta data/processed/CAFA3_training_data/protein_embeddings --include mean`

Names of the available ESM2 models: https://huggingface.co/facebook/esm2_t33_650M_UR50D

The solution we propose is a stacked ensemble model that uses multiple components:
- **Naive**: always predicts the most frequent GO terms. (see the DeepGOPlus paper)
- **DiamondScore**: uses BLAST to find similar proteins and then uses their GO terms. (see the DeepGOPlus paper)
- **InteractionScore**: uses the PPI network to find interacting proteins and then uses their GO terms (similarly to DiamondScore).
- **EmbeddingSimilarityScore**: uses cosine similarity between protein _sequence_ embeddings to find similar proteins and then uses their GO terms (similarly to DiamondScore, except the cosine similarities are re-weighted for better performance, esp. for high _k_).
- **FC** on protein embeddings
- **GNN** on graph with protein embeddings as node features and PPI edges

The last 2 are the only ones based on neural networks training.

### Ideas to improve the current solution

**Minor:**
- Reduce the number of training GO terms considered, like DeepGOPlus did (they sued only the GO terms that were associated to at leat 50 proteins in the training set). This way you can use more memory-demanding methods.
- Change the proportion of train-test split for training the base models to generate the level-1 train set.
- Try using dropout instead of batch normalization in FC-on-embeddings.
- Try reducing the number of linear regressors used in the stacked ensemble.
- Try increasing the size of the neural-network models.
- Try using a different criterion (other than general F_max) for early stopping when training NN models.

**Major:**
- Think of ways to improve the GNN component, which gives low performance gain when included in the ensemble.
- Use protein features from InterPro as input, like other papers did (e.g., [NetGO](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6602452/), [NetGO 2.0](https://academic.oup.com/nar/article/49/W1/W469/6285266) and [DeepGraphGO](https://academic.oup.com/bioinformatics/article/37/Supplement_1/i262/6319663)).
- Use text embeddings of protein-associated documents as input, a bit like [NetGO 2.0](https://academic.oup.com/nar/article/49/W1/W469/6285266#267025483) did (see "LR-text").
- Add [Proteinfer](https://google-research.github.io/proteinfer/) as component ([GitHub](https://github.com/google-research/proteinfer/tree/master)).
- Add [DeepGOA](https://ieeexplore.ieee.org/document/8983075) as component.
- Improve the current GCN with new methods such as [over-squashing prevention](https://arxiv.org/abs/2306.03589) and [half-hop](https://www.linkedin.com/posts/petarvelickovic_icml2023-activity-7090395512402534401-TGxD/?utm_source=share&utm_medium=member_desktop).
- Use [SIGN](https://arxiv.org/pdf/2004.11198.pdf) or [GraphSAINT](https://arxiv.org/abs/1907.04931) instead of the current GCN.
- Add as input the 3D structure of the proteins, coming from DBs like the Protein Data Bank (PDB). [Here](https://www.nature.com/articles/s41467-021-23303-9) is a Nature paper that uses it to predict protein function.
- Add information from other PPI networks besides STRING.
- Take inspiration from NetGO papers ([here](https://github.com/paccanarolab/netgo)'s an unofficial implementation of the oldest one).

## Current best results

### On 2016 dataset

Stacked ensemble with 5 of the 6 components **(GNN was excluded)**, using **ESM2 15B** sequence embeddings:
- **MFO** | F_max: 0.594 (optimal threshold=0.19) | S_min: 8.651 | AUPR: 0.534
- **BPO** | F_max: 0.493 (optimal threshold=0.32) | S_min: 33.050 | AUPR: 0.426
- **CCO** | F_max: 0.722 (optimal threshold=0.33) | S_min: 7.138 | AUPR: 0.728

Stacked ensemble with just 4 components **(no FC-on-embeddings nor GNN)**, using **ESM2 15B** sequence embeddings:
- **MFO** | F_max: 0.591 (optimal threshold=0.18) | S_min: 8.617 | AUPR: 0.537
- **BPO** | F_max: 0.493 (optimal threshold=0.31) | S_min: 32.820 | AUPR: 0.434
- **CCO** | F_max: 0.718 (optimal threshold=0.32) | S_min: 7.198 | AUPR: 0.737

## How to run the solution

Since the solution is an ensemble method, there are multiple components to it. Each one of these base models can be run independently, to measure its performance. Provided that the `data` directory already contains all the necessary files (described in the sections above), you can run the experiments using the following commands:

- Naive: `TASK_DATASET_PATH=data/processed/task_datasets/2016 python src/solution/components/naive/demo.py`
- DiamondScore: `TASK_DATASET_PATH=data/processed/task_datasets/2016 python src/solution/components/diamondscore/demo.py`
- InteractionScore: `TASK_DATASET_PATH=data/processed/task_datasets/2016 python src/solution/components/interactionscore/demo.py`
- EmbeddingSimilarityScore: `TASK_DATASET_PATH=data/processed/task_datasets/2016 python src/solution/components/embeddingsimilarityscore/main.py`
- FC on embeddings: `TASK_DATASET_PATH=data/processed/task_datasets/2016 python src/solution/components/FC_on_embeddings/main.py`
- GNN on PPI & embeddings: `TASK_DATASET_PATH=data/processed/task_datasets/2016 python src/solution/components/GNN_on_PPI_with_embeddings/main.py`
- **Ensemble method**: `TASK_DATASET_PATH=data/processed/task_datasets/2016 python src/solution/stacked_ensemble/demo.py`

By default, the ensemble method only includes 4 of the 6 base models. To include all, toggle the `USE_ALL_COMPONENTS` boolean variable in `src/solution/stacked_ensemble/demo.py`.

## Notes for paper writing

### Differential analysis

We could include in the final paper the differential analysis of various architectural decisions. Here are some of the dimensions that could be tested:
- FC on protein embeddings vs GCN with PPI edges
- ESM2 3B vs 15B
- GAT (Graph Attention Network) vs SAGEConv vs GraphSAINT vs SIGN
- different PPI nets (provided that we can get to improve the performance based on the information contained)
- different neighbor sampling thresholds
- different number of neurons or layers
- ablation of base learners in the ensemble

### Contributions of this research

- Evaluating how informative ESM2 protein embeddings are for function prediction.
- Comparing the different types of ESM2 embeddings.
- Evaluating how informative PPI networks are for function prediction, on top of the embeddings (ablation study).
- \[?\] Evaluating how informative protein 3D structures are for function prediction (ablation study).
- Evaluating which kind of GNN is best for this task.

### Relevant papers

**CAFA:**
- [CAFA1](http://www.ncbi.nlm.nih.gov/pubmed/23353650)
- [CAFA2](http://www.ncbi.nlm.nih.gov/pubmed/27604469)
- [CAFA3](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-019-1835-8)

**Top methods for protein function prediction:**
- [NetGO 3.0](https://www.sciencedirect.com/science/article/pii/S1672022923000669)
