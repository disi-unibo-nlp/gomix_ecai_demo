"""
Build a pre-computed FAISS index for fast protein function prediction.

This script loads all training protein embeddings and builds a single FAISS index
that can be loaded instantly during runtime, avoiding the 30-40 minute delay of
loading 65,000+ individual .pt files.

Usage:
    # Build with default paths (uses demo_utils structure)
    python build_faiss_index.py
    
    # Build with custom paths
    python build_faiss_index.py \
        --train-annotations path/to/train.json \
        --output-dir path/to/output \
        --embedding-types sequence
"""

import os
import sys
import json
import argparse
import faiss
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add src to path for imports
root = next(p for p in Path(__file__).resolve().parents if (p / "src").exists())
sys.path.insert(0, str(root))


def build_faiss_index(
    train_annotations_path: str,
    output_dir: str,
    embedding_types: list = None
):
    """
    Build and save a FAISS index for all training proteins.
    
    Args:
        train_annotations_path: Path to train.json with protein annotations
        output_dir: Directory where FAISS index files will be saved
        embedding_types: List of embedding types to use (default: ['sequence'])
    """
    # Import here after TASK_DATASET_PATH is set
    from src.utils.ProteinEmbeddingLoader import ProteinEmbeddingLoader
    
    if embedding_types is None:
        embedding_types = ['sequence']
    
    print("=" * 80)
    print("🚀 Building FAISS Index for Protein Embeddings")
    print("=" * 80)
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load training annotations
    print(f"📖 Loading training annotations from:")
    print(f"   {train_annotations_path}")
    with open(train_annotations_path, 'r') as f:
        train_annotations = json.load(f)
    
    num_proteins = len(train_annotations)
    protein_ids = list(train_annotations.keys())
    print(f"   ✅ Loaded {num_proteins:,} training proteins")
    print()
    
    # Initialize embedding loader
    print(f"🔧 Initializing embedding loader with types: {embedding_types}")
    prot_embedding_loader = ProteinEmbeddingLoader(types=embedding_types)
    embedding_size = prot_embedding_loader.get_embedding_size()
    print(f"   ✅ Embedding size: {embedding_size} dimensions")
    print()
    
    # Create FAISS index
    print(f"🏗️  Creating FAISS index (type: Flat, metric: Inner Product)")
    index = faiss.index_factory(embedding_size, "Flat", faiss.METRIC_INNER_PRODUCT)
    print(f"   ✅ Index created")
    print()
    
    # Load and add embeddings
    print(f"📦 Loading embeddings for {num_proteins:,} proteins...")
    print(f"   ⏱️  This will take approximately 30-40 minutes")
    print(f"   💡 Tip: Grab a coffee! This only needs to be done once.")
    print()
    
    failed_proteins = []
    embeddings_list = []
    valid_protein_ids = []
    
    for prot_id in tqdm(protein_ids, desc="   Processing", unit="protein"):
        try:
            # Load embedding
            embedding = prot_embedding_loader.load(prot_id)
            embedding = embedding.numpy().reshape(1, -1).astype('float32')
            
            # Normalize for cosine similarity
            faiss.normalize_L2(embedding)
            
            # Store for batch addition
            embeddings_list.append(embedding)
            valid_protein_ids.append(prot_id)
            
        except Exception as e:
            failed_proteins.append((prot_id, str(e)))
            continue
    
    # Add all embeddings to index at once
    if embeddings_list:
        print()
        print(f"➕ Adding {len(embeddings_list):,} embeddings to FAISS index...")
        all_embeddings = np.vstack(embeddings_list)
        index.add(all_embeddings)
        print(f"   ✅ Added {index.ntotal:,} vectors to index")
    
    # Report any failures
    if failed_proteins:
        print()
        print(f"⚠️  Warning: Failed to load {len(failed_proteins)} protein(s):")
        for prot_id, error in failed_proteins[:5]:
            print(f"   - {prot_id}: {error}")
        if len(failed_proteins) > 5:
            print(f"   ... and {len(failed_proteins) - 5} more")
    
    print()
    
    # Save FAISS index
    index_path = os.path.join(output_dir, "embeddings_faiss_index.faiss")
    print(f"💾 Saving FAISS index to:")
    print(f"   {index_path}")
    faiss.write_index(index, index_path)
    print(f"   ✅ Saved ({os.path.getsize(index_path) / (1024**3):.2f} GB)")
    print()
    
    # Save protein IDs order
    protein_ids_path = os.path.join(output_dir, "protein_ids_order.json")
    print(f"💾 Saving protein IDs mapping to:")
    print(f"   {protein_ids_path}")
    with open(protein_ids_path, 'w') as f:
        json.dump(valid_protein_ids, f, indent=2)
    print(f"   ✅ Saved ({os.path.getsize(protein_ids_path) / (1024**2):.2f} MB)")
    print()
    
    # Save metadata
    metadata = {
        "num_proteins": len(valid_protein_ids),
        "embedding_types": embedding_types,
        "embedding_size": embedding_size,
        "index_type": "Flat",
        "metric": "METRIC_INNER_PRODUCT",
        "failed_proteins": len(failed_proteins),
        "train_annotations_path": train_annotations_path
    }
    metadata_path = os.path.join(output_dir, "faiss_index_metadata.json")
    print(f"💾 Saving metadata to:")
    print(f"   {metadata_path}")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   ✅ Saved")
    print()
    
    # Summary
    print("=" * 80)
    print("✨ FAISS Index Built Successfully!")
    print("=" * 80)
    print(f"📊 Summary:")
    print(f"   • Total proteins indexed: {len(valid_protein_ids):,}")
    print(f"   • Embedding dimensions: {embedding_size:,}")
    print(f"   • Index file size: {os.path.getsize(index_path) / (1024**3):.2f} GB")
    print(f"   • Output directory: {output_dir}")
    print()
    print(f"🎯 Next Steps:")
    print(f"   1. The FAISS index is ready to use!")
    print(f"   2. Run your demo application - it will automatically detect and use the index")
    print(f"   3. Predictions will now be 1000x faster (40 min → <5 sec)")
    print()
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Build a pre-computed FAISS index for protein embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build with default paths (recommended for demo)
  python build_faiss_index.py
  
  # Build with custom paths
  python build_faiss_index.py \\
      --train-annotations path/to/train.json \\
      --output-dir path/to/output \\
      --embedding-types sequence text
        """
    )
    
    # Get default paths
    script_dir = Path(__file__).resolve().parent
    # Use task_datasets path for training data (where the actual annotations and embeddings are)
    default_train_path = script_dir.parent / "data" / "processed" / "task_datasets" / "2016" / "propagated_annotations" / "train.json"
    # Save to demo_utils for easy access by demo app
    default_output_dir = script_dir.parent / "demo_utils" / "faiss_index"
    
    parser.add_argument(
        "--train-annotations",
        type=str,
        default=str(default_train_path),
        help="Path to train.json with protein annotations"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(default_output_dir),
        help="Directory where FAISS index will be saved"
    )
    
    parser.add_argument(
        "--embedding-types",
        type=str,
        nargs='+',
        default=['sequence'],
        choices=['sequence', 'text', 'interpro'],
        help="Types of embeddings to use"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.train_annotations):
        print(f"❌ Error: Training annotations file not found:")
        print(f"   {args.train_annotations}")
        print()
        print(f"💡 Tip: Make sure you're running this from the correct directory")
        sys.exit(1)
    
    # Set environment variable for ProteinEmbeddingLoader
    task_dataset_path = Path(args.train_annotations).parent.parent
    os.environ["TASK_DATASET_PATH"] = str(task_dataset_path)
    
    # Build the index
    build_faiss_index(
        train_annotations_path=args.train_annotations,
        output_dir=args.output_dir,
        embedding_types=args.embedding_types
    )


if __name__ == "__main__":
    main()
