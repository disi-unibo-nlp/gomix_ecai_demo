#!/bin/bash

# Build FAISS Index for Protein Embeddings
# This script builds a pre-computed FAISS index to speed up protein function prediction
# from 30-40 minutes to <5 seconds per prediction.

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo ""
echo "================================================================================"
echo -e "${BLUE}🚀 FAISS Index Builder for GOMix${NC}"
echo "================================================================================"
echo ""

# Get project root (parent of scripts directory)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

echo -e "${BLUE}📁 Project root:${NC} $PROJECT_ROOT"
echo ""

# Check if virtual environment exists
VENV_PATH="$PROJECT_ROOT/venv"
if [ ! -d "$VENV_PATH" ]; then
    echo -e "${RED}❌ Virtual environment not found at:${NC} $VENV_PATH"
    echo ""
    echo -e "${YELLOW}💡 Please create a virtual environment first:${NC}"
    echo "   python -m venv venv"
    echo "   source venv/Scripts/activate  # On Windows Git Bash"
    echo "   pip install -r gomix/requirements.txt"
    echo ""
    exit 1
fi

# Activate virtual environment
echo -e "${BLUE}🔧 Activating virtual environment...${NC}"
if [ -f "$VENV_PATH/Scripts/activate" ]; then
    # Windows (Git Bash)
    source "$VENV_PATH/Scripts/activate"
elif [ -f "$VENV_PATH/bin/activate" ]; then
    # Linux/Mac
    source "$VENV_PATH/bin/activate"
else
    echo -e "${RED}❌ Could not find activate script in venv${NC}"
    exit 1
fi
echo -e "${GREEN}   ✅ Virtual environment activated${NC}"
echo ""

# Check if required packages are installed
echo -e "${BLUE}📦 Checking required packages...${NC}"
python -c "import faiss" 2>/dev/null || {
    echo -e "${YELLOW}   ⚠️  faiss-cpu not found, installing...${NC}"
    pip install faiss-cpu
}

python -c "import tqdm" 2>/dev/null || {
    echo -e "${YELLOW}   ⚠️  tqdm not found, installing...${NC}"
    pip install tqdm
}

python -c "import torch" 2>/dev/null || {
    echo -e "${RED}   ❌ PyTorch not found. Please install it first:${NC}"
    echo "      pip install torch"
    exit 1
}

echo -e "${GREEN}   ✅ All required packages are installed${NC}"
echo ""

# Navigate to gomix directory
cd "$PROJECT_ROOT/gomix"

# Check if training annotations exist
TRAIN_ANNOTATIONS="$PROJECT_ROOT/gomix/src/data/processed/task_datasets/2016/propagated_annotations/train.json"
if [ ! -f "$TRAIN_ANNOTATIONS" ]; then
    echo -e "${RED}❌ Training annotations not found at:${NC}"
    echo "   $TRAIN_ANNOTATIONS"
    echo ""
    echo -e "${YELLOW}💡 Make sure your data is in the correct location${NC}"
    exit 1
fi

# Set output directory
OUTPUT_DIR="$PROJECT_ROOT/gomix/src/demo_utils/faiss_index"

echo -e "${BLUE}🎯 Build Configuration:${NC}"
echo "   • Training annotations: $TRAIN_ANNOTATIONS"
echo "   • Output directory: $OUTPUT_DIR"
echo "   • Embedding types: sequence"
echo ""

# Ask for confirmation
echo -e "${YELLOW}⏱️  This will take approximately 30-40 minutes.${NC}"
echo -e "${YELLOW}   The script will load 65,000+ protein embeddings and build a FAISS index.${NC}"
echo ""
read -p "Continue? [y/N] " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}❌ Build cancelled${NC}"
    exit 0
fi

echo ""
echo "================================================================================"
echo -e "${GREEN}🏗️  Starting FAISS Index Build${NC}"
echo "================================================================================"
echo ""

# Run the build script
python src/data_processing/build_faiss_index.py \
    --train-annotations "$TRAIN_ANNOTATIONS" \
    --output-dir "$OUTPUT_DIR" \
    --embedding-types sequence

# Check if build was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "================================================================================"
    echo -e "${GREEN}✅ FAISS Index Built Successfully!${NC}"
    echo "================================================================================"
    echo ""
    echo -e "${GREEN}🎉 The pre-built index is ready to use!${NC}"
    echo ""
    echo -e "${BLUE}📊 Index Location:${NC}"
    echo "   $OUTPUT_DIR"
    echo ""
    echo -e "${BLUE}🚀 Next Steps:${NC}"
    echo "   1. Run your demo application:"
    echo "      ./scripts/run-app.sh"
    echo ""
    echo "   2. The app will automatically detect and use the pre-built index"
    echo ""
    echo "   3. Enjoy 1000x faster predictions! ⚡"
    echo "      • Before: 30-40 minutes per prediction"
    echo "      • After: <5 seconds per prediction"
    echo ""
    echo "================================================================================"
else
    echo ""
    echo "================================================================================"
    echo -e "${RED}❌ Build Failed${NC}"
    echo "================================================================================"
    echo ""
    echo -e "${YELLOW}💡 Troubleshooting:${NC}"
    echo "   1. Check that you have enough disk space (~2 GB free)"
    echo "   2. Verify that all embedding files exist in:"
    echo "      gomix/src/data/processed/task_datasets/2016/all_protein_sequence_embeddings/"
    echo "   3. Make sure all required Python packages are installed"
    echo ""
    exit 1
fi
