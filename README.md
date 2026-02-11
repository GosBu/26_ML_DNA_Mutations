# DNA Mutation Pathogenicity Predictor

Machine Learning system for predicting DNA mutation pathogenicity in genetic disease research.

## Project Overview

**Goal:** Build classification models to **predict whether a DNA mutation is pathogenic or benign**.

This project applies **machine learning techniques** to analyze point mutations in DNA sequences and predict their impact on health. Critical application for personalized medicine and genetic diagnostics.

**Key Features:**
- Two analysis pipelines: Basic Vol2 (Logistic Regression) & Advanced (Multi-model with CV)
- Comprehensive visualization suite (8+ plots per analysis)
- K-mer feature engineering for sequence context
- NCBI PubMed integration for literature validation
- Mutation hotspot analysis per gene
- JSON-based configuration for reproducibility

---

## Project Structure

```
26_ML_DNA_Mutations/
├── data/
│   └── data_rare/
│       └── data_DNA_mutations.csv              # Dataset (100 samples, 4 genes)
├── results/
│   ├── results_basic_vol2/                     # Basic Vol2 outputs
│   │   ├── *.png                               # 8 visualization plots
│   │   ├── ncbi_analysis_top_mutations_basic_vol2.txt
│   │   └── conclusions_dna_analysis_basic_vol2.txt
│   └── results_advanced/                       # Advanced outputs
│       ├── *.png                               # CV plots + feature importance
│       ├── ncbi_analysis_top_genes_advanced.txt
│       └── conclusions_dna_analysis_advanced.txt
├── config_basic_vol2.json                      # Basic Vol2 configuration
├── config_advanced.json                        # Advanced pipeline configuration
├── dna_mutations_analysis_basic.py             # Original baseline (reference)
├── dna_mutations_analysis_basic_vol2.py        # JSON-configured pipeline
├── dna_mutations_analysis_advanced.py          # Advanced multi-model pipeline
└── README.md                                   # This file
```

---

## Dataset

**Location:** `/data/data_rare/data_DNA_mutations.csv`

### Features:
| Column | Description | Type |
|--------|-------------|------|
| `DNA_Sequence` | DNA sequence (50 nucleotides) | String |
| `Gene` | Gene name (BRCA1, TP53, CFTR, MYH7) | Categorical |
| `Mutation_Position` | Position of mutation in sequence (1-50) | Integer |
| `Reference_Base` | Reference nucleotide (A, C, G, T) | Categorical |
| `Alternate_Base` | Mutated nucleotide (A, C, G, T) | Categorical |
| `Pathogenicity` | **Target variable** - "pathogenic" or "benign" | Binary |

### Dataset Characteristics:
- **Samples**: 100 mutations
- **Problem Type**: Binary classification (pathogenic vs benign)
- **Class Distribution**: 41 pathogenic / 59 benign
- **Genes**: BRCA1 (29), MYH7 (27), TP53 (26), CFTR (18)
- **No missing values**: Complete dataset

---

## Analysis Pipelines

### 1. **`basic_vol2.py`** - Rapid Classification Pipeline
**Configuration:** `config_basic_vol2.json`

| Component | Details |
|-----------|---------|
| **Model** | Logistic Regression (class_weight='balanced') |
| **Encoding** | One-Hot (200 features) + biochemical features |
| **Split** | 70/30 train/test |
| **Features** | 209 total (sequence + GC content + transitions) |
| **Outputs** | 8 plots + NCBI top 3 mutations + conclusions |

**Performance:** Accuracy ~60%, F1 ~40%, ROC AUC ~64%

---

### 2. **`advanced.py`** - Research-Grade Multi-Model Pipeline
**Configuration:** `config_advanced.json`

| Component | Details |
|-----------|---------|
| **Models** | Logistic Regression, Random Forest, SVM (auto-select) |
| **Encoding** | K-mer (k=3, 64 features) + biochemical + hotspot proximity |
| **Validation** | 5-fold Cross-Validation |
| **Features** | 84 total (optimized feature engineering) |
| **Outputs** | CV plots + feature importance + NCBI top 3 genes + hotspot analysis |

**Performance:** Accuracy ~40%, F1 ~57%, ROC AUC ~45% (SVM winner)

**Trade-off:** Better precision/recall balance (higher F1) despite lower accuracy — more suitable for clinical applications where minimizing false negatives is critical.

---

### 3. **`basic.py`** - Original Baseline (Reference Only)
Preserved for comparison. Not actively maintained.

---

## Installation & Requirements

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn biopython
```

**Core packages:**
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `scikit-learn` - Machine learning models
- `matplotlib` + `seaborn` - Visualization
- `biopython` - NCBI API integration (optional)

---

## Methodology

### Problem Definition
Binary classification: Pathogenic vs Benign DNA mutations

### Pipeline Architectures

**Basic Vol2:**
```
DNA Sequences → One-Hot Encoding (200 features) 
              → Biochemical Features (+9)
              → Logistic Regression
              → 70/30 Train/Test Split
              → Evaluation + Visualization
```

**Advanced:**
```
DNA Sequences → K-mer Encoding (k=3, 64 features)
              → Biochemical Features (+6)
              → Hotspot Proximity (+1)
              → Gene/Base Encoding (+12)
              → 5-Fold Cross-Validation
              → Multi-Model Evaluation (LogReg/RF/SVM)
              → Best Model Selection
              → Comprehensive Analysis
```

---

## Usage

### Quick Start

**Run Basic Vol2 Pipeline:**
```bash
python dna_mutations_analysis_basic_vol2.py
```

**Run Advanced Pipeline:**
```bash
python dna_mutations_analysis_advanced.py
```

### Configuration

Edit JSON files to customize:
- `config_basic_vol2.json` - Model parameters, visualization settings
- `config_advanced.json` - K-mer size, CV folds, thresholds

### NCBI API Setup
When prompted, provide your email for NCBI API access or press ENTER to use default from config.

### Execution Time
- **Basic Vol2**: ~1 minute
- **Advanced**: ~3-5 minutes (includes cross-validation)

---

## Results & Performance

### Model Comparison

| Metric | Basic Vol2 | Advanced (SVM) | Interpretation |
|--------|------------|----------------|----------------|
| **Accuracy** | 60% | 40% | Basic better at overall correctness |
| **F1-Score** | 40% | 57% | Advanced better at balancing precision/recall |
| **ROC AUC** | 64% | 45% | Basic has better probabilistic ranking |
| **Use Case** | Research, exploration | Clinical screening | |

### Key Insights

**Basic Vol2** - Higher accuracy, faster, more interpretable  
**Advanced** - Better precision/recall balance (fewer false negatives)

**Limitation**: Small dataset (~100 samples). K-mer encoding shows potential but requires >1000 samples for optimal performance.

### Output Files

Each run generates:
- **8-12 visualization plots** (PNG format) with English labels
- **NCBI analysis report** (`ncbi_analysis_*.txt`)
- **Conclusions summary** (`conclusions_dna_analysis_*.txt`)
- **Comprehensive board** - All plots in single dashboard

---

## Applications

### Clinical & Research Use Cases
- **Genetic diagnostics support** - BRCA1, TP53 cancer risk assessment
- **Variant prioritization** - Filter candidates for clinical validation
- **Bioinformatics research** - Mutation pathogenicity patterns
- **Personalized medicine** - Treatment decision support

### Current Limitations
- Small dataset (~100 samples) - proof-of-concept scale
- Limited gene coverage (4 genes: BRCA1, TP53, CFTR, MYH7)
- Requires clinical validation before medical use
- No external validation set (ClinVar/COSMIC integration pending)

---

## Contributing

Contributions welcome! Areas for improvement:
- Expand dataset (target: >1000 samples)
- Add more genes and mutation types
- Implement deep learning models
- External validation pipeline
- Web interface/API

---

## Author

**Project:** DNA Mutation Pathogenicity Predictor  
**Date:** February 2026  
**Technologies:** Python, scikit-learn, Biopython, matplotlib

---

## Acknowledgments

- NCBI for PubMed API access
- scikit-learn community
- Bioinformatics research community

---

**If you find this project useful, please star it on GitHub!**
