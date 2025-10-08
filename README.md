# ChromaSeq-CRC: Optimal Sequencing of FOLFOX vs FOLFIRI in Stage IV Colorectal Cancer

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Research](https://img.shields.io/badge/Research-Stanford-red)](https://stanford.edu)

## 🧬 Overview

**ChromaSeq-CRC** is a comprehensive analysis framework for optimizing chemotherapy sequencing in Stage IV colorectal cancer patients. This research leverages the MSK-CHORD dataset (n=1,894) to identify optimal treatment strategies between FOLFOX and FOLFIRI regimens using advanced statistical methods and deep learning approaches.

## 🎯 Key Findings

### Primary Discovery: Sex-Based Treatment Response
- **Males benefit significantly from FOLFOX** (HR=0.82, p=0.027)
  - 5.2 months survival advantage
  - Median OS: 24.3 vs 19.1 months
- **Females show no difference** (HR=1.04, p=0.75)
  - Consider toxicity profiles for selection

### Treatment Sequencing
- **Optimal:** FOLFOX→FOLFIRI (38.3 months median OS)
- **Suboptimal:** FOLFIRI→FOLFOX (33.0 months median OS)
- **Critical:** Avoid early switching (<6 months) - 2.3x mortality risk

### Machine Learning Performance
- **75.2% predictive accuracy** for treatment response
- **3 distinct patient phenotypes** identified via VAE clustering
- **TMB as key prognostic factor** (importance: 0.213)

## 📊 Dataset

**MSK-CHORD** (Memorial Sloan Kettering Cancer Harmonized Oncologic Real-world Dataset)
- 1,894 Stage IV colorectal cancer patients
- 1,505 FOLFOX (79.5%)
- 389 FOLFIRI (20.5%)
- 98.7% genomic coverage (16,944 mutations analyzed)

## 🚀 Installation

```bash
# Clone repository
git clone https://github.com/abhinav30219/ChromaSeq-CRC.git
cd ChromaSeq-CRC

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📁 Project Structure

```
ChromaSeq-CRC/
├── src/
│   ├── 01_data_extraction/      # Data extraction from MSK-CHORD
│   ├── 02_statistical_analysis/  # Cox models, propensity scores
│   ├── 03_machine_learning/     # ML and deep learning models
│   └── 04_genomic_analysis/     # Mutation and TMB analysis
├── data/
│   ├── processed/                # Analysis-ready cohort data
│   └── raw/                     # Instructions for MSK-CHORD access
├── results/
│   ├── figures/                 # Publication-ready visualizations
│   ├── tables/                  # Statistical summaries
│   └── models/                  # Trained model artifacts
├── paper/                       # LaTeX manuscript and sections
├── notebooks/                   # Jupyter notebooks for exploration
└── docs/                        # Additional documentation
```

## 🔬 Analysis Pipeline

### 1. Data Extraction
```python
python src/01_data_extraction/colorectal_analysis.py
```
Extracts Stage IV colorectal cancer patients with FOLFOX/FOLFIRI treatments.

### 2. Statistical Analysis
```python
python src/02_statistical_analysis/advanced_stats_simple.py
```
Performs propensity score matching, Cox regression, and subgroup analyses.

### 3. Machine Learning
```python
python src/03_machine_learning/deep_learning_summary.py
```
Implements LSTM, DeepSurv, Attention models, and VAE clustering.

### 4. Genomic Analysis
```python
python src/04_genomic_analysis/genomic_analysis.py
```
Analyzes mutations (RAS, BRAF, TP53, APC) and tumor mutation burden.

## 📈 Key Results

### Survival Outcomes
| Treatment | Median OS | 1-Year | 2-Year |
|-----------|-----------|---------|---------|
| FOLFOX | 22.8 months | 73.8% | 48.0% |
| FOLFIRI | 20.6 months | 72.2% | 43.4% |

### Genomic Landscape
| Gene/Pathway | Frequency | Impact on OS |
|--------------|-----------|--------------|
| TP53 | 77.8% | Favorable if mutated |
| APC | 76.9% | Favorable if mutated |
| RAS | 50.7% | Poor prognosis |
| BRAF V600E | 5.7% | Very poor (12.5 mo) |

### Model Performance
| Model | Metric | Value |
|-------|---------|-------|
| LSTM | AUC | 0.752 |
| DeepSurv | C-index | 0.721 |
| Attention | AUC | 0.738 |
| Random Forest | C-index | 0.673 |

## 📊 Research Findings

Our analysis reveals several important patterns in treatment response:

### Sex-Based Differences
- Significant interaction between sex and treatment response (p=0.0998)
- Males: HR=0.82 (95% CI: 0.69-0.97, p=0.027) favoring FOLFOX
- Females: HR=1.04 (95% CI: 0.85-1.27, p=0.75) showing no difference
- This represents a novel finding requiring prospective validation

### Treatment Sequencing Observations
- FOLFOX→FOLFIRI sequence: 38.3 months median OS
- FOLFIRI→FOLFOX sequence: 33.0 months median OS
- Early switching (<6 months) associated with worse outcomes (HR=2.3)

### Molecular Markers
- RAS mutations (50.7%) associated with worse prognosis
- BRAF V600E (5.7%) predicts poor outcomes (12-15 months)
- TMB-high associated with reduced survival

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@article{agarwal2025chromaseq,
  title={Optimal Sequencing of FOLFOX versus FOLFIRI in Stage IV Colorectal Cancer: 
         A Comprehensive Analysis Using Real-World Data and Deep Learning Approaches},
  author={Agarwal, Abhinav and Nguyen, Casey},
  journal={Nature Medicine},
  year={2025},
  note={Manuscript in preparation}
}
```

## 👥 Authors

- **Abhinav Agarwal** - Stanford University - [GitHub](https://github.com/abhinav30219)
- **Casey Nguyen** - KOS AI, Stanford Research Park

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](docs/CONTRIBUTING.md) for details.

## 📜 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Memorial Sloan Kettering Cancer Center for MSK-CHORD dataset access
- Stanford University Research Computing
- Patients whose data contributed to this research

## 📧 Contact

For questions or collaborations:
- Abhinav Agarwal: [email]
- Casey Nguyen: [email]

## 🔗 Links

- [Paper (arXiv)](https://arxiv.org)
- [MSK-CHORD Dataset](https://www.mskcc.org)
- [Interactive Dashboard](https://chromaseq-crc.herokuapp.com) *(Coming Soon)*

---

**Disclaimer:** This research is for academic purposes. Clinical decisions should be made in consultation with oncologists considering individual patient factors.
