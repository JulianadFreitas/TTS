
# 🚀 Pull Request Lifetime Prediction

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0+-orange.svg)](https://scikit-learn.org/)

*A publication-ready machine learning framework for predicting pull request lifetime in open-source software projects.*

[Quick Start](#-quick-start) • [Results](#-results) • [Contribute](#-contributing)

</div>

---

## 📋 Overview

This project develops and evaluates **Random Forest classifiers** to predict the lifetime duration of Pull Requests (PRs) across 15 major open-source repositories. The research investigates how **code metrics** and **label-derived features** impact prediction accuracy.

### Research Question
> How can we accurately predict whether a PR will be merged within 24 hours, 1 week, 1 month, or longer based on code-level characteristics and historical label patterns?
---

## 📁 Project Structure

```
TTS/
├── 📊 data/                          # Dataset and preprocessing
│   ├── RAW/                          # 15 repositories' combined PR data
│   ├── PREPROCESSED/                 # Cleaned & grouped datasets
│   ├── train.csv, test.csv          # Split datasets
│   └── features.txt                  # Feature definitions
│
├── 🤖 RF/                            # Random Forest experiments
│   ├── gridsearch/                   # GridSearchCV automation
│   │   ├── main.py                   # Experiment orchestrator
│   │   ├── runner.py                 # Standalone engine
│   │   ├── features.py               # Feature selection & ablation
│   │   ├── params.py                 # Hyperparameter grids
│   │   ├── pipeline.py               # ML pipeline construction
│   │   ├── sampling.py               # Resampling strategies
│   │   ├── metadata.py               # Reproducibility tracking
│   │   ├── reporting.py              # Artifact export
│   │   ├── analyze_results.py        # Publication-ready analysis
│   │   ├── README.md                 # Module documentation
│   │   └── outputs/                  # Experiment results
│   │
│   ├── data_preparation.py           # Dataset preprocessing
│   ├── label_processor.py            # Label feature engineering
│   └── plot_utils.py                 # Visualization utilities
│
├── requirements.txt                  # Python dependencies
└── README.md                          # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

```bash
# Clone the repository
git clone https://github.com/JulianadFreitas/TTS.git
cd TTS

# Create virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

#### 1. Run a Complete Experiment

```bash
cd RF/gridsearch
python3 main.py \
  --data-path ../data/train.csv \
  --out-dir outputs/my_experiment \
  --modes pre post \
  --label-features-modes full count_only
```

#### 2. Analyze Results (Single Root)

```bash
python3 analyze_results.py outputs/rf__labels-full__seed-42
```

#### 3. Compare Two Scenarios (Label Feature Impact)

```bash
python3 analyze_results.py \
  outputs/rf__labels-full__seed-42 \
  outputs/rf__labels-count_only__seed-42 \
  --scenario-with "with_labels" \
  --scenario-without "no_labels" \
  --out-dir analysis_labels_impact
```

### Output Files
- `results.json`: Detailed experiment metadata and metrics
- `cv_results.csv`: All GridSearchCV configurations tested
- `cv_best_per_strategy.csv`: Best hyperparameters per balancing strategy
- `FINAL_REPORT.md`: Publication-ready Markdown report with plots
- `plot_*.png`: Performance visualizations

## 🔄 Dataset Overview

### Repositories Analyzed
- ant-design
- audacity
- bitcoin
- flutter
- freeCodeCamp
- jabref
- langchain
- mrdoob/three.js
- nodejs
- playwright
- powertoys
- pytorch
- rust
- swift
- transformers

### PR Lifetime Groups
1. **Group 1**: 0–24 hours
2. **Group 2**: 1–7 days
3. **Group 3**: 8–30 days
4. **Group 4**: 1–3 months
5. **Group 5**: 6+ months

---

## 🛠️ Technologies

| Category | Tools |
|----------|-------|
| **ML Framework** | scikit-learn 1.3.0+ |
| **Data Processing** | pandas, numpy |
| **Visualization** | matplotlib, seaborn |
| **Serialization** | joblib, JSON |
| **Documentation** | Markdown |

---

## 📊 Reproducibility

This project is designed for **full reproducibility**:

✅ **Metadata Tracking**
- UTC timestamps (ISO 8601)
- Dataset SHA256 checksums
- Python & scikit-learn versions
- Complete hyperparameter logs

✅ **Random Seeds**
- Fixed random_state across all experiments
- Stratified K-Fold CV for stable splits
- Deterministic preprocessing pipeline

✅ **Artifact Export**
- All results saved to JSON
- Full CV grid outputs in CSV
---
## 👥 Contact & Contributors

- **Juliana Freitas** — Principal Researcher  
  📧 [Juliana.Freitas@lsu.edu](mailto:Juliana.Freitas@lsu.edu)

- **Felipe Fronchetti** — Advisor  
  📧 [ffronchetti@lsu.edu](mailto:ffronchetti@lsu.edu)

---

## 🤝 Contributing

We welcome contributions! Please feel free to:
- 🐛 Report bugs via GitHub Issues
- 💡 Suggest features or improvements
- 🔧 Submit pull requests with enhancements

For major changes, please open an issue first to discuss proposed changes.

---

<div align="center">

**⭐ If you find this project useful, please consider starring it!**

</div>
