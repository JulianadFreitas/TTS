
# 🚀 Pull Request Lifetime Prediction

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Publication-ICSME%202026-brightgreen)](https://www.lsu.edu)

*A publication-ready machine learning framework for predicting pull request lifetime in open-source software projects.*

[Quick Start](#-quick-start) • [Features](#-features) • [Results](#-results) • [Contribute](#-contributing)

</div>

---

## 📋 Overview

This project develops and evaluates **Random Forest classifiers** to predict the lifetime duration of Pull Requests (PRs) across 15 major open-source repositories. The research investigates how **code metrics** and **label-derived features** impact prediction accuracy.

### Research Question
> How can we accurately predict whether a PR will be merged within 24 hours, 1 week, 1 month, or longer based on code-level characteristics and historical label patterns?

### Key Contributions
- ✅ **Multi-repository dataset**: 15 GitHub projects with 10K+ PRs
- ✅ **Feature engineering**: Code metrics + label indicator features
- ✅ **Balanced comparison**: Full label features vs. count-only ablation
- ✅ **Comprehensive analysis**: PRE/POST feature availability scenarios
- ✅ **Publication-ready**: Reproducible pipeline with full metadata tracking

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

---

## 🎯 Features

### Code Metrics (from Pull Request data)
- **Commits**: Number of commits, revisions, milestones
- **Changes**: Lines of code, files changed, code change ratio
- **Quality**: Test coverage, complexity metrics
- **Timeline**: Creation date, review duration

### Label Features (engineered)
- `CONT_label_count`: Total number of labels applied
- `CONT_label_*`: Binary indicator for each label type
  - bug, enhancement, documentation, etc.

### Feature Selection Modes
- **PRE**: Initial code metrics only
- **POST**: Code metrics + label features (available after PR creation)

### Label Feature Ablation
- **Full**: All CONT_label_* features (255 features)
- **Count-only**: Aggregate count only (17 features)

---

## 🔬 Experimental Design

### Methodology
1. **Data Preparation**: Clean, preprocess, and stratify PRs by lifetime
2. **Feature Selection**: PRE/POST scenarios with/without label ablation
3. **GridSearchCV**: Test 729 hyperparameter configurations
4. **Balancing Strategies**: none, undersample, oversample
5. **Evaluation**: Macro-F1 (CV) + F1-macro & Balanced Accuracy (test)
6. **Analysis**: Comparative reports with statistical summaries

### Hyperparameter Space
```python
{
    "strategy": ["none", "undersample", "oversample"],
    "n_estimators": [200, 500, 1000],
    "max_depth": [None, 20, 40],
    "max_features": ["sqrt", "log2", 0.5],
    "min_samples_split": [2, 10, 20],
    "min_samples_leaf": [1, 2, 4],
}
# Total combinations: 3^6 = 729 per CV fold
```

---

## 📈 Key Results

### Performance Summary

| Scenario | Feature Set | n_features | Test F1-Macro | Balanced Acc | CV Time (min) |
|----------|-------------|-----------|---------------|--------------|---------------|
| **With labels** | POST | 172 | **0.6731** | **0.659** | 86.98 |
| With labels | PRE | 163 | 0.66 | 0.6435 | 80 |
| No labels | POST | 25 | 0.6691 | 0.6551 | 81.77 |
| No labels | PRE | 16 | 0.6593 | 0.6434 | 79.32 |

### Key Insights
- ✅ **Label features improve POST performance**: +0.4% F1-macro (full vs count-only)
- ✅ **POST outperforms PRE**: +1.31% improvement with labels, +0.98% without
- ✅ **"None" strategy performs best**: No resampling preserves natural distribution
- ✅ **Feature count matters less than quality**: 25 vs 172 features achieve similar results

### Balancing Strategy Impact (with labels)

| Strategy | CV Macro-F1 (POST) | CV Macro-F1 (PRE) |
|----------|-------------------|-------------------|
| none | 0.6525 | 0.6302 |
| oversample | 0.6398 | 0.6178 |
| undersample | 0.6295 | 0.6086 |

---

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

---

## 📚 Documentation

- **[GridSearch Module](RF/gridsearch/README.md)**: Detailed documentation of the ML pipeline
- **[Data Directory](data/README.md)**: Feature definitions and dataset information
- **[Feature Engineering](RF/label_processor.py)**: Label feature extraction pipeline

---

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
- Publication-ready Markdown reports

---

## 📖 Citation

If you use this project in your research, please cite:

```bibtex
@inproceedings{freitas2026predicting,
  title={Predicting Pull Request Lifetime: A Multi-Repository Study},
  author={Freitas, Juliana and Fronchetti, Felipe},
  booktitle={2026 IEEE International Conference on Software Maintenance and Evolution (ICSME)},
  year={2026},
  organization={IEEE}
}
```

---

## 👥 Contact & Contributors

- **Juliana Freitas** — Principal Researcher  
  📧 [Juliana.Freitas@lsu.edu](mailto:Juliana.Freitas@lsu.edu)

- **Felipe Fronchetti** — Advisor  
  📧 [ffronchetti@lsu.edu](mailto:ffronchetti@lsu.edu)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

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

Made with ❤️ by the TTS Research Team at LSU

</div>
