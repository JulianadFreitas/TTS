#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
rf_runner_optimized.py

Versão otimizada com:
- Grid search reduzido (evita travamento)
- Opção de RandomizedSearchCV (muito mais rápido)
- Busca em 2 estágios (coarse → fine)
- Progresso detalhado
- Checkpoints automáticos
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Tuple
import warnings

import numpy as np
import pandas as pd
from joblib import dump

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    StratifiedShuffleSplit,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import resample

warnings.filterwarnings('ignore')

TARGET_COL = "PR_lifetime"
LABEL_PREFIX = "CONT_label_"


# ----------------------------
# Sampling
# ----------------------------

def oversample_multiclass(
    X: pd.DataFrame, y: np.ndarray, random_state: int = 42
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Oversample minority classes up to the maximum class size."""
    y_ser = pd.Series(y).reset_index(drop=True)
    X_df = X.reset_index(drop=True)

    counts = y_ser.value_counts()
    max_n = int(counts.max())

    X_parts: List[pd.DataFrame] = []
    y_parts: List[pd.Series] = []

    for cls, n in counts.items():
        idx = y_ser[y_ser == cls].index
        X_cls = X_df.loc[idx]
        y_cls = y_ser.loc[idx]

        if int(n) < max_n:
            X_up, y_up = resample(
                X_cls,
                y_cls,
                replace=True,
                n_samples=max_n,
                random_state=random_state,
            )
            X_parts.append(X_up)
            y_parts.append(y_up)
        else:
            X_parts.append(X_cls)
            y_parts.append(y_cls)

    X_bal = pd.concat(X_parts).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    y_bal = pd.concat(y_parts).sample(frac=1.0, random_state=random_state).reset_index(drop=True).to_numpy()
    return X_bal, y_bal


def undersample_multiclass(
    X: pd.DataFrame, y: np.ndarray, random_state: int = 42
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Undersample majority classes down to the minimum class size."""
    y_ser = pd.Series(y).reset_index(drop=True)
    X_df = X.reset_index(drop=True)

    counts = y_ser.value_counts()
    min_n = int(counts.min())

    X_parts: List[pd.DataFrame] = []
    y_parts: List[pd.Series] = []

    for cls, n in counts.items():
        idx = y_ser[y_ser == cls].index
        X_cls = X_df.loc[idx]
        y_cls = y_ser.loc[idx]

        if int(n) > min_n:
            X_down, y_down = resample(
                X_cls,
                y_cls,
                replace=False,
                n_samples=min_n,
                random_state=random_state,
            )
            X_parts.append(X_down)
            y_parts.append(y_down)
        else:
            X_parts.append(X_cls)
            y_parts.append(y_cls)

    X_bal = pd.concat(X_parts).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    y_bal = pd.concat(y_parts).sample(frac=1.0, random_state=random_state).reset_index(drop=True).to_numpy()
    return X_bal, y_bal


class ResampledRF(BaseEstimator, ClassifierMixin):
    """RandomForest wrapper with resampling."""

    def __init__(
        self,
        strategy: str = "none",
        random_state: int = 42,
        n_estimators: int = 500,
        max_depth=None,
        max_features: Any = "sqrt",
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        class_weight: Any = None,
    ):
        self.strategy = strategy
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.class_weight = class_weight

    def fit(self, X, y):
        y_in = np.asarray(y)
        X_df = pd.DataFrame(X)

        if self.strategy == "oversample":
            X_df, y_in = oversample_multiclass(X_df, y_in, random_state=self.random_state)
        elif self.strategy == "undersample":
            X_df, y_in = undersample_multiclass(X_df, y_in, random_state=self.random_state)
        elif self.strategy != "none":
            raise ValueError("strategy must be one of: none, undersample, oversample")

        self.model_ = RandomForestClassifier(
            random_state=self.random_state,
            n_jobs=-1,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            max_features=self.max_features,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            class_weight=self.class_weight,
        )
        self.model_.fit(X_df, y_in)
        return self

    def predict(self, X):
        return self.model_.predict(pd.DataFrame(X))


# ----------------------------
# Feature selection
# ----------------------------

def select_features(
    df: pd.DataFrame,
    mode: str,
    pre_cols: List[str],
    post_only_cols: List[str],
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """Select features based on mode (pre or post)."""
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found.")

    label_cols = [
        c for c in df.columns
        if c.startswith(LABEL_PREFIX) and c != "CONT_label_count"
    ]

    if mode == "pre":
        wanted = pre_cols + label_cols
    elif mode == "post":
        wanted = pre_cols + post_only_cols + label_cols
    else:
        raise ValueError("mode must be 'pre' or 'post'")

    wanted = [c for c in wanted if c in df.columns]
    used = list(dict.fromkeys(wanted))

    X = df[used].copy()
    y = df[TARGET_COL].astype(int).to_numpy()
    return X, y, used


# ----------------------------
# Preprocess
# ----------------------------

def build_preprocess(X: pd.DataFrame) -> ColumnTransformer:
    """Impute missing values + one-hot encode categorical columns."""
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )


# ----------------------------
# Parameter grids
# ----------------------------

def get_param_grid_quick() -> Dict[str, Any]:
    """
    Grid RÁPIDO - testa apenas configurações essenciais.
    ~54 combinações = 270 modelos (5-fold CV)
    Tempo estimado: 10-30 minutos
    """
    return {
        "clf__strategy": ["none", "oversample"],  # 2 opções
        "clf__n_estimators": [200, 500],  # 2 opções
        "clf__max_depth": [None, 20],  # 2 opções
        "clf__max_features": ["sqrt", 0.5],  # 2 opções
        "clf__min_samples_split": [2, 10],  # 2 opções
        "clf__min_samples_leaf": [1, 4],  # 2 opções
    }
    # Total: 2^6 = 64 combinações


def get_param_grid_medium_lite() -> Dict[str, Any]:
    """
    Grid MÉDIO-LITE - balanceado (MANTÉM undersample).
    ~108 combinações = 540 modelos (5-fold CV)
    Tempo estimado: 25-40 minutos
    
    Reduz n_estimators e max_features mas mantém as 3 estratégias.
    """
    return {
        "clf__strategy": ["none", "undersample", "oversample"],  # 3 - MANTIDO
        "clf__n_estimators": [200, 500],  # 2 - reduzido (era [200, 500, 1000])
        "clf__max_depth": [None, 20, 40],  # 3 - mantido
        "clf__max_features": ["sqrt"],  # 1 - reduzido (era ["sqrt", 0.5])
        "clf__min_samples_split": [2, 10],  # 2 - reduzido
        "clf__min_samples_leaf": [1, 2, 4],  # 3 - mantido
    }
    # Total: 3 * 2 * 3 * 1 * 2 * 3 = 108 combinações


def get_param_grid_medium() -> Dict[str, Any]:
    """
    Grid MÉDIO - busca mais completa (REDUZIDO mas mantém estrutura).
    ~162 combinações = 810 modelos (5-fold CV)
    Tempo estimado: 40-60 min
    
    Mantém as 3 estratégias e estrutura original, apenas reduz n_estimators.
    """
    return {
        "clf__strategy": ["none", "undersample", "oversample"],  # 3 - MANTIDO
        "clf__n_estimators": [200, 500],  # 2 - reduzido (era [200, 500, 1000])
        "clf__max_depth": [None, 20, 40],  # 3 - mantido
        "clf__max_features": ["sqrt", 0.5],  # 2 - mantido
        "clf__min_samples_split": [2, 10],  # 2 - reduzido (era [2, 10, 20])
        "clf__min_samples_leaf": [1, 2, 4],  # 3 - mantido
    }
    # Total: 3 * 2 * 3 * 2 * 2 * 3 = 216 combinações
    # (antes era 162, mas sem n_estimators=1000 fica mais rápido)


def get_param_grid_full() -> Dict[str, Any]:
    """
    Grid COMPLETO - busca extensiva (CUIDADO: pode levar muitas horas).
    ~729 combinações = 3.645 modelos (5-fold CV)
    """
    return {
        "clf__strategy": ["none", "undersample", "oversample"],
        "clf__n_estimators": [200, 500, 1000],
        "clf__max_depth": [None, 20, 40],
        "clf__max_features": ["sqrt", "log2", 0.5],
        "clf__min_samples_split": [2, 10, 20],
        "clf__min_samples_leaf": [1, 2, 4],
    }


def get_param_distributions() -> Dict[str, Any]:
    """
    Distribuições para RandomizedSearchCV (muito mais rápido).
    Permite testar ranges contínuos sem explodir o espaço de busca.
    """
    from scipy.stats import randint, uniform
    
    return {
        "clf__strategy": ["none", "undersample", "oversample"],
        "clf__n_estimators": [100, 200, 300, 500, 700, 1000],
        "clf__max_depth": [None, 10, 20, 30, 40, 50],
        "clf__max_features": ["sqrt", "log2", 0.3, 0.5, 0.7],
        "clf__min_samples_split": randint(2, 30),
        "clf__min_samples_leaf": randint(1, 10),
    }


# ----------------------------
# Reporting
# ----------------------------

def best_per_strategy_from_cv_results(cv_results: pd.DataFrame) -> pd.DataFrame:
    """Returns the best CV row per balancing strategy."""
    if "param_clf__strategy" not in cv_results.columns:
        return pd.DataFrame()

    keep_cols = [
        "param_clf__strategy",
        "mean_test_score",
        "std_test_score",
        "rank_test_score",
        "param_clf__n_estimators",
        "param_clf__max_depth",
        "param_clf__max_features",
        "param_clf__min_samples_split",
        "param_clf__min_samples_leaf",
    ]
    keep_cols = [c for c in keep_cols if c in cv_results.columns]
    compact = cv_results[keep_cols].copy()

    best_rows = (
        compact.sort_values(["param_clf__strategy", "mean_test_score"], ascending=[True, False])
        .groupby("param_clf__strategy", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    return best_rows


def export_experiment_artifacts(
    out_dir: str,
    mode: str,
    best_params: Dict[str, Any],
    best_cv_score: float,
    test_metrics: Dict[str, Any],
    features_used: List[str],
    cv_results: pd.DataFrame,
    best_per_strategy: pd.DataFrame,
    model: Pipeline,
    search_method: str,
    elapsed_time: float,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # Save artifacts
    pd.Series(features_used, name="feature").to_csv(
        os.path.join(out_dir, "features_used.csv"), index=False
    )
    cv_results.to_csv(os.path.join(out_dir, "cv_results.csv"), index=False)
    best_per_strategy.to_csv(os.path.join(out_dir, "cv_best_per_strategy.csv"), index=False)
    dump(model, os.path.join(out_dir, "best_model.joblib"))

    # JSON summary
    payload = {
        "feature_set": mode,
        "n_features": int(len(features_used)),
        "search_method": search_method,
        "elapsed_time_minutes": round(elapsed_time / 60, 2),
        "best_cv_f1_macro": float(best_cv_score),
        "best_params": best_params,
        "test_metrics": {
            "test_f1_macro": float(test_metrics["test_f1_macro"]),
            "test_balanced_accuracy": float(test_metrics["test_balanced_accuracy"]),
        },
        "artifacts_dir": out_dir,
    }
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # Text report
    report_lines: List[str] = []
    report_lines.append(f"Feature set: {mode}")
    report_lines.append(f"Search method: {search_method}")
    report_lines.append(f"Number of features: {len(features_used)}")
    report_lines.append(f"Elapsed time: {elapsed_time/60:.2f} minutes")
    report_lines.append(f"Best CV macro-F1: {best_cv_score:.6f}")
    report_lines.append(f"Best params: {best_params}")
    report_lines.append("")
    report_lines.append("Test metrics:")
    report_lines.append(f"- test_f1_macro: {test_metrics['test_f1_macro']:.6f}")
    report_lines.append(f"- test_balanced_accuracy: {test_metrics['test_balanced_accuracy']:.6f}")
    report_lines.append("")
    report_lines.append("Best per strategy (CV):")
    report_lines.append(best_per_strategy.to_string(index=False))
    report_lines.append("")
    report_lines.append("Classification report:")
    report_lines.append(test_metrics["classification_report"])
    report_lines.append("")
    report_lines.append("Confusion matrix:")
    report_lines.append(np.array2string(np.array(test_metrics["confusion_matrix"])))

    with open(os.path.join(out_dir, "REPORT.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))


# ----------------------------
# Main evaluation
# ----------------------------

def evaluate_rf_pre_post(
    data_path: str,
    pre_cols: List[str],
    post_only_cols: List[str],
    results_root: str = "outputs_rf",
    random_state: int = 42,
    test_size: float = 0.2,
    cv_splits: int = 5,
    search_method: str = "quick",  # "quick" | "medium" | "full" | "random"
    n_iter_random: int = 50,  # para RandomizedSearchCV
) -> None:
    """
    Main evaluation function.
    
    search_method:
        - "quick": Grid pequeno, ~64 combinações (~15-30 min)
        - "medium-lite": Grid balanceado, ~108 combinações (~25-40 min) - MANTÉM 3 estratégias
        - "medium": Grid médio, ~216 combinações (~40-60 min) - Estrutura original completa
        - "full": Grid completo, ~729 combinações (MUITAS horas)
        - "random": RandomizedSearchCV, testa n_iter_random amostras (RECOMENDADO para exploração rápida)
    """
    os.makedirs(results_root, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"INICIANDO EXPERIMENTO RF")
    print(f"Search method: {search_method}")
    print(f"{'='*80}\n")

    df = pd.read_csv(data_path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in {data_path}")

    # Fixed stratified split
    y_all = df[TARGET_COL].astype(int).to_numpy()
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, y_all))

    summary_rows: List[Dict[str, Any]] = []
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    # Select param grid based on method
    if search_method == "quick":
        param_grid = get_param_grid_quick()
        n_combinations = np.prod([len(v) for v in param_grid.values()])
        print(f"Grid QUICK: ~{n_combinations} combinações\n")
    elif search_method == "medium-lite":
        param_grid = get_param_grid_medium_lite()
        n_combinations = np.prod([len(v) for v in param_grid.values()])
        print(f"Grid MEDIUM-LITE: ~{n_combinations} combinações\n")
    elif search_method == "medium":
        param_grid = get_param_grid_medium()
        n_combinations = np.prod([len(v) for v in param_grid.values()])
        print(f"Grid MEDIUM: ~{n_combinations} combinações\n")
    elif search_method == "full":
        param_grid = get_param_grid_full()
        n_combinations = np.prod([len(v) for v in param_grid.values()])
        print(f"⚠️  Grid FULL: ~{n_combinations} combinações (PODE LEVAR MUITAS HORAS)\n")
    elif search_method == "random":
        param_grid = get_param_distributions()
        print(f"RandomizedSearchCV: {n_iter_random} iterações aleatórias\n")
    else:
        raise ValueError(f"search_method inválido: {search_method}")

    for mode in ["pre", "post"]:
        print(f"\n{'='*80}")
        print(f"Feature set: {mode.upper()}")
        print(f"{'='*80}\n")
        
        # Check if already completed (checkpoint)
        checkpoint_file = os.path.join(results_root, mode, "results.json")
        if os.path.exists(checkpoint_file):
            print(f"⚠️  {mode.upper()} já foi concluído anteriormente!")
            print(f"   Arquivo encontrado: {checkpoint_file}")
            print(f"   Pulando para o próximo...\n")
            
            # Load previous results for summary
            with open(checkpoint_file, 'r') as f:
                prev_results = json.load(f)
            summary_rows.append({
                "feature_set": mode,
                "n_features": prev_results['n_features'],
                "search_method": prev_results['search_method'],
                "elapsed_time_min": prev_results['elapsed_time_minutes'],
                "best_cv_f1_macro": prev_results['best_cv_f1_macro'],
                "best_strategy": prev_results['best_params'].get('clf__strategy', None),
                "test_f1_macro": prev_results['test_metrics']['test_f1_macro'],
                "test_balanced_accuracy": prev_results['test_metrics']['test_balanced_accuracy'],
                "artifacts_dir": os.path.join(results_root, mode),
            })
            continue
        
        start_time = time.time()
        
        X, y, used_cols = select_features(df, mode, pre_cols, post_only_cols)
        print(f"Features selecionadas: {len(used_cols)}")
        print(f"Samples treino: {len(train_idx)}, teste: {len(test_idx)}\n")

        X_train = X.iloc[train_idx].reset_index(drop=True)
        y_train = y[train_idx]
        X_test = X.iloc[test_idx].reset_index(drop=True)
        y_test = y[test_idx]

        preprocess = build_preprocess(X_train)

        pipe = Pipeline(
            steps=[
                ("preprocess", preprocess),
                ("clf", ResampledRF(strategy="none", random_state=random_state)),
            ]
        )

        # Choose search type
        if search_method == "random":
            searcher = RandomizedSearchCV(
                estimator=pipe,
                param_distributions=param_grid,
                n_iter=n_iter_random,
                scoring="f1_macro",
                cv=cv,
                n_jobs=1,
                verbose=2,
                refit=True,
                random_state=random_state,
            )
        else:
            searcher = GridSearchCV(
                estimator=pipe,
                param_grid=param_grid,
                scoring="f1_macro",
                cv=cv,
                n_jobs=1,
                verbose=2,
                refit=True,
            )

        print(f"Iniciando busca de hiperparâmetros...")
        searcher.fit(X_train, y_train)
        
        elapsed = time.time() - start_time
        print(f"\n✓ Busca concluída em {elapsed/60:.2f} minutos")

        best_model = searcher.best_estimator_
        best_params = searcher.best_params_
        best_cv = float(searcher.best_score_)

        print(f"Best CV F1-macro: {best_cv:.4f}")
        print(f"Best params: {best_params}\n")

        y_pred = best_model.predict(X_test)

        test_metrics = {
            "test_f1_macro": float(f1_score(y_test, y_pred, average="macro")),
            "test_balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
            "classification_report": classification_report(y_test, y_pred, digits=4),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }

        print(f"Test F1-macro: {test_metrics['test_f1_macro']:.4f}")
        print(f"Test Balanced Acc: {test_metrics['test_balanced_accuracy']:.4f}\n")

        cv_results = pd.DataFrame(searcher.cv_results_)
        best_strategy_df = best_per_strategy_from_cv_results(cv_results)

        exp_dir = os.path.join(results_root, mode)
        export_experiment_artifacts(
            out_dir=exp_dir,
            mode=mode,
            best_params=best_params,
            best_cv_score=best_cv,
            test_metrics=test_metrics,
            features_used=used_cols,
            cv_results=cv_results,
            best_per_strategy=best_strategy_df,
            model=best_model,
            search_method=search_method,
            elapsed_time=elapsed,
        )

        summary_rows.append(
            {
                "feature_set": mode,
                "n_features": int(len(used_cols)),
                "search_method": search_method,
                "elapsed_time_min": round(elapsed / 60, 2),
                "best_cv_f1_macro": best_cv,
                "best_strategy": best_params.get("clf__strategy", None),
                "test_f1_macro": test_metrics["test_f1_macro"],
                "test_balanced_accuracy": test_metrics["test_balanced_accuracy"],
                "artifacts_dir": exp_dir,
            }
        )
        
        # CRITICAL: Free memory before next iteration
        print(f"💾 Liberando memória...")
        del X, y, X_train, X_test, y_train, y_test
        del preprocess, pipe, searcher, best_model
        del cv_results, best_strategy_df
        import gc
        gc.collect()
        print(f"✓ Memória liberada\n")

    # Final summary
    summary_df = pd.DataFrame(summary_rows).sort_values("test_f1_macro", ascending=False)
    summary_df.to_csv(os.path.join(results_root, "SUMMARY.csv"), index=False)

    with open(os.path.join(results_root, "REPORT_GLOBAL.txt"), "w", encoding="utf-8") as f:
        f.write("Random Forest Experiments (PRE vs POST)\n\n")
        f.write(f"Dataset: {data_path}\n")
        f.write(f"Target: {TARGET_COL}\n")
        f.write(f"Search method: {search_method}\n")
        f.write(f"CV splits: {cv_splits}\n")
        f.write(f"Test size: {test_size}\n\n")
        f.write(summary_df.to_string(index=False))
        f.write("\n")

    print(f"\n{'='*80}")
    print("EXPERIMENTO CONCLUÍDO!")
    print(f"Resultados salvos em: {results_root}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    DATA_PATH = "../data/PREPROCESSED/Final_TtST_Dataset.csv"

    PRE_COLS = [
        "pr_number",
        "CONT_repo",
        "CONT_label_count",
        "CONT_day_of_week",
        "CONT_weekday",
        "CONT_PR_text_wordiness",
        "CODE_number_of_commits",
        "CODE_lines_of_code_changed",
        "CODE_number_of_files_changed",
        "CODE_number_of_milestones",
        "ORG_author_PRs_opened",
        "ORG_PRs_opened_in_last_2_weeks",
        "ORG_PRs_closed_in_last_2_weeks",
        "ORG_open_PRs_at_open_date",
        "ACT_time_since_last_commit",
        "ACT_number_of_linked_issues",
    ]

    POST_ONLY_COLS = [
        "CODE_number_of_revisions",
        "ACT_number_of_comments",
        "ACT_number_of_review_comments",
        "ACT_number_of_reviewers",
        "ACT_number_of_approvals",
        "ACT_time_to_first_response",
        "ACT_number_of_assignees",
        "ACT_number_of_changes_requested",
        "ACT_number_of_reviews_requested",
    ]

    # ============================================================================
    # ESCOLHA O MÉTODO DE BUSCA AQUI:
    # ============================================================================
    # 
    # OPÇÃO 1 - RÁPIDO (RECOMENDADO PARA COMEÇAR): 15-30 minutos
    # SEARCH_METHOD = "quick"
    
    # OPÇÃO 2 - MUITO RÁPIDO (EXPLORAÇÃO): 10-20 minutos, 50 combinações aleatórias
    SEARCH_METHOD = "random"
    
    # OPÇÃO 3 - BALANCEADO (MANTÉM undersample): 25-40 minutos, 108 combinações
    # SEARCH_METHOD = "medium-lite"
    
    # OPÇÃO 4 - MÉDIO (Estrutura original): 40-60 min, 216 combinações
    # SEARCH_METHOD = "medium"
    
    # OPÇÃO 5 - COMPLETO (CUIDADO!): Muitas horas, pode travar
    # SEARCH_METHOD = "full"
    # ============================================================================

    evaluate_rf_pre_post(
        data_path=DATA_PATH,
        pre_cols=PRE_COLS,
        post_only_cols=POST_ONLY_COLS,
        results_root="outputs_rf_optimized",
        random_state=42,
        test_size=0.2,
        cv_splits=5,
        search_method=SEARCH_METHOD,
        n_iter_random=50,  # usado apenas se SEARCH_METHOD="random"
    )