from typing import Dict, List, Tuple, Any, Optional
import time
import json
import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, log_loss, roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from vgc_matchup_predictor.utils.config import load_config
from vgc_matchup_predictor.visualizers.vizualizers import *
from vgc_matchup_predictor.data.process_json import lookup, calc_type_matchup, calc_attack_ratio

config = load_config('data_config.yaml')
LEAD_POSITIONS = ('p1a', 'p1b', 'p2a', 'p2b')
P1_LEADS = ('p1a', 'p1b')
P2_LEADS = ('p2a', 'p2b')
MATCHUP_FEATURE_NAMES = [
    *(f'{pos}_speed' for pos in LEAD_POSITIONS),
    *(f'{atk}_vs_{dfn}_type' for atk in P1_LEADS for dfn in P2_LEADS),
    *(f'{atk}_vs_{dfn}_type' for atk in P2_LEADS for dfn in P1_LEADS),
    *(f'{atk}_vs_{dfn}_atk_ratio' for atk in P1_LEADS for dfn in P2_LEADS),
    *(f'{atk}_vs_{dfn}_atk_ratio' for atk in P2_LEADS for dfn in P1_LEADS),
]

class VGCRandomForest:
    # Random Forest for battle prediction using differential multi-hot encoding

    def __init__(
        self,
        n_estimators: int = 200,    # hyperparams
        max_depth: Optional[int] = 15,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Any = 'sqrt',
        class_weight: Optional[Any] = None,
        random_state: int = 42,
        n_jobs: int = -1,           # use all available CPU cores
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.mlb = MultiLabelBinarizer()
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self.is_fitted = False
        self.X_train = None
        self.y_train = None

    @staticmethod
    def extract_team(battle: Dict, player_prefix: str) -> List[str]:
        # Extracts the 6 Pokemon on a player's team (leads + back)
        lead_a = battle.get(f'{player_prefix}a')
        lead_b = battle.get(f'{player_prefix}b')
        back = battle.get(f'{player_prefix}_back', [])

        team = []
        if lead_a:
            team.append(lead_a)
        if lead_b:
            team.append(lead_b)
        team.extend(back)
        return team

    @staticmethod
    def extract_matchup_features(battle: Dict) -> List[float]:
        features: List[float] = []

        # base speed for each lead pokemon; 4 features
        for pos in LEAD_POSITIONS:
            mon = battle.get(pos)
            if mon:
                speed = lookup(mon, 'speed')
                features.append(float(speed) if speed not in ('', None) else 0.0)
            else:
                features.append(0.0)

        # directional lead type matchups; 8 features
        for atk_pos in P1_LEADS:
            for def_pos in P2_LEADS:
                atk, dfn = battle.get(atk_pos), battle.get(def_pos)
                features.append(float(calc_type_matchup(atk, dfn)) if atk and dfn else 0.0)

        for atk_pos in P2_LEADS:
            for def_pos in P1_LEADS:
                atk, dfn = battle.get(atk_pos), battle.get(def_pos)
                features.append(float(calc_type_matchup(atk, dfn)) if atk and dfn else 0.0)

        # directional lead attack/defense ratios; 8 features
        for atk_pos in P1_LEADS:
            for def_pos in P2_LEADS:
                atk, dfn = battle.get(atk_pos), battle.get(def_pos)
                features.append(float(calc_attack_ratio(atk, dfn)) if atk and dfn else 0.0)

        for atk_pos in P2_LEADS:
            for def_pos in P1_LEADS:
                atk, dfn = battle.get(atk_pos), battle.get(def_pos)
                features.append(float(calc_attack_ratio(atk, dfn)) if atk and dfn else 0.0)

        return features

    def prepare_features(self, battles: List[Dict], is_train: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        # Convert the processed battle logs into differential multi-hot + matchup feature matrices
        p1_teams = [self.extract_team(b, 'p1') for b in battles]
        p2_teams = [self.extract_team(b, 'p2') for b in battles]

        if is_train:
            all_teams = p1_teams + p2_teams
            self.mlb.fit(all_teams)

        X_p1 = self.mlb.transform(p1_teams)
        X_p2 = self.mlb.transform(p2_teams)

        # Difference vector: P1 presence (+1) vs P2 presence (-1)
        X_roster = X_p1 - X_p2

        # Compute lead speed, type matchup, attack/defense ratio features
        if len(battles) > 0:
            X_matchup = np.array([self.extract_matchup_features(b) for b in battles], dtype=float)
            X = np.hstack([X_roster, X_matchup])
        else:
            X = np.empty((0, X_roster.shape[1] + len(MATCHUP_FEATURE_NAMES)), dtype=float)

        y = np.array([1 if b.get('winner') == 'p1' else 0 for b in battles])

        return X, y

    def fit(self, train_data: List[Dict]) -> None:
        # Trains the Random Forest model on training battle split
        X_train, y_train = self.prepare_features(train_data, is_train=True)
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, battle: Dict) -> str:
        # Predicts winner ('p1' or 'p2') for a single battle
        if not self.is_fitted:
            raise RuntimeError('Model must be fitted before running inference')
        X_battle, _ = self.prepare_features([battle], is_train=False)
        pred = self.model.predict(X_battle)[0]
        return 'p1' if pred == 1 else 'p2'

    def evaluate(self, test_data: List[Dict]) -> Dict[str, float]:
        # returns accuracy, precision, recall, and F1 score on out-of-sample test split
        if not self.is_fitted:
            raise RuntimeError('Model must be fitted before evaluation')

        if len(test_data) == 0:
            return {
                'test_samples': 0,
                'random_forest_accuracy': 0.0,
                'random_forest_precision': 0.0,
                'random_forest_recall': 0.0,
                'random_forest_f1': 0.0,
                'random_forest_log_loss': 0.0,
                'elapsed_time': 0.0
            }

        start = time.perf_counter()

        X_test, y_test = self.prepare_features(test_data, is_train=False)
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)

        stop = time.perf_counter()

        elapsed_time = stop - start

        # plots
        plot_confusion_matrix(X_test, y_test, self.model, "Random Forest Classifier")
        plot_roc_curve(X_test, y_test, self.model, "Random Forest Classifier")

        if self.X_train is not None and self.y_train is not None:
            plot_learning_curve_log_loss(self.model, self.X_train, self.y_train, X_test, y_test)

        importances = self.feature_importances()
        plot_feature_importances(importances, 15)

        return {
            'test_samples': len(test_data),
            'random_forest_accuracy': float(accuracy_score(y_test, y_pred)),
            'random_forest_precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'random_forest_recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'random_forest_f1': float(f1_score(y_test, y_pred, zero_division=0)),
            'random_forest_log_loss': float(log_loss(y_test, y_proba)),
            'evaluation_time_seconds': round(float(elapsed_time), 6)
        }

    def feature_importances(self) -> Dict[str, float]:
        # Returns species feature importances sorted from highest to lowest
        if not self.is_fitted:
            raise RuntimeError('Model must be fitted to get feature importances')

        importances = self.model.feature_importances_
        species_names = self.mlb.classes_

        importance_dict = {
            species: float(imp)
            for species, imp in zip(species_names, importances)
        }
        return dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))

def tune_hyperparameters(
    train_data: List[Dict],
    n_splits: int = 5,
    scoring: str = 'log_loss',
    n_candidates: int = 30,
    random_state: int = 42,
) -> Dict[str, Any]:
    # like Log. Reg., performs leak-free Stratified K-Fold CV on train_data to select optimal hyperparameters for VGCRandomForest
    if scoring not in ('log_loss', 'roc_auc', 'accuracy'):
        raise ValueError("scoring must be one of 'log_loss', 'roc_auc', or 'accuracy'")

    param_grid = {
        'n_estimators': [200, 400],
        'max_depth': [6, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 3, 5, 10],
        'max_features': ['sqrt', 0.2, 0.4],
        'class_weight': [None, 'balanced']
    }

    baseline_config = {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 'sqrt',
        'class_weight': None
    }

    all_configs = [
        cfg for cfg in ParameterGrid(param_grid) if cfg != baseline_config
    ]
    rng = random.Random(random_state)
    sampled_configs = rng.sample(all_configs, k=min(max(0, n_candidates - 1), len(all_configs)))
    candidate_params = [baseline_config] + sampled_configs

    y_all = np.array([1 if b.get('winner') == 'p1' else 0 for b in train_data])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = list(skf.split(np.zeros(len(train_data)), y_all))

    cv_records: List[Dict[str, Any]] = []
    start_time = time.perf_counter()

    for params in candidate_params:
        fold_log_losses: List[float] = []
        fold_aucs: List[float] = []
        fold_accs: List[float] = []

        for train_idx, val_idx in folds:
            fold_train = [train_data[i] for i in train_idx]
            fold_val = [train_data[i] for i in val_idx]

            # Fit MultiLabelBinarizer strictly inside each fold to prevent data leakage
            candidate_model = VGCRandomForest(
                **params,
                random_state=random_state,
                n_jobs=-1,
            )
            candidate_model.fit(fold_train)

            X_val, y_val = candidate_model.prepare_features(fold_val, is_train=False)
            y_pred = candidate_model.model.predict(X_val)
            y_proba = candidate_model.model.predict_proba(X_val)

            fold_log_losses.append(float(log_loss(y_val, y_proba)))
            fold_aucs.append(float(roc_auc_score(y_val, y_proba[:, 1])))
            fold_accs.append(float(accuracy_score(y_val, y_pred)))

        cv_records.append({
            'params': params,
            'mean_cv_log_loss': float(np.mean(fold_log_losses)),
            'mean_cv_roc_auc': float(np.mean(fold_aucs)),
            'mean_cv_accuracy': float(np.mean(fold_accs)),
        })

    if scoring == 'log_loss':
        cv_records.sort(key=lambda r: r['mean_cv_log_loss'])
    elif scoring == 'roc_auc':
        cv_records.sort(key=lambda r: r['mean_cv_roc_auc'], reverse=True)
    else:
        cv_records.sort(key=lambda r: r['mean_cv_accuracy'], reverse=True)

    best_record = cv_records[0]
    elapsed = time.perf_counter() - start_time

    return {
        'selection_metric': scoring,
        'n_splits': n_splits,
        'candidates_evaluated': len(candidate_params),
        'tuning_time_seconds': round(float(elapsed), 4),
        'best_params': best_record['params'],
        'best_cv_metrics': {
            'mean_cv_log_loss': best_record['mean_cv_log_loss'],
            'mean_cv_roc_auc': best_record['mean_cv_roc_auc'],
            'mean_cv_accuracy': best_record['mean_cv_accuracy'],
        },
        'top_5_configs': cv_records[:5],
    }

def run_test(
    reg_a_path: str = './data/processed/' + config['dataset']['m_a'],
    reg_b_path: str = './data/processed/' + config['dataset']['m_b'],
    train_ratio: float = 0.8,
    seed: int = 42,
    tune: bool = True
):
    random.seed(seed)

    # Load reg M-A and split 80/20
    with open(reg_a_path, 'r', encoding='utf-8') as f:
        reg_a_battles = json.load(f)

    random.shuffle(reg_a_battles)
    split_idx = int(len(reg_a_battles) * train_ratio)
    train_reg_a = reg_a_battles[:split_idx]
    test_reg_a = reg_a_battles[split_idx:]

    # Load reg M-B validation dataset
    with open(reg_b_path, 'r', encoding='utf-8') as f:
        val_reg_b = json.load(f)

    # tune hyperparams before building and fitting model
    if tune:
        tuning_results = tune_hyperparameters(
            train_reg_a,
            n_splits=5,
            scoring='log_loss',
            n_candidates=30,
            random_state=seed,
        )
        best_params = tuning_results['best_params']
        model = VGCRandomForest(**best_params, random_state=seed)
    else:
        tuning_results = None
        model = VGCRandomForest(random_state=seed)

    # Fit and evaluate Random Forest
    model.fit(train_reg_a)

    reg_a_metrics = model.evaluate(test_reg_a)
    reg_b_metrics = model.evaluate(val_reg_b)

    results = {
        'tuning_summary': tuning_results,
        'reg_a_in_distribution_test': reg_a_metrics,
        'reg_b_cross_regulation_validation': reg_b_metrics,
    }

    print('Random Forest Evaluation Results:')
    print(json.dumps(results, indent=2))

    # # Print top 15 most influential species
    # print("\nTop 10 Most Important Species Features:")
    # top_features = list(model.feature_importances().items())[:15]
    # for species, imp in top_features:
    #     print(f"  - {species}: {imp:.4f}")

if __name__ == '__main__':
    run_test()