from typing import Dict, List, Tuple, Any, Optional
import json
import random
import numpy as np
import time
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from vgc_matchup_predictor.utils.config import load_config
from vgc_matchup_predictor.visualizers.vizualizers import plot_confusion_matrix, plot_roc_curve, plot_learning_curve_log_loss
from vgc_matchup_predictor.data.process_json import lookup, calc_type_matchup, calc_attack_ratio

config = load_config('data_config.yaml')
LEAD_POSITIONS = ('p1a', 'p1b', 'p2a', 'p2b')
P1_LEADS = ('p1a', 'p1b')
P2_LEADS = ('p2a', 'p2b')
SKLEARN_VERSION = tuple(int(part) for part in sklearn.__version__.split('.')[:2])

class VGCLogisticRegression:
    # Logistic Regression baseline for battle prediction using difference multi-hot encoding
    # augmented with lead speed, type matchup, atk/def ratio features

    def __init__(
            self,
            C: float = 0.01,
            l1_ratio: float = 0.0,
            solver: str = 'lbfgs',
            class_weight: Optional[Any] = None,
            penalty: Optional[str] = None,
            max_iter: int = 1000,
            random_state: int = 42
        ):
        # Support legacy penalty strings ('l1' / 'l2') while avoiding sklearn >= 1.8 deprecation warnings
        if penalty == 'l1':
            l1_ratio = 1.0
        elif penalty == 'l2':
            l1_ratio = 0.0

        self.C = C
        self.l1_ratio = l1_ratio
        self.solver = solver
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.random_state = random_state
        self.mlb = MultiLabelBinarizer()
        self.scaler = StandardScaler()
        self.model = LogisticRegression(C=C, max_iter=max_iter, random_state=random_state)
        self.is_fitted = False
        self.X_train = None # store vectorized training data to help with plotting later
        self.y_train = None

    def build_estimator(self) -> LogisticRegression:
        if SKLEARN_VERSION >= (1, 8):
            return LogisticRegression(
                C=self.C,
                l1_ratio=self.l1_ratio,
                solver=self.solver,
                class_weight=self.class_weight,
                max_iter=self.max_iter,
                random_state=self.random_state,
            )

        if self.l1_ratio == 1.0:
            legacy_penalty = 'l1'
            extra_kwargs = {}
        elif self.l1_ratio == 0.0:
            legacy_penalty = 'l2'
            extra_kwargs = {}
        else:
            legacy_penalty = 'elasticnet'
            extra_kwargs = {'l1_ratio': self.l1_ratio}

        return LogisticRegression(
            penalty=legacy_penalty,
            C=self.C,
            solver=self.solver,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            random_state=self.random_state,
            **extra_kwargs,
        )

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
        # extracts numeric lead-matchup features using lookup, calc_type_matchup, calc_attack_ratio
        features: List[float] = []

        # base speed for each lead pokemon; 4 features
        for pos in LEAD_POSITIONS:
            mon = battle.get(pos)
            if mon:
                speed = lookup(mon, 'speed')
                features.append(float(speed) if speed not in ('', None) else 0.0)
            else:
                features.append(0.0)

        # directional lead type matchups; P1->P2 & P2->P1; 8 features

        # P1->P2
        for atk_pos in P1_LEADS:
            for dfn_pos in P2_LEADS:
                atk, dfn = battle.get(atk_pos), battle.get(dfn_pos)
                features.append(float(calc_type_matchup(atk, dfn)) if atk and dfn else 0.0)

        # P2->P1
        for atk_pos in P2_LEADS:
            for dfn_pos in P1_LEADS:
                atk, dfn = battle.get(atk_pos), battle.get(dfn_pos)
                features.append(float(calc_type_matchup(atk, dfn)) if atk and dfn else 0.0)

        # directional lead attack/defense ratios: P1->P2 & P2->P1; 8 features
        
        # P1->P2
        for atk_pos in P1_LEADS:
            for dfn_pos in P2_LEADS:
                atk, dfn = battle.get(atk_pos), battle.get(dfn_pos)
                features.append(float(calc_attack_ratio(atk, dfn)) if atk and dfn else 0.0)

        # P2->P1
        for atk_pos in P2_LEADS:
            for dfn_pos in P1_LEADS:
                atk, dfn = battle.get(atk_pos), battle.get(dfn_pos)
                features.append(float(calc_attack_ratio(atk, dfn)) if atk and dfn else 0.0)

        return features

    def prepare_features(self, battles: List[Dict], is_train: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        # Converts raw battle logs into differential multi-hot feature matrices
        p1_teams = [self.extract_team(b, 'p1') for b in battles]
        p2_teams = [self.extract_team(b, 'p2') for b in battles]

        if is_train:
            # Build vocabulary from all training rosters
            all_teams = p1_teams + p2_teams
            self.mlb.fit(all_teams)

        # Multi-hot encode rosters (unseen species during transform are safely ignored)
        X_p1 = self.mlb.transform(p1_teams)
        X_p2 = self.mlb.transform(p2_teams)

        # Compute difference vector: P1 presence (+1) vs P2 presence (-1)
        X_roster = X_p1 - X_p2

        # Compute scale and lead speed, type matchup, and attack ratio features
        if len(battles) > 0:
            X_matchup = np.array([self.extract_matchup_features(b) for b in battles], dtype=float)
            if is_train:
                X_matchup = self.scaler.fit_transform(X_matchup)
            else:
                X_matchup = self.scaler.transform(X_matchup)
            X = np.hstack([X_roster, X_matchup])
        else:
            X = np.empty((0, X_roster.shape[1] + 20), dtype=float)

        y = np.array([1 if b.get('winner') == 'p1' else 0 for b in battles])

        return X, y

    def fit(self, train_data: List[Dict]) -> None:
        # Trains the Logistic Regression model on training battle split
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
        # Evaluates accuracy, precision, recall, f1-score, log loss on out-of-sample test split
        if not self.is_fitted:
            raise RuntimeError('Model must be fitted before evaluation')

        if len(test_data) == 0:
            return {
                'test_samples': 0,
                'logistic_regression_accuracy': 0.0,
                'logistic_regression_precision': 0.0,
                'logistic_regression_recall': 0.0,
                'logistic_regression_f1': 0.0,
                'logistic_regression_log_loss': 0.0,
                'evaluation_time_seconds': 0.0
            }

        start = time.perf_counter()                 # start time

        X_test, y_test = self.prepare_features(test_data, is_train=False)
        y_pred = self.model.predict(X_test)         # discrete probabilities ('p1' or 'p2')
        y_proba = self.model.predict_proba(X_test)  # continuous probabilities (P(p1), P(p2)); required for log loss function

        stop = time.perf_counter()                  # stop time

        elapsed_time = stop - start

        # plots
        plot_confusion_matrix(X_test, y_test, self.model, 'Logistic Regression')
        plot_roc_curve(X_test, y_test, self.model, 'Logistic Regression')

        if self.X_train is not None and self.y_train is not None:
            plot_learning_curve_log_loss(self.model, self.X_train, self.y_train, X_test, y_test)

        return {
            'test_samples': len(test_data),
            'logistic_regression_accuracy': float(accuracy_score(y_test, y_pred)),
            'logistic_regression_precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'logistic_regression_recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'logistic_regression_f1': float(f1_score(y_test, y_pred, zero_division=0)),
            'logistic_regression_log_loss': float(log_loss(y_test, y_proba)),
            'evaluation_time_seconds': round(float(elapsed_time), 6)
        }

def tune_hyperparameters(
    train_data: List[Dict],
    n_splits: int = 5,
    scoring: str = 'log_loss',
    random_state: int = 42,
) -> Dict[str, Any]:
    # leak-free Stratified K-Fold cross validation on train_data to select optimal hyperparameters for Log. Reg. model (K = 5)
    if scoring not in ('log_loss', 'roc_auc', 'accuracy'):
        raise ValueError("scoring must be one of 'log_loss', 'roc_auc', or 'accuracy'")

    # Candidate hyperparameter combinations (L2 via lbfgs, L1 via liblinear)
    candidate_params: List[Dict[str, Any]] = []
    for C in (0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0):
        for class_weight in (None, 'balanced'):
            candidate_params.append({
                'C': C,
                'l1_ratio': 0.0,
                'solver': 'lbfgs',
                'class_weight': class_weight,
            })

    for C in (0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0):
        for class_weight in (None, 'balanced'):
            candidate_params.append({
                'C': C,
                'l1_ratio': 1.0,
                'solver': 'liblinear',
                'class_weight': class_weight,
            })

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

            # Fit MLB and StandardScaler strictly inside each fold to prevent data leakage
            candidate_model = VGCLogisticRegression(
                **params,
                max_iter=1000,
                random_state=random_state,
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
    tune: bool = False
):
    # standalone load, split, fit, and evaluation for Logistic Regression
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

    # use tuning to determine hyperparameters to build model with
    if tune:
        tuning_results = tune_hyperparameters(
            train_reg_a,
            n_splits=5,
            scoring='log_loss',
            random_state=seed
        )
        best_params = tuning_results['best_params']
        model = VGCLogisticRegression(**best_params, random_state=seed)
    else:
        tuning_results = None
        model = VGCLogisticRegression(random_state=seed)

    # Fit and evaluate Logistic Regression
    model.fit(train_reg_a)

    reg_a_metrics = model.evaluate(test_reg_a)
    reg_b_metrics = model.evaluate(val_reg_b)

    results = {
        'tuning_summary': tuning_results,
        'reg_a_in_distribution_test': reg_a_metrics,
        'reg_b_cross_regulation_validation': reg_b_metrics,
    }

    print('Logistic Regression Evaluation Results:')
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    run_test()