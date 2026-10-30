from typing import Dict, List, Tuple
import json
import random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, log_loss
from vgc_matchup_predictor.utils.config import load_config

config = load_config('data_config.yaml')

class VGCLogisticRegression:
    # Logistic Regression baseline for battle prediction using difference multi-hot encoding

    def __init__(self, C: float = 1.0, max_iter: int = 1000, random_state: int = 42):
        self.mlb = MultiLabelBinarizer()
        self.model = LogisticRegression(C=C, max_iter=max_iter, random_state=random_state)
        self.is_fitted = False

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
        X = X_p1 - X_p2
        y = np.array([1 if b.get('winner') == 'p1' else 0 for b in battles])

        return X, y

    def fit(self, train_data: List[Dict]) -> None:
        # Trains the Logistic Regression model on training battle split
        X_train, y_train = self.prepare_features(train_data, is_train=True)
        self.model.fit(X_train, y_train)
        self.is_fitted = True

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
                'logistic_regression_log_loss': 0.0
            }

        X_test, y_test = self.prepare_features(test_data, is_train=False)
        y_pred = self.model.predict(X_test)         # discrete probabilities ('p1' or 'p2')
        y_proba = self.model.predict_proba(X_test)  # continuous probabilities (P(p1), P(p2)); required for log loss function

        return {
            'test_samples': len(test_data),
            'logistic_regression_accuracy': float(accuracy_score(y_test, y_pred)),
            'logistic_regression_precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'logistic_regression_recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'logistic_regression_f1': float(f1_score(y_test, y_pred, zero_division=0)),
            'logistic_regression_log_loss': float(log_loss(y_test, y_proba))
        }

def run_test(
    reg_a_path: str = './data/processed/' + config['dataset']['m_a'],
    reg_b_path: str = './data/processed/' + config['dataset']['m_b'],
    train_ratio: float = 0.8,
    seed: int = 42,
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

    # Fit and evaluate Logistic Regression
    model = VGCLogisticRegression(random_state=seed)
    model.fit(train_reg_a)

    reg_a_metrics = model.evaluate(test_reg_a)
    reg_b_metrics = model.evaluate(val_reg_b)

    results = {
        'reg_a_in_distribution_test': reg_a_metrics,
        'reg_b_cross_regulation_validation': reg_b_metrics,
    }

    print('Logistic Regression Evaluation Results:')
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    run_test()