from typing import Dict, List, Tuple
import json
import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer
from vgc_matchup_predictor.utils.config import load_config

config = load_config('data_config.yaml')

class VGCRandomForest:
    # Random Forest for battle prediction using differential multi-hot encoding

    def __init__(
        self,
        n_estimators: int = 200,    # hyperparams
        max_depth: int = 15,        #  
        random_state: int = 42,     # 
        n_jobs: int = -1,           # use all available CPU cores
    ):
        self.mlb = MultiLabelBinarizer()
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=n_jobs,
        )
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
        # Convert the processed battle logs into differential multi-hot feature matrices
        p1_teams = [self.extract_team(b, 'p1') for b in battles]
        p2_teams = [self.extract_team(b, 'p2') for b in battles]

        if is_train:
            all_teams = p1_teams + p2_teams
            self.mlb.fit(all_teams)

        X_p1 = self.mlb.transform(p1_teams)
        X_p2 = self.mlb.transform(p2_teams)

        # Difference vector: P1 presence (+1) vs P2 presence (-1)
        X = X_p1 - X_p2
        y = np.array([1 if b.get('winner') == 'p1' else 0 for b in battles])

        return X, y

    def fit(self, train_data: List[Dict]) -> None:
        # Trains the Random Forest model on training battle split
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
            }

        X_test, y_test = self.prepare_features(test_data, is_train=False)
        y_pred = self.model.predict(X_test)

        return {
            'test_samples': len(test_data),
            'random_forest_accuracy': float(accuracy_score(y_test, y_pred)),
            'random_forest_precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'random_forest_recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'random_forest_f1': float(f1_score(y_test, y_pred, zero_division=0)),
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


def run_test(
    reg_a_path: str = './data/processed/' + config['dataset']['m_a'],
    reg_b_path: str = './data/processed/' + config['dataset']['m_b'],
    train_ratio: float = 0.8,
    seed: int = 42,
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

    # Fit and evaluate Random Forest
    model = VGCRandomForest(random_state=seed)
    model.fit(train_reg_a)

    reg_a_metrics = model.evaluate(test_reg_a)
    reg_b_metrics = model.evaluate(val_reg_b)

    results = {
        'reg_a_in_distribution_test': reg_a_metrics,
        'reg_b_cross_regulation_validation': reg_b_metrics,
    }

    print('Random Forest Evaluation Results:')
    print(json.dumps(results, indent=2))

    # Print top 10 most influential species
    print("\nTop 10 Most Important Species Features:")
    top_features = list(model.feature_importances().items())[:10]
    for species, imp in top_features:
        print(f"  - {species}: {imp:.4f}")

if __name__ == '__main__':
    run_test()