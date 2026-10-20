import json
import random
from collections import defaultdict
from typing import Dict, List
from vgc_matchup_predictor.utils.config import load_config

config = load_config('data_config.yaml')

class VGCNaiveBenchmark:
    # Naive baseline models for VGC battle outcome prediction
    # processes JSON battle logs matching Regulation M-A / M-B formats

    def __init__(self, data_path: str = None, raw_data: List[Dict] = None):
        if raw_data is not None:
            self.data = raw_data
        elif data_path:
            with open(data_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        else:
            self.data = []

        self.pokemon_winrates: Dict[str, float] = {}
        self.majority_winner: str = 'p1'

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

    def fit(self, train_data: List[Dict]) -> None:
        # Trains the naive baselines on the training split
        p1_wins = sum(1 for b in train_data if b.get('winner') == 'p1')
        p2_wins = sum(1 for b in train_data if b.get('winner') == 'p2')
        self.majority_winner = 'p1' if p1_wins >= p2_wins else 'p2'

        pokemon_appearances = defaultdict(int)
        pokemon_wins = defaultdict(int)

        for battle in train_data:
            winner = battle.get('winner')
            p1_team = self.extract_team(battle, 'p1')
            p2_team = self.extract_team(battle, 'p2')

            for pokemon in p1_team:
                pokemon_appearances[pokemon] += 1
                if winner == 'p1':
                    pokemon_wins[pokemon] += 1

            for pokemon in p2_team:
                pokemon_appearances[pokemon] += 1
                if winner == 'p2':
                    pokemon_wins[pokemon] += 1

        # Calculate individual win rates per Pokemon
        for pokemon, count in pokemon_appearances.items():
            self.pokemon_winrates[pokemon] = pokemon_wins[pokemon] / count

    def predict_majority(self, battle: Dict) -> str:
        # Baseline 1: Always predict the majority winner class
        return self.majority_winner

    def predict_winrate_sum(self, battle: Dict) -> str:
        # Baseline 2: Predict winner based on higher cumulative team Pokemon win rates
        p1_team = self.extract_team(battle, "p1")
        p2_team = self.extract_team(battle, "p2")

        p1_score = sum(self.pokemon_winrates.get(pokemon, 0.5) for pokemon in p1_team)
        p2_score = sum(self.pokemon_winrates.get(pokemon, 0.5) for pokemon in p2_team)

        if p1_score == p2_score:
            return self.majority_winner
        return 'p1' if p1_score > p2_score else 'p2'

    def evaluate(self, test_data: List[Dict]) -> Dict[str, float]:
        # Evaluates majority class and winrate-sum heuristics on test split
        correct_majority = 0
        correct_winrate = 0
        total = len(test_data)

        if total == 0:
            return {'majority_acc': 0.0, 'winrate_sum_acc': 0.0, 'test_samples': 0}

        for battle in test_data:
            actual = battle.get('winner')

            if self.predict_majority(battle) == actual:
                correct_majority += 1

            if self.predict_winrate_sum(battle) == actual:
                correct_winrate += 1

        return {
            'majority_accuracy': correct_majority / total,
            'winrate_sum_accuracy': correct_winrate / total,
            'test_samples': total
        }

def run_benchmark(
    reg_a_path: str = './data/processed/' + config['dataset']['m_a'],
    reg_b_path: str = './data/processed/' + config['dataset']['m_b'],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    # Trains on 80% Reg M-A, tests on 20% Reg M-A, validates on Reg M-B
    random.seed(seed)

    # Load Reg M-A dataset
    with open(reg_a_path, 'r', encoding='utf-8') as f:
        reg_a_battles = json.load(f)

    # Split reg M-A
    random.shuffle(reg_a_battles)
    split_idx = int(len(reg_a_battles) * train_ratio)
    train_reg_a = reg_a_battles[:split_idx]
    test_reg_a = reg_a_battles[split_idx:]

    # Load Reg B dataset
    with open(reg_b_path, 'r', encoding='utf-8') as f:
        val_reg_b = json.load(f)

    # Train model on 80% Reg A
    model = VGCNaiveBenchmark()
    model.fit(train_reg_a)

    # Evaluate on 20% Reg A and full Reg B
    reg_a_test_metrics = model.evaluate(test_reg_a)
    reg_b_val_metrics = model.evaluate(val_reg_b)

    return {
        'reg_a_in_distribution_test': reg_a_test_metrics,
        'reg_b_cross_regulation_validation': reg_b_val_metrics
    }

if __name__ == '__main__':
    # execution on processed datasets
    try:
        results = run_benchmark()
        print("Naive Benchmark Evaluation Results:")
        print(json.dumps(results, indent=2))
    except FileNotFoundError:
        print("Data file not found.")