import pytest
from vgc_matchup_predictor.models.benchmark import VGCNaiveBenchmark

@pytest.fixture
def sample_battle():
    return {
        'p1a': 'Pikachu',
        'p1b': 'Charizard',
        'p1_back': ['Venusaur', 'Blastoise', 'Snorlax', 'Gengar'],
        'p2a': 'Dragonite',
        'p2b': 'Garchomp',
        'p2_back': ['Tyranitar', 'Salamence', 'Metagross', 'Hydreigon'],
        'winner': 'p1'
    }


def test_extract_team(sample_battle):
    # Extract teams
    p1_team = VGCNaiveBenchmark.extract_team(sample_battle, 'p1')
    p2_team = VGCNaiveBenchmark.extract_team(sample_battle, 'p2')

    # Verify team size
    assert len(p1_team) == 6
    assert len(p2_team) == 6

     # Verify order (2 leads + 4 back)
    expected_p1 = ['Pikachu', 'Charizard', 'Venusaur', 'Blastoise', 'Snorlax', 'Gengar']
    expected_p2 = ['Dragonite', 'Garchomp', 'Tyranitar', 'Salamence', 'Metagross', 'Hydreigon']
    
    assert p1_team == expected_p1
    assert p2_team == expected_p2


def test_fit_and_majority_predict(sample_battle):
    model = VGCNaiveBenchmark()

    battle_p2_wins = sample_battle.copy()
    battle_p2_wins['winner'] = 'p2'
    train_data = [sample_battle, battle_p2_wins, battle_p2_wins]

    model.fit(train_data)

    assert model.majority_winner == 'p2'
    assert model.predict_majority(sample_battle) == 'p2'


def test_fit_and_winrate_sum_predict(sample_battle):
    model = VGCNaiveBenchmark()

    model.fit([sample_battle])

    assert model.pokemon_winrates['Pikachu'] == 1.0
    assert model.pokemon_winrates['Dragonite'] == 0.0

    pred = model.predict_winrate_sum(sample_battle)
    assert pred == 'p1'



def test_evaluate(sample_battle):
    model = VGCNaiveBenchmark()

    model.fit([sample_battle])

    metrics = model.evaluate([sample_battle])

    assert 'majority_accuracy' in metrics
    assert 'winrate_sum_accuracy' in metrics
    assert 'test_samples' in metrics
    assert metrics['test_samples'] == 1
    assert metrics['majority_accuracy'] == 1.0
    assert metrics['winrate_sum_accuracy'] == 1.0