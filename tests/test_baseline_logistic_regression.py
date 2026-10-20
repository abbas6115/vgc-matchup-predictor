import pytest
import numpy as np
from vgc_matchup_predictor.models.baseline_logistic_regression import VGCLogisticRegressionBaseline

@pytest.fixture
def sample_battle():
    return{
        'p1a': 'Pikachu',
        'p1b': 'Charizard',
        'p1_back': ['Venusaur', 'Blastoise', 'Snorlax', 'Gengar'],
        'p2a': 'Dragonite',
        'p2b': 'Garchomp',
        'p2_back': ['Tyranitar', 'Salamence', 'Metagross', 'Hydreigon'],
        'winner': 'p1'
    }

def test_extract_team(sample_battle):
    # Extract teams for p1 & p2
    p1_team = VGCLogisticRegressionBaseline.extract_team(sample_battle, 'p1')
    p2_team = VGCLogisticRegressionBaseline.extract_team(sample_battle, 'p2')

    # Verify team sizes (2 leads + 4 back = 6 total mons)
    assert len(p1_team) == 6
    assert len(p2_team) == 6

    # Verify order (2 leads + 4 back)
    expected_p1 = ['Pikachu', 'Charizard', 'Venusaur', 'Blastoise', 'Snorlax', 'Gengar']
    expected_p2 = ['Dragonite', 'Garchomp', 'Tyranitar', 'Salamence', 'Metagross', 'Hydreigon']

    assert p1_team == expected_p1
    assert p2_team == expected_p2


def test_prepate_features(sample_battle):
    model = VGCLogisticRegressionBaseline()

    X, y = model.prepare_features([sample_battle], is_train=True)

    # Assert matrix and targets
    assert X.shape[0] == 1
    assert y[0] == 1


def test_fit_and_predict(sample_battle):
    model = VGCLogisticRegressionBaseline()

    # Fake dataset with both winner outcomes
    battle_p2_wins = sample_battle.copy()
    battle_p2_wins['winner'] = 'p2'
    train_data = [sample_battle] * 5 + [battle_p2_wins] * 5

    # Train model
    model.fit(train_data)
    assert model.is_fitted is True

    # Predict single battle winner
    prediction = model.predict(sample_battle)
    assert prediction in ['p1', 'p2']


def test_evaluate(sample_battle):
    model = VGCLogisticRegressionBaseline()
    # Logistic regression requires both winner classes in its training data.
    battle_p2_wins = sample_battle.copy()
    battle_p2_wins['winner'] = 'p2'

    model.fit([sample_battle, battle_p2_wins])

    metrics = model.evaluate([sample_battle])

    # Assert metrics
    assert 'logistic_regression_accuracy' in metrics
    assert 'test_samples' in metrics
    assert metrics['test_samples'] == 1
    assert isinstance(metrics['logistic_regression_accuracy'], float)


def test_predict_unfitted_raises_error(sample_battle):
    model = VGCLogisticRegressionBaseline()

    # Ensure predict is called before RuntimeError
    with pytest.raises(RuntimeError):
        model.predict(sample_battle)
