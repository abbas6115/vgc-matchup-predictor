import pytest
import numpy as np
from vgc_matchup_predictor.models.random_forest import VGCRandomForest

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
    p1_team = VGCRandomForest.extract_team(sample_battle, 'p1')
    p2_team = VGCRandomForest.extract_team(sample_battle, 'p2')

    assert len(p1_team) == 6
    assert len(p2_team) == 6

    expected_p1 = ['Pikachu', 'Charizard', 'Venusaur', 'Blastoise', 'Snorlax', 'Gengar']
    expected_p2 = ['Dragonite', 'Garchomp', 'Tyranitar', 'Salamence', 'Metagross', 'Hydreigon']

    assert p1_team == expected_p1
    assert p2_team == expected_p2


def test_prepare_features(sample_battle):
    model = VGCRandomForest()

    X, y = model.prepare_features([sample_battle], is_train=True)

    assert X.shape[0] == 1
    assert y[0] == 1


def test_fit_and_predict(sample_battle):
    model = VGCRandomForest(n_estimators=10)

    battle_p2_wins = sample_battle.copy()
    battle_p2_wins['winner'] = 'p2'
    train_data = [sample_battle] * 5 + [battle_p2_wins] * 5

    model.fit(train_data)
    assert model.is_fitted is True

    prediction = model.predict(sample_battle)
    assert prediction in ['p1', 'p2']


def test_evaluate(sample_battle):
    model = VGCRandomForest(n_estimators=10)

    model.fit([sample_battle])

    metrics = model.evaluate([sample_battle])

    assert 'random_forest_accuracy' in metrics
    assert 'random_forest_precision' in metrics
    assert 'random_forest_recall' in metrics
    assert 'random_forest_f1' in metrics
    assert 'test_samples' in metrics
    assert metrics['test_samples'] == 1


def test_feature_importances(sample_battle):
    model = VGCRandomForest(n_estimators=10)
    model.fit([sample_battle])

    importances = model.feature_importances()

    assert isinstance(importances, dict)
    assert 'Pikachu' in importances
    assert 'Dragonite' in importances


def test_predict_unfitted_raises_error(sample_battle):
    model = VGCRandomForest()

    with pytest.raises(RuntimeError):
        model.predict(sample_battle)

    with pytest.raises(RuntimeError):
        model.feature_importances()