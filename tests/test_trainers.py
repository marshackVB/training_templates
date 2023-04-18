import mlflow
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from training_templates import SkLearnHyperoptTrainer


@pytest.fixture
def trainer(spark, default_training_args, default_tuner):

    default_training_args['tuner'] = default_tuner
    default_training_args['model'] = RandomForestClassifier
    default_training_args['model_name'] = "random_forecast"
    default_training_args['model_type'] = "classifier"

    trainer = SkLearnHyperoptTrainer(
        **default_training_args
    )
    return trainer


def test_train(trainer):
    """
    The full training workflow executes and logs a model to MLflow that
    can be loaded an scores data.
    """
    trainer.train()
    logged_model = f"runs:/{trainer.run_id}/model"
    loaded_model = mlflow.pyfunc.load_model(logged_model)

    predictions = loaded_model.predict(trainer.X_val)
    observations = trainer.X_val.shape[0]
    assert predictions.shape[0] == (observations)

    f1_eval = f1_score(trainer.y_val, 
                       predictions, 
                       average='weighted')
    
    assert f1_eval >= 0.7