import mlflow


def get_or_create_experiment(experiment_location: str) -> None:
    """
    Given a DBFS directory, create an MLflow Experiment using that location if one
    does not already exist
    """

    if not mlflow.get_experiment_by_name(experiment_location):
        print("Experiment does not exist. Creating experiment")

        mlflow.create_experiment(experiment_location)

    mlflow.set_experiment(experiment_location)


def get_commit_info(commit_sha, release_version):
    tags = {"commit_sha": commit_sha, "release_version": release_version}
    return tags


def register_and_promote_model():
    pass
