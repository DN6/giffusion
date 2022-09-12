import comet_ml


def start_experiment():
    try:
        experiment = comet_ml.APIExperiment()
        return experiment

    except Exception as e:
        return None
