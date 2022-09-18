import comet_ml


def start_experiment():
    try:
        api = comet_ml.API()
        workspace = api.get_default_workspace()
        project_name = comet_ml.config.get_config()["comet.project_name"]

        experiment = comet_ml.APIExperiment(
            workspace=workspace, project_name=project_name
        )
        experiment.log_other("Created from", "stable-diffusion")
        return experiment

    except Exception as e:
        return None
