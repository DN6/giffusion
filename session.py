import logging
import json

import diffusers
import gradio as gr
from huggingface_hub import HfApi, create_repo, hf_hub_download, scan_cache_dir

api = HfApi()

logger = logging.getLogger(__name__)

def fetch_available_models(path):
    hf_cache_info = scan_cache_dir(path)
    repo_ids = [repo.repo_id for repo in hf_cache_info.repos if repo.repo_type == "model"]

    return repo_ids


def fetch_available_pipelines():
    pipelines = [pipe for pipe in diffusers.pipelines.__all__ if "Pipeline" in pipe]
    return pipelines


def save_session(org_id, repo_id, path, session_name=None):
    if org_id is None:
        raise gr.Error("Please provide an Organization Id in order to save this run")

    try:
        create_repo(repo_id, private=True, repo_type="dataset")
    except Exception as e:
        logger.info(f"Repo Exists: {e}")

    repo_id = f"{org_id}/{repo_id}"
    path_in_repo = path.split("/")[-1] if session_name is None else session_name
    api.upload_folder(
        folder_path=path,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
    )


def load_session(org_id, repo_id, path, session_name=None):
    if org_id is None:
        raise gr.Error("Please provide an Organization Id in order to load this run")

    repo_id = f"{org_id}/{repo_id}"
    path = hf_hub_download(
        repo_id, subfolder=path, filename="parameters.json", repo_type="dataset"
    )
    with open(path, "r") as f:
        parameters = json.load(f)

    return parameters
