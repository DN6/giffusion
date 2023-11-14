import json

import gradio as gr
from huggingface_hub import HfApi, create_repo, hf_hub_download

api = HfApi()


def save_session(org_id, repo_id, path, session_name=None):
    if org_id is None:
        raise gr.Error("Please provide an Organization Id in order to save this run")

    try:
        repo_url = create_repo(repo_id, private=True, repo_type="dataset")
    except Exception as e:
        repo_url = None

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
