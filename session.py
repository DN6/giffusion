import gradio as gr
from huggingface_hub import HfApi, create_repo

api = HfApi()


def save_session(org_id, repo_id, path, session_name=None):
    if org_id is None:
        raise gr.Error("Please provide an Organization Id in order to save this run")

    datasets = api.list_datasets(author=org_id)
    if not any([repo_id == dataset.id for dataset in datasets]):
        create_repo(repo_id, private=True, repo_type="dataset")

    repo_id = f"{org_id}/{repo_id}"
    path_in_repo = path.split("/")[-1] if session_name is None else session_name
    api.upload_folder(
        folder_path=path,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
    )
