from huggingface_hub import HfApi, create_repo

api = HfApi()


def save_session(repo_id, path):
    try:
        repo_url = create_repo(repo_id, private=True)
    except Exception as e:
        repo_url = None

    api.upload_folder(
        folder_path=path,
        path_in_repo=path.split("/")[-1],
        repo_id=repo_id,
        repo_type="dataset",
    )
