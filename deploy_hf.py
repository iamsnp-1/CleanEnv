import os
import sys
from huggingface_hub import HfApi

def deploy_to_huggingface(token, username, repo_name="datacleanenv-x"):
    api = HfApi(token=token)
    repo_id = f"{username}/{repo_name}"

    print(f"Creating Space: {repo_id}...")
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="docker",
            private=False,
            exist_ok=True
        )
        print("Space successfully mapped or already exists.")
    except Exception as e:
        print(f"Failed to create space: {e}")
        sys.exit(1)

    print("Uploading environment bounds...")
    try:
        api.upload_folder(
            folder_path=".",
            repo_id=repo_id,
            repo_type="space",
            ignore_patterns=[
                ".git",
                "__pycache__",
                ".env",
                "deploy_hf.py",
                "*.pyc"
            ]
        )
        print(f"\nDeployment Complete! View your space at: https://huggingface.co/spaces/{repo_id}")
    except Exception as e:
        print(f"Failed to upload files: {e}")
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Deploy DCX to HuggingFace Spaces")
    parser.add_argument("--token", required=True, help="Your HuggingFace Write Token")
    parser.add_argument("--username", required=True, help="Your HuggingFace Username")
    
    args = parser.parse_args()
    deploy_to_huggingface(args.token, args.username)
