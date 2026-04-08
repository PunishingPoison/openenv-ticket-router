"""
Hugging Face Space Uploader Script
Helper script to deploy this OpenEnv project to a Hugging Face Space.
"""

import sys
import os
from huggingface_hub import HfApi

def main():
    print("=== OpenEnv Hugging Face Deployment ===")
    print("This script will upload your local directory to a Hugging Face Space.")
    
    space_id = input("\nEnter your Space ID (format -> username/space_name for e.g. johndoe/ticket-router): ").strip()
    
    if not space_id or "/" not in space_id:
        print("Error: Invalid format. Please enter username/space_name")
        sys.exit(1)
        
    print(f"\nUploading to Space: {space_id}...")
    try:
        api = HfApi()
        # Uploads the entire current directory to the root of the repo
        api.upload_folder(
            folder_path=".",
            repo_id=space_id,
            repo_type="space",
            # Exclude directories we don't want to push
            ignore_patterns=["__pycache__/*", "*.pyc", ".git/*", ".env", ".gemini/*"],
        )
        print(f"\n\u2705 Success! Your code was uploaded to https://huggingface.co/spaces/{space_id}")
        print("Hugging Face will now build the Docker container. You can monitor the building process on the website.")
    except Exception as e:
        print("\n\u274C Error uploading to Hugging Face:")
        print(e)
        print("\nMake sure you are logged in (python -c \"import huggingface_hub; huggingface_hub.login()\")")
        print("and that your Space exists and you have 'Write' access to it.")

if __name__ == "__main__":
    main()
