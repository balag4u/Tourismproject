from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

api.upload_folder(
    folder_path="tourismproject/deployment",   # local folder containing app.py & requirements.txt
    repo_id="nsa9/Tourism-Package-Prediction",         # Hugging Face Space repo
    repo_type="space",                                 # hosting as a Space
    path_in_repo="",                                   # root of the Space
)
