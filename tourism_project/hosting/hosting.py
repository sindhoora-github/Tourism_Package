from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("TP_HF_TOKEN"))
api.upload_folder(
    folder_path="tourism_project/deployment",     # the local folder containing your files
    repo_id="sindhoorasuresh/Tourism-Package",          # the target repo
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
