import kagglehub
import shutil
import os

# Download dataset
path = kagglehub.dataset_download(
    "hafiznouman786/potato-plant-diseases-data"
)

print("Downloaded:", path)

# Optional copy
destination="dataset"

if not os.path.exists(destination):
    shutil.copytree(path,destination)

print("Dataset copied")