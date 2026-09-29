# import kagglehub
# import os
# import shutil

# # Download UTKFace dataset
# path = kagglehub.dataset_download("jangedoo/utkface-new")

# print("Dataset downloaded to:")
# print(path)

import kagglehub
import os

path = kagglehub.dataset_download("jangedoo/utkface-new")

print("Dataset path:", path)

for root, dirs, files in os.walk(path):
    print("Folder:", root)
    print("Number of files:", len(files))
    print()
