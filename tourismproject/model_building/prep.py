# Data Preparation for Travel Package Dataset

# for file system and environment handling
import os

# for data manipulation
import pandas as pd

# for train-test split
from sklearn.model_selection import train_test_split

# for Hugging Face Hub interaction
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# Hugging Face setup
api = HfApi(token=os.getenv("HF_TOKEN"))

DATASET_REPO_ID = "nsa9/tourism-package-prediction"
REPO_TYPE = "dataset"

# Ensure dataset repo exists
try:
    api.repo_info(repo_id=DATASET_REPO_ID, repo_type=REPO_TYPE)
except RepositoryNotFoundError:
    create_repo(
        repo_id=DATASET_REPO_ID,
        repo_type=REPO_TYPE,
        private=False
    )

# Load raw dataset from Hugging Face
DATASET_PATH = "hf://datasets/nsa9/tourism-package-prediction/tourism.csv"
dataset = pd.read_csv(DATASET_PATH)

# Target and feature definitions
TARGET = "ProdTaken"

numeric_features = [
    "Age",
    "NumberOfPersonVisiting",
    "PreferredPropertyStar",
    "NumberOfTrips",
    "NumberOfChildrenVisiting",
    "MonthlyIncome",
    "PitchSatisfactionScore",
    "NumberOfFollowups",
    "DurationOfPitch",
]

categorical_features = [
    "TypeofContact",
    "CityTier",
    "Occupation",
    "Gender",
    "MaritalStatus",
    "Designation",
    "ProductPitched",
]

binary_features = [
    "Passport",
    "OwnCar",
]

feature_columns = numeric_features + categorical_features + binary_features

# Train / Test split
X = dataset[feature_columns]
y = dataset[TARGET]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Save splits locally
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

# Upload splits to Hugging Face dataset repo
FILES_TO_UPLOAD = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file_path in FILES_TO_UPLOAD:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path,
        repo_id=DATASET_REPO_ID,
        repo_type=REPO_TYPE,
    )
