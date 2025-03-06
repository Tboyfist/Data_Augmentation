import os
import pandas as pd
from pymongo import MongoClient
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as naf
from deepseek import DeepSeekAPI # Hypothetical library for DeepSeek
from transformers import BertTokenizer, BertForSequenceClassification, DebertaTokenizer, DebertaForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch

# Step 1: Load Data from MongoDB
def load_data_from_mongo():
    uri = 'mongodb://localhost:27017'
    client = MongoClient(uri)
    db = client['Data_Augmentation']
    collection = db['cleaned_data']
    documents = collection.find()
    df = pd.DataFrame(list(documents))
    if '_id' in df.columns:
        df.drop('_id', axis=1, inplace=True)  # Remove MongoDB object ID
    return df

df = load_data_from_mongo()

def apply_random_augmentations(text):
    flow = naf.Sequential([nac.RandomCharAug(action="insert"), nac.RandomCharAug(action="delete"), nac.RandomCharAug(action="swap")])
    return flow.augment(text)

augmentation_methods = {

    "Random_Augmentations": apply_random_augmentations,

}
# Step 3: Create Folders and Save Augmented Data
output_dir = "augmentation_results"
os.makedirs(output_dir, exist_ok=True)

for method, func in augmentation_methods.items():
    df_aug = df.copy()
    df_aug[method] = df_aug['article'].apply(func)
    technique_folder = f"{output_dir}/{method}"
    os.makedirs(technique_folder, exist_ok=True)
    df_aug.to_csv(f"{technique_folder}/{method}.csv", index=False)
    print(f"Augmented data saved to folder: {technique_folder} as {method}.csv")

    # Save to MongoDB
    uri = 'mongodb://localhost:27017'
    client = MongoClient(uri)
    db = client['Data_Augmentation']
    aug_collection = db[f'augmented_{method}']
    records = df_aug.to_dict(orient='records')
    aug_collection.insert_many(records)
    print(f"Augmented data successfully saved to MongoDB under collection: augmented_{method}")
