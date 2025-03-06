import os
import pandas as pd
from pymongo import MongoClient
import nlpaug.augmenter.word as naw
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Step 1: Load Data from MongoDB
def load_data_from_mongo():
    try:
        uri = 'mongodb://localhost:27017'
        client = MongoClient(uri)
        db = client['Data_Augmentation']
        collection = db['cleaned_data']
        documents = list(collection.find())

        if not documents:
            raise ValueError("No data found in the MongoDB collection 'cleaned_data'.")

        df = pd.DataFrame(documents)

        if '_id' in df.columns:
            df.drop('_id', axis=1, inplace=True)  # Remove MongoDB object ID

        if 'article' not in df.columns:
            raise KeyError("Column 'article' not found in the dataset. Check your MongoDB collection structure.")

        return df, client
    except Exception as e:
        logging.error(f"Error loading data from MongoDB: {e}")
        return None, None

df, mongo_client = load_data_from_mongo()
if df is None:
    logging.error("Data loading failed. Exiting script.")
    exit()

# Step 2: Load Contextual Replacement Model
contextual_aug = naw.ContextualWordEmbsAug(
    model_path='distilbert-base-uncased',  # Using a smaller model for faster processing
    model_type='bert',
    action='substitute',
    device='cpu',
    top_k=10  # Limit to top 10 candidates for faster processing
)

def apply_contextual_replacement(text):
    """Apply BERT-based contextual word replacement."""
    if not isinstance(text, str) or text.strip() == "":
        return text  # Skip empty values
    try:
        return contextual_aug.augment(text)
    except Exception as e:
        logging.error(f"Error during contextual replacement: {e}")
        return text  # Return original text if error occurs

# Step 3: Apply Contextual Replacement and Save Results
output_dir = "augmentation_results/Contextual_Replacement"
os.makedirs(output_dir, exist_ok=True)

# Process data in batches
batch_size = 10  # Adjust based on your system's capacity
num_batches = len(df) // batch_size + (1 if len(df) % batch_size != 0 else 0)

for i in range(num_batches):
    logging.info(f"Processing batch {i+1}/{num_batches}")
    batch_df = df.iloc[i*batch_size:(i+1)*batch_size].copy()
    batch_df["Contextual_Replacement"] = batch_df["article"].apply(apply_contextual_replacement)

    # Save batch to CSV
    batch_csv_path = f"{output_dir}/Contextual_Replacement_batch_{i+1}.csv"
    batch_df.to_csv(batch_csv_path, index=False)
    logging.info(f"Batch {i+1} saved to: {batch_csv_path}")

    # Save batch to MongoDB
    db = mongo_client['Data_Augmentation']
    aug_collection = db['augmented_Contextual_Replacement']
    records = batch_df.to_dict(orient='records')
    aug_collection.insert_many(records)
    logging.info(f"Batch {i+1} saved to MongoDB.")

# Close MongoDB Connection
mongo_client.close()
logging.info("✅ Contextual Replacement completed.")
