import os
import pandas as pd
from pymongo import MongoClient
from deep_translator import GoogleTranslator

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
        print(f"Error loading data from MongoDB: {e}")
        return None, None

df, mongo_client = load_data_from_mongo()
if df is None:
    print("Data loading failed. Exiting script.")
    exit()

# Step 2: Define Back Translation Function
def apply_back_translation(text, intermediate_lang="fr"):
    """
    Apply back translation by translating the text to an intermediate language
    (e.g., French) and then translating it back to English.
    """
    if not isinstance(text, str) or text.strip() == "":
        return text  # Skip empty values

    try:
        # Translate to intermediate language (French by default)
        translated_text = GoogleTranslator(source="en", target=intermediate_lang).translate(text)
        
        # Translate back to English
        back_translated_text = GoogleTranslator(source=intermediate_lang, target="en").translate(translated_text)

        return back_translated_text
    except Exception as e:
        print(f"Error during back translation: {e}")
        return text  # Return original text if error occurs

# Step 3: Apply Back Translation and Save Results
output_dir = "augmentation_results/Back_Translation"
os.makedirs(output_dir, exist_ok=True)

df["Back_Translated"] = df["article"].apply(lambda x: apply_back_translation(x, intermediate_lang="fr"))  # Using French

# Save to CSV
csv_path = f"{output_dir}/Back_Translation.csv"
df.to_csv(csv_path, index=False)
print(f"Back Translated data saved to: {csv_path}")

# Save to MongoDB
db = mongo_client['Data_Augmentation']
aug_collection = db['augmented_Back_Translation']
records = df.to_dict(orient='records')
aug_collection.insert_many(records)
print("Back Translated data saved to MongoDB.")

# Close MongoDB Connection
mongo_client.close()
print("✅ Back Translation completed.")


