from pymongo import MongoClient

uri = 'mongodb://localhost:27017'

client =  MongoClient(uri)

db = client['Data_Augmentation']

collection = db['My_Project']
documents =  collection.find()


new_documents = []

for document in documents:
    if document.get('article'): 
        if document['article'] != "":
            new_documents.append(document)

new_collection = db['cleaned_data']

# Use insert_many instead of insert (deprecated)
if new_documents:  # Ensure there is data to insert
    new_collection.insert_many(new_documents)