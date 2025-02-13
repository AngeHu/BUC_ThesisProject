import os
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import librosa

uri = "mongodb+srv://Simulator:Simulator@dolphincleanaudio.q2bsd.mongodb.net/?retryWrites=true&w=majority&appName=DolphinCleanAudio"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)


audio_dir = "/media/angela/HIKVISION/Informatica/Thesis/short_whistle"
count = 1
# Upload each FLAC file

db = client['dolphin_database']
collection = db['short_whistle_files']  # Collection for storing files

for filename in os.listdir(audio_dir):
    if filename.endswith(".flac"):
        file_path = os.path.join(audio_dir, filename)

        # Read the FLAC file as binary data
        with open(file_path, "rb") as f:
            flac_binary = f.read()
            # calulate duration
            duration = librosa.get_duration(filename=file_path)


        if duration <= 0.4:
            custom_id = count
            count += 1
        else:
            continue

        # Insert into MongoDB with metadata
        collection.insert_one({
            "_id": custom_id,  # Use UUID as the primary key
            "filename": filename,
            "audio_data": flac_binary,
            "format": "FLAC",
            "additional_metadata": {  # Add any other fields
                "duration_seconds": duration
            }
        })

        print(f"Uploaded: {filename} with ID: {custom_id}")
