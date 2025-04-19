from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import numpy as np
import matplotlib.pyplot as plt
from bson.binary import Binary
import soundfile as sf
import io
import os
# retrieve chirps from database in order
from dotenv import load_dotenv

DEBUG = False

load_dotenv()
USERNAME = os.getenv("DB_USERNAME")
PASSWORD = os.getenv("DB_PASSWORD")

uri = f"mongodb+srv://{USERNAME}:{PASSWORD}@dolphincleanaudio.q2bsd.mongodb.net/?retryWrites=true&w=majority&appName=DolphinCleanAudio"
client = MongoClient(uri, server_api=ServerApi('1'))
client.admin.command('ping')
print("Pinged your deployment. You successfully connected to MongoDB!")

database = client['dolphin_database']
collection = database['short_whistle_files']
dest_collection = database['short_whistle_files_normalized']

def normalize_audio(audio):
    return audio / np.max(np.abs(audio))

# retrieve chirps from database in order
collection_size = collection.count_documents({})
if DEBUG: print("Collection size: ", collection_size)

for i in range(1, collection_size):
    result = collection.find_one({"_id": i})
    audio_bytes = result["audio_data"]
    if DEBUG: print(audio_bytes[:20])  # Print first 20 bytes
    audio, sr = sf.read(io.BytesIO(audio_bytes), dtype='float32')  # Convert to NumPy array
    audio = normalize_audio(audio)

    buffer = io.BytesIO()
    subtype = 'FLOAT'
    if result["format"].lower() == 'flac':
        subtype = 'PCM_16'
        audio = (audio * 32767).astype(np.int16)

    sf.write(buffer, audio, sr, format=result["format"], subtype=subtype)

    dest_collection.insert_one({
        "_id": i,
        "filename": result["filename"],
        "audio_data": Binary(buffer.getvalue()),
        "format": result["format"],
        "additional_metadata": result["additional_metadata"]
    })

    if DEBUG:
        # plot audio
        plt.figure()
        plt.plot(audio)
        plt.title(result["filename"])
        plt.show()
        plt.pause(1)

