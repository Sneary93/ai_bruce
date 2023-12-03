import streamlit as st
import configparser 
from langchain.document_loaders.csv_loader import CSVLoader
from audiocraft.models import MusicGen
from audiocraft.models import MultiBandDiffusion
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import os
import subprocess

from google.colab import drive
drive.mount('/content/drive')

config_ini_location = '/content/drive/MyDrive/Colab Notebooks/IS883/config.ini'
config = configparser.ConfigParser()
config.read(config_ini_location)
openai_api_key = config['OpenAI']['API_KEY']

# Change directory to the cloned repository
os.chdir('ai_bruce')

# Pull the latest changes from GitHub (optional, in case changes were made outside Colab)
subprocess.run(['git', 'pull', 'origin', 'main'])

# # Pull the latest changes from GitHub (optional, in case changes were made outside Colab)
# !git pull origin main

# Load CSV file
csv_path = "/content/drive/MyDrive/Colab Notebooks/IS883/IS883 Project_Team1/Sleep_health_and_lifestyle_dataset.csv"
loader = CSVLoader(csv_path)
data = loader.load()
text_data = data[2].page_content

# Spotify API credentials
SPOTIPY_CLIENT_ID = '20dbe90b141f4177b3331613a262e24e'
SPOTIPY_CLIENT_SECRET = '3e164a1c553e4ad3b5e450af22c15462'
SPOTIPY_REDIRECT_URI = 'http://localhost:3000/callback'

# Initialize Spotify client
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=SPOTIPY_CLIENT_ID,
    client_secret=SPOTIPY_CLIENT_SECRET,
    redirect_uri=SPOTIPY_REDIRECT_URI,
    scope="user-read-private user-read-email",
))

# Music generation setup
model = MusicGen.get_pretrained('facebook/musicgen-small')
model.set_generation_params(
    use_sampling=True,
    top_k=250,
    top_p=0.0,
    temperature=1.0,
    duration=15.0,
    cfg_coef=3.0,
)

# Streamlit app
st.title("Music Generation and Spotify Integration App")

# Display CSV data
st.header("CSV Data")
st.write(text_data)

# Display Spotify playlists
st.header("Spotify Playlists")
user_playlists = sp.current_user_playlists()
for playlist in user_playlists['items']:
    st.write(playlist['name'])

# Music generation based on CSV data
st.header("Generated Music")
result_text = " ".join(text_data.split()[:50])  # Example: Take the first 50 words from CSV data
output = model.generate(descriptions=[result_text], progress=True, return_tokens=True)
st.audio(output[0], format="audio/wav")

# Display Spotify user profile
st.header("Spotify User Profile")
user_profile = sp.current_user()
st.write(user_profile)


