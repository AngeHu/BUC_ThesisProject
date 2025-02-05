import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
from scipy.signal import butter, lfilter

absolute_path = "/media/angela/HIKVISION/Informatica/Thesis"
destination_path = "./test/whistle"

def high_pass_filter(y, sr, cutoff=500):
    # Design a high-pass filter
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = butter(4, normal_cutoff, btype='high', analog=False)
    y_filtered = lfilter(b, a, y)
    return y_filtered

def low_pass_filter(y, sr, cutoff=500):
    # Design a low-pass filter
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = butter(4, normal_cutoff, btype='low', analog=False)
    y_filtered = lfilter(b, a, y)
    return y_filtered


def estimate_noise(y, sr, segment_duration=1.0):
    # Select a random noise segment outside the given signal
    total_duration = len(y) / sr
    if total_duration <= segment_duration * 2:
        return np.zeros_like(y)  # Not enough data for noise estimation

    # Pick a random segment avoiding the main signal
    noise_start = int(np.random.uniform(0, total_duration - segment_duration) * sr)
    noise_end = noise_start + int(segment_duration * sr)
    noise_segment = y[noise_start:noise_end]

    return noise_segment


def spectral_subtraction(y, sr, noise_segment):
    # Compute Short-Time Fourier Transform (STFT)
    D = librosa.stft(y)
    noise_D = librosa.stft(noise_segment)

    # Estimate noise magnitude
    noise_mag = np.median(np.abs(noise_D), axis=1, keepdims=True)

    # Compute magnitude and phase
    magnitude, phase = librosa.magphase(D)

    # Subtract estimated noise from magnitude
    cleaned_mag = np.maximum(magnitude - noise_mag, 0)

    # Reconstruct the signal
    cleaned_stft = cleaned_mag * phase
    y_cleaned = librosa.istft(cleaned_stft)
    return y_cleaned


def remove_noise(y, sr):
    # Apply high-pass filter with a stronger cutoff to remove more low frequencies
    y_filtered = high_pass_filter(y, sr, cutoff=8000)  # Increasing cutoff to 1000 Hz
    # Apply low-pass filter to remove high frequencies
    y_filtered = low_pass_filter(y_filtered, sr, cutoff=30000)  # Decreasing cutoff to 1000 Hz
    # Estimate noise from a random segment
    noise_segment = estimate_noise(y, sr)

    # Apply spectral subtraction using the estimated noise segment
    y_cleaned = spectral_subtraction(y_filtered, sr, noise_segment)
    return y_cleaned

def plot_spectrogram(flac_file, start_time, end_time, segment_type):
    # Load audio file
    y, sr = librosa.load(flac_file, sr=None)

    # Convert time to sample index
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    segment = y[start_sample:end_sample]

    # Apply noise filtering
    segment = remove_noise(segment, sr)

    # Compute spectrogram for segment
    D = librosa.amplitude_to_db(np.abs(librosa.stft(segment)), ref=np.max)

    # Plot spectrogram
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"Spectrogram from {start_time}s to {end_time}s\nType: {segment_type}")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.show()


def find_matching_files(directory):
    available_files = set()
    for root, _, files in os.walk(directory):
        for file in files:
            available_files.add(file)
    return available_files


def process_csv(csv_file, audio_dir):
    df = pd.read_csv(csv_file)
    available_files = find_matching_files(audio_dir)
    df = df[(df['file_name'].isin(available_files)) & (df['type']=='whistle')]

    for index, row in df.iterrows():
        audio_file_path = os.path.join(audio_dir, row['file_name'])
        plot_spectrogram(audio_file_path, row['initial_point'], row['finish_point'], row['type'])

if __name__ == "__main__":
    process_csv("./test/Tagging.csv", absolute_path)