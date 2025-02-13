import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import gc
from scipy.signal import butter, lfilter
import soundfile as sf
import noisereduce as nr

absolute_path = "/media/angela/HIKVISION/Informatica/Thesis"
destination_path = "/media/angela/HIKVISION/Informatica/Thesis/whistle"
noise_duration = 7
counter = 1

save = True

def trim_flac_librosa(input_path, output_dir, start_sec, end_sec):
    # Load audio data and sample rate
    y, sr = librosa.load(input_path, sr=None, offset=start_sec, duration=end_sec - start_sec)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save trimmed audio
    output_path = os.path.join(output_dir, "trimmed_librosa.flac")
    sf.write(output_path, y, sr, subtype='PCM_24')  # FLAC with 24-bit depth
    return output_path

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


def spectral_subtraction(y, sr, noise_segment):
    n_fft = 2048
    hop_length = n_fft // 4
    # Compute Short-Time Fourier Transform (STFT)
    noise_D = librosa.stft(noise_segment, n_fft=n_fft, hop_length=hop_length)
    noise_magnitude, noise_phase = librosa.magphase(noise_D)

    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = librosa.magphase(D)

    # Estimate noise using the frames preceding the whistle segment
    noise_mag = np.median(noise_magnitude[:, :], axis=1, keepdims=True)

    '''
    noise_frames = max(start_sample - 5 * sr, 0)  # Take 5 seconds before the start
    noise_mag = np.median(magnitude[:, :], axis=1, keepdims=True)
    '''
    oversubtraction_factor = 2 # Adjust empirically
    spectral_floor = 0.01 * np.mean(noise_mag)  # Adaptive floor
    # Subtract estimated noise from magnitude
    cleaned_mag = np.maximum(magnitude - oversubtraction_factor * noise_mag, spectral_floor)

    # Reconstruct the signal
    cleaned_stft = cleaned_mag * phase
    y_cleaned = librosa.istft(cleaned_stft)
    return y_cleaned


def remove_noise(y, sr, start_sample, noise_segment):
    # Apply high-pass filter with a stronger cutoff to remove more low frequencies
    y_filtered = high_pass_filter(y, sr, cutoff=5000)  # Increasing cutoff to 1000 Hz
    # Apply low-pass filter to remove high frequencies
    y_filtered = low_pass_filter(y_filtered, sr, cutoff=30000)  # Decreasing cutoff to 1000 Hz
    # Apply spectral subtraction using estimated noise spectrum
    y_cleaned = spectral_subtraction(y_filtered, sr, noise_segment)
    #y_cleaned = nr.reduce_noise(y=y_filtered, y_noise=noise_segment, sr=sr)
    return y_cleaned


def plot_spectrogram(flac_file, file_name, start_time, end_time, segment_type, confidence):
    global noise_duration, counter
    # Load audio file
    y, sr = librosa.load(flac_file, sr=None)

    # Convert time to sample index
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    # Apply noise filtering
    original_segment = y[start_sample:end_sample]
    noise_segment = y[max(0, start_sample - noise_duration * sr):start_sample-sr]
    segment = remove_noise(original_segment, sr, start_sample, noise_segment)

    print(f"Segment duration: {len(segment) / sr:.2f}s")
    print(f"Segment start: {start_sample}, end: {end_sample}")

    # Compute spectrogram for segment
    D = librosa.amplitude_to_db(np.abs(librosa.stft(segment)), ref=np.max)

    # Plot spectrogram
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    file_name = file_name.split(".")[0]
    plt.title(f"Spectrogram {file_name} from {start_time}s to {end_time}s\nType: {segment_type} | Confidence: {confidence}")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.show()
    if save:
        user_input = input("Save? ")
        print(f"You entered: {user_input}")
        if user_input == 'y':
            name = f"/{counter}.flac"
            plt.savefig(destination_path+f"{counter}.png")
            counter += 1
            sf.write(destination_path+"/original"+name, original_segment, sr, subtype='PCM_24')
            sf.write(destination_path+name, segment, sr, subtype='PCM_24')

    del y, sr, original_segment, noise_segment, segment, D
    gc.collect()

def find_matching_files(directory):
    available_files = set()
    for file in os.listdir(directory):
        available_files.add(file)
    return available_files


def process_csv(csv_file, audio_dir):
    df = pd.read_csv(csv_file)
    available_files = find_matching_files(audio_dir)
    df = df[(df['file_name'].isin(available_files)) & (df['type']=='whistle')]
    print(len(df))

    for index, row in df.iterrows():
        audio_file_path = os.path.join(audio_dir, row['file_name'])
        plot_spectrogram(audio_file_path, row['file_name'], row['initial_point'], row['finish_point'], row['type'], row['confidence'])



if __name__ == "__main__":
    os.makedirs(destination_path, exist_ok=True)
    os.makedirs(destination_path+"/original", exist_ok=True)
    process_csv("../test/Tagging.csv", absolute_path)