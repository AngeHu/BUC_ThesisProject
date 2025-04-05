# transmitter
import numpy as np
from matplotlib import pyplot as plt
import channel
import params as tf
import time
import sys
import csv
from scipy.signal import chirp
import random
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import soundfile as sf
import io
import os
import librosa
import subprocess
import multiprocessing

#TODO: animare il segnale trasmesso senza effetto doppler!

if tf.DEBUG:
    import cProfile
    import atexit
    import pstats

    # Set up profiling to save to a file
    profiler = cProfile.Profile()
    profiler.enable()

    # Save profile results to a file on exit
    def save_profile():
        with open('transmitter_profile.prof', 'w') as f:  # Use 'receiver_profile.prof' for receiver.py
            ps = pstats.Stats(profiler, stream=f)
            ps.strip_dirs().sort_stats('cumulative').print_stats()

    atexit.register(save_profile)

# global variables
os.makedirs("./animation", exist_ok=True)
animation_file = "./animation/transmitter.csv"
doppler_animation_file = "./animation/transmitter_doppler.csv"
t_slot = np.linspace(0, tf.t_slot, tf.chirp_samples_doppler) # vettore tempo
chirp_signal = chirp(t_slot, f0=tf.f_min, f1=tf.f_max, t1=tf.t_slot, method='linear') # segnale chirp
e_signal = sum(chirp_signal**2) / tf.chirp_samples_doppler # potenza segnale
t_sampling = tf.T_SAMPLING

def arrange_step(start, num, step = t_sampling):
    # return np.arange(start, start + (num-1) * step, step)
    return np.linspace(start, start + num * step, num, endpoint=False)  # Ensures exact num elements

def run_script(script_name):
    subprocess.run(["python3", script_name])

# plot every 4*2 bits
def plot_function(x, y_sig, y_freq=[]):
    # Check if frequency data is provided
    plot_freq = len(y_freq) > 0

    # Create subplots dynamically
    if plot_freq:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    else:
        fig, ax2 = plt.subplots(1, 1, figsize=(10, 4))

    ax2.set_title('Transmitted Signal')
    # Always plot the signal
    ax2.plot(x, y_sig, color='r', label='Signal')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Signal')
    ax2.set_xlim(0, tf.T_frame_doppler)  # Replace with your time limit
    upper_lim = max(y_sig) + 0.1 * max(y_sig) if max(y_sig) < 1 else 1
    ax2.set_ylim(-upper_lim, upper_lim)
    ax2.grid(True)

    # Plot frequency if data exists
    if plot_freq:
        ax1.plot(x, y_freq, color='g', label='Frequency')
        ax1.set_ylabel('Frequency')
        ax1.set_xlim(0, tf.T_frame_doppler)
        ax1.set_ylim(0, 50000)
        ax1.grid(True)
        ax1.set_xticklabels([])  # Remove redundant x-axis labels for top plot

    plt.tight_layout()
    plt.show()


def encode_signal(data):
    encoded_signal = []
    binary_data = []
    if tf.BIO_SIGNALS:
        for i in data:
            if i % 4 == 0:
                encoded_signal.append(0)
                binary_data.append(0)
                binary_data.append(0)
            elif i % 4 == 1:
                encoded_signal.append(1)
                binary_data.append(0)
                binary_data.append(1)
            elif i % 4 == 2:
                encoded_signal.append(2)
                binary_data.append(1)
                binary_data.append(1)
            elif i % 4 == 3:
                encoded_signal.append(3)
                binary_data.append(1)
                binary_data.append(0)
    else:
        binary = data
        for i in range(0, len(data), 2):
            if data[i] == 0 and data[i+1] == 0:
                encoded_signal.append(0)
            elif data[i] == 0 and data[i+1] == 1:
                encoded_signal.append(1)
            elif data[i] == 1 and data[i+1] == 1:
                encoded_signal.append(2)
            elif data[i] == 1 and data[i+1] == 0:
                encoded_signal.append(3)
    if tf.DEBUG: print("Encoded signal: ", encoded_signal)
    return encoded_signal, binary_data


class Transmitter():

    def __init__(self):
        self.channel = channel.Channel("wb")
        if not tf.BER_SNR_SIMULATION: print("Transmitter ON")
        self.x = np.linspace(0, tf.T_FRAME, tf.sig_samples)
        self.frequency = np.array([])
        self.signal = np.array([])
        self.original_frequency = []
        self.original_signal = []
        # if signal is affected by doppler effect, keep count of the number of bits sent
        self.previous_data = -1
        self.extra_zero = tf.sig_samples_doppler - tf.sig_samples
        self.last_frame = 0
        if tf.BIO_SIGNALS:
            self.temp_files = dict()

    ### Generate the signal to be sent:
    # 1. Generate the frequency
    # 2. Generate the chirp signal
    def generate_signal(self, data):
        self.generate_frequency(data)
        self.generate_chirp(data)
        print("Signal length: ", len(self.signal))

    def generate_frequency(self, data):
        # generate 2 frequencies if doppler effect is present
        if tf.v_relative != 0:
            start = data * tf.chirp_samples
            end = start + tf.chirp_samples
            self.original_frequency = np.full(tf.sig_samples, 0)
            self.original_frequency[start:end] = np.linspace(tf.f_min, tf.f_max, tf.chirp_samples)
        start = data * tf.chirp_samples
        end = start + tf.chirp_samples_doppler
        self.frequency = np.full(tf.sig_samples_doppler, 0)
        self.frequency[start:end] = np.linspace(tf.f_min_scaled, tf.f_max_scaled, tf.chirp_samples_doppler)

        # adjust frequency samples if signal is affected by doppler effect
        if self.extra_zero < 0: #signal is affected by doppler effect: compressed, so we need to add extra zeros
            #np.append(self.frequency, np.zeros(abs(self.extra_zero)))
            print("extra zero: ", abs(self.extra_zero))
            self.frequency = np.pad(self.frequency, (0, abs(int(self.extra_zero))), 'constant', constant_values=0)
            print("frequency samples: ", len(self.frequency))

        elif self.extra_zero > 0: #signal is affected by doppler effect: expanded, so we need to remove extra zeros
            if self.previous_data != 3 and data != 3: # no dati sporgenti
                self.frequency = self.frequency[:-self.extra_zero] # remove the extra zeros
            elif self.previous_data == 3: # dati sporgenti
                if data == 0:
                    self.frequency = self.frequency[:-2*self.extra_zero] # remove the extra zeros twice
                elif data == 3:
                    self.frequency = self.frequency[self.extra_zero:] # remove the initial extra zeros
                else: # if the previous data was 11
                    self.frequency = self.frequency[:-self.extra_zero]  # remove final the extra zeros
                    self.frequency = self.frequency[self.extra_zero:]

        print(f"data: {data} - frequency samples: {len(self.frequency)} - max frequency: {self.frequency[-1]}")
        self.previous_data = data
        return self.frequency, self.original_frequency

    def generate_chirp(self, data):
        if tf.v_relative != 0: # signal is affected by doppler effect
            if self.extra_zero > 0: #signal is affected by doppler effect: compressed, so we need to add extra zeros
                x = np.linspace(0, tf.T_frame_doppler, tf.sig_samples_doppler)
                x = x[:len(self.frequency)]
                self.signal = np.sin(2 * np.pi * self.frequency * (x - data * tf.t_slot))
            elif self.extra_zero <= 0:
                self.signal = np.sin(2 * np.pi * self.frequency * (self.x - data * tf.t_slot))
            self.original_signal = np.sin(2 * np.pi * self.original_frequency * (self.x - data * tf.t_slot))
        else:
            self.signal = np.sin(2 * np.pi * self.frequency * (self.x - data*tf.t_slot))
        return self.signal

    def retrieve_signal(self, db_collection, data): # retrieve audio data from database
        for i in data:
            #check if exist already in temp_files
            if i in self.temp_files:
                continue
            query = {"_id": int(i)}
            result = db_collection.find_one(query)

            if result and 'audio_data' in result:
                file_path = self.save_to_tempfile(result['_id'], result['audio_data'])
                self.temp_files[i] = file_path

    def save_to_tempfile(self, file_id, audio_data):
        os.makedirs("/tmp/transmitter_cache/audio", exist_ok=True)
        file_path = f"/tmp/transmitter_cache/audio/{file_id}.flac"
        with open(file_path, 'wb') as f:
            f.write(audio_data)
        return file_path

    def cleanup(self):
        # Deletes all temporary files after processing.
        for file_path in self.temp_files.values():
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Deleted: {file_path}")

    def generate_biosignal(self, data, audio_data, duration = 0.4): # data is slot number
        file_path = self.temp_files[audio_data]
        y, sr = librosa.load(file_path, sr=None)
        self.signal = np.zeros(tf.sig_samples, dtype=np.float32)
        self.frequency = np.zeros(tf.sig_samples, dtype=np.float32) # for animation purposes

        start_idx = data * tf.chirp_samples
        end_idx = start_idx + len(y)
        self.signal[start_idx : end_idx] = y
        e_signal = sum(self.signal**2) / len(y)
        return e_signal

    def save_to_csv(self, counter, filename = animation_file):
        # Calculate the time values for the current slot
        time_values = self.x + counter * tf.T_FRAME
        # Prepare data rows in bulk
        if tf.v_relative != 0:
            rows = [[t, sig, freq] for t, sig, freq in zip(time_values, self.original_signal, self.original_frequency)]
        else:
            rows = [[t, sig, freq] for t, sig, freq in zip(time_values, self.signal, self.frequency)]
        # Append rows to the CSV file
        with open(filename, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(rows)
            file.flush()

    def save_to_csv_doppler(self, filename = doppler_animation_file):
        # Calculate the time values for the current slot
        time_values = arrange_step(self.last_frame, len(self.signal))
        print("time_values:", len(time_values))
        self.last_frame = time_values[-1] + t_sampling
        # Prepare data rows in bulk
        rows = [[t, sig, freq] for t, sig, freq in zip(time_values, self.signal, self.frequency)]
        # Append rows to the CSV file
        with open(filename, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(rows)
            file.flush()


    def send_signal(self, noise):
        self.channel.send_signal(self.signal, noise)


if __name__ == "__main__":

    transmitter = Transmitter()

    if tf.BIO_SIGNALS:
        # connect to database
        client = MongoClient(tf.uri, server_api=ServerApi('1'))
        try:
            client.admin.command('ping')
            print("Pinged your deployment. You successfully connected to MongoDB!")
        except Exception as e:
            print(e)
            exit(1)

        database = client['dolphin_database']
        collection = database['short_whistle_files_normalized']
        collection_size = collection.count_documents({})
        random.seed(tf.seed)
        data = [random.randint(1, collection_size) for _ in range(int(tf.num_bits / 2))]
        transmitter.retrieve_signal(collection, data)
    else:
        data = np.random.randint(0, 2, tf.num_bits)
    print("Data:", data)


    if tf.BER_SNR_SIMULATION:
        SNR = float(sys.argv[1])
        # turn snr to linear scale
        SNR = 10 ** (SNR / 10)
    else:
        SNR = 10 ** (tf.SNR / 10)

    i = 0
    data_slot, binary = encode_signal(data)
    print("Binary:", binary)

    # create animation file where to save values
    with open(animation_file, 'w') as file:
        writer = csv.DictWriter(file, fieldnames=["time", "signal", "frequency"])
        writer.writeheader()

    with open(doppler_animation_file, 'w') as file:
        writer = csv.DictWriter(file, fieldnames=["time", "signal", "frequency"])
        writer.writeheader()

    if tf.ANIMATION:
        animation = multiprocessing.Process(target=run_script, args=("./transmitter_animation.py",))
        animation.start()

    while i < len(data_slot):
        if tf.BIO_SIGNALS:
            e = transmitter.generate_biosignal(data_slot[i], data[i])
            transmitter.send_signal(e / SNR)
            transmitter.save_to_csv(i)
            if tf.PLOT: plot_function(transmitter.x, transmitter.signal)
        else:
            transmitter.generate_signal(data_slot[i])
            transmitter.send_signal(e_signal / SNR) # send signal with noise
            # write to file for animation
            transmitter.save_to_csv(i)
            transmitter.save_to_csv_doppler()
            if tf.PLOT: plot_function(transmitter.x, transmitter.signal, transmitter.frequency)
        i += 1

    time.sleep(1) # wait for receiver to finish receiving
    transmitter.channel.close()

    if not tf.BER_SNR_SIMULATION:
        print("Transmitter OFF")

    if tf.BIO_SIGNALS:
        transmitter.cleanup()

    if tf.ANIMATION: animation.join()


