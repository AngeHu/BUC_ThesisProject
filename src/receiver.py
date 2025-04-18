# Receiver script
# This script simulates a receiver that receives a signal from a transmitter, processes it, and decodes the information.
# It includes functions for filtering, correlation, and plotting the results.

import time
import numpy as np
from matplotlib import pyplot as plt
import channel
import params as tf
from scipy.signal import chirp, spectrogram, correlate, stft, hilbert
from scipy.fft import fftshift
from scipy.signal import butter, lfilter, find_peaks
import sys
import csv
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import random
import os
import librosa
import subprocess
import multiprocessing

def _print(*args, **kwargs):
    # Custom print function to print Receier before every message
    print("RECEIVER: ", *args, **kwargs)

os.makedirs("./animation", exist_ok=True)
animation_file = "./animation/receiver.csv"

SAVE_IMG = tf.SAVE_IMG
# plots and results are saved in the following directories - to change the path, change the params.py file
# image directory - contains plots of correlation
os.makedirs(tf.img_directory, exist_ok=True)
# result directory - contains results of decoding
os.makedirs(tf.res_directory, exist_ok=True)

# enable profiling if DEBUG is true
if tf.DEBUG:
    import cProfile
    import atexit
    import pstats

    # set up profiling to save to a file
    profiler = cProfile.Profile()
    profiler.enable()

    # save profile results to a file on exit
    def save_profile():
        with open('receiver_profile.prof', 'w') as f:
            ps = pstats.Stats(profiler, stream=f)
            ps.strip_dirs().sort_stats('cumulative').print_stats()

    atexit.register(save_profile)

# increase agg.path.chunksize for faster plotting
plt.rcParams['agg.path.chunksize'] = 10000

t_slot = np.linspace(0, tf.t_slot, tf.chirp_samples) # slot time vector
t_frame = np.linspace(0, tf.T_FRAME, tf.sig_samples) # frame time vector
chirp_signal = chirp(t_slot, f0=tf.f_min, f1=tf.f_max, t1=tf.t_slot, method='linear') # reference chirp signal

# time intervals
slot1 = {"start": 0, "end": 1, "data": [0, 0]}
slot2 = {"start": 1, "end": 2, "data": [0, 1]}
slot3 = {"start": 2, "end": 3, "data": [1, 1]}
slot4 = {"start": 3, "end": 4, "data": [1, 0]}
intervals = [slot1, slot2, slot3, slot4]

# wrap the mean function to handle empty indices
def mean(x, indices):
    if indices.size == 0:
        return 0
    return np.mean(x[indices])

def run_script(script_name):
    subprocess.run(["python3", script_name])

# plot function - used to plot the signal and correlation
def plot_function(x, y):
    if(len(x) != len(y)):
        _print("Error: x and y must have the same length")
        return
    figure, ax = plt.subplots()
    sig, = ax.plot(x, y, color='r', label='Segnale')  # Crea il grafico

    ax.set_xlabel('Time(s)')  # Aggiunge un'etichetta all'asse x
    ax.set_ylabel('Amplitude')  # Aggiunge un'etichetta all'asse y1
    ax.set_xlim(0, 4*tf.T_FRAME)
    ax.set_ylim(-5, 5)
    plt.grid(True)  # Aggiunge una griglia al grafico (opzionale)
    plt.show()  # Mostra il grafico

class Receiver:
    def __init__(self):
        self.channel = channel.Channel('rb')
        if not tf.BER_SNR_SIMULATION: _print("ON")
        self.x = np.linspace(0, tf.T_FRAME, tf.sig_samples)
        self.correlation = []
        self.amplitude_envelope = []
        # results array - one per decoding algorithm
        self.mean_peak_decoded =np.array([], dtype=np.int8)
        self.max_peak_decoded = np.array([], dtype=np.int8)
        self.slot_peak_decoded = np.array([], dtype=np.int8)
        self.peak_density_decoded = np.array([], dtype=np.int8)
        self.default_value = [0, 0]
        self.temp_files = dict()

    def __delete__(self):
        if os.path.exists(self.channel.channel):
            os.remove(self.channel.channel)
            _print(f"Channel named {self.channel.channel} is deleted successfully", file=sys.stderr)


    def read(self):
        data = self.channel.read_signal()
        return data

    def plot_data(self, data):
        # _print("Data length: ", len(data))
        x = np.linspace(0, 4*tf.T_frame, len(data))
        plot_function(x, data)

    def plot_correlation(self, signal: np.array):
        _print("Plotting correlation")
        global chirp_signal
        self.correlation = correlate(signal, chirp_signal, mode='full')
        lags = np.arange(-len(signal) + 1, len(chirp_signal)) / (tf.sig_samples)
        # lags = np.arange(0, len(signal))
        plt.figure()
        plt.plot(lags, self.correlation)
        plt.title('Cross-Correlation between Signal and Chirp')
        plt.xlabel('Lag')
        plt.grid(True)
        plt.show()

    def plot_spectrogram(self, signal: np.array):
        _print("Plotting spectrogram")
        _print("Signal length: ", len(signal))
        f, t, Sxx = spectrogram(signal, fs=tf.F_SAMPLING, window='hamming', nperseg=256, noverlap=256 * 0.25,
                                nfft=256)

        plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap="viridis", vmin=-150, vmax=0)
        # plt.pcolormesh(t, f, Sxx_magnitude, shading='gouraud', cmap="viridis", vmin=-150, vmax=0)
        plt.colorbar(label="Power/Frequency (dB/Hz)")
        plt.title("Spectrogram of Noisy Signal")
        plt.ylabel("Frequency [Hz]")
        plt.xlabel("Time [s]")
        plt.tight_layout()  # Adjust layout to make room for the labels
        plt.show()

    # compute power of the signal to plot the spectrogram
    def compute_power(self, signal: np.array):
        _print("Computing power")
        f, t, Sxx = spectrogram(signal, fs=tf.F_SAMPLING, window='hamming', nperseg=2048, noverlap=2048 * 0.25,
                                nfft=2048)
        power = np.sum(Sxx, axis=0)
        _print("Power: ", power)

    # decode signal
    def lowpass_filter(self, data, fs=tf.F_SAMPLING, lowcut=tf.f_max, order=8):
        b, a = butter(order, lowcut, fs=fs, btype='low')

        filtered_data = lfilter(b, a, data)
        return filtered_data
    
    def highpass_filter(self, data, fs=tf.F_SAMPLING, highcut=tf.f_min, order=8):
        b, a = butter(order, highcut, fs=fs, btype='high')
        filtered_data = lfilter(b, a, data)
        return filtered_data

    def bandpass_filter(self, data, fs=tf.F_SAMPLING, lowcut=tf.f_min, highcut=tf.f_max, order=8):
        b, a = butter(order, [lowcut, highcut], fs=fs, btype='band')
        filtered_data = lfilter(b, a, data)
        return filtered_data

    # --------------- DECODING METHODS -----------------------------
    def max_peak(self):
        # Calculate the maximum peak
        max_peak_index = np.argmax(self.amplitude_envelope)
        for slot in intervals:
            if slot['start'] * tf.chirp_samples <= max_peak_index <= slot['end'] * tf.chirp_samples:
                return slot['data']
        return self.default_value

    def mean_peak_position(self, peaks):
        # calculate average position of the peaks
        mean_position = mean(t_frame, peaks)
        for slot in intervals:
            if slot['start'] * tf.t_slot <= mean_position <= slot['end'] * tf.t_slot:
                return slot['data']
        return self.default_value

    def mean_peak_height(self, peaks):
        # calculate the average height of the peaks in each slot
        mean_peaks = np.zeros(4)
        for i in range(4):
            peaks_slot = peaks[np.where((peaks >= i * tf.chirp_samples) & (peaks < (i + 1) * tf.chirp_samples))]
            mean_peaks[i] = mean(self.amplitude_envelope, peaks_slot)
        slot = np.argmax(mean_peaks)
        return intervals[slot]['data'] if slot is not None else self.default_value

    def peak_density(self, peaks):
        # Calculate the density of peaks in each slot
        peak_density = np.zeros(4)
        for i in range(4):
            peaks_slot = peaks[np.where((peaks >= i * tf.chirp_samples) & (peaks < (i + 1) * tf.chirp_samples))]
            peak_density[i] = len(peaks_slot)
        slot = np.argmax(peak_density)
        return intervals[slot]['data'] if slot is not None else self.default_value
    
    # -------------------------------------------------------------

    def decode_signal(self, signal : np.array, expected_signal=None):
        global chirp_signal

        if signal.size == 0:
            _print("Empty signal", file=sys.stderr)
            return None

        # filter and correlate signal
        if tf.BIO_SIGNALS:
            file_path = self.temp_files[expected_signal]
            correlation_signal, sr = librosa.load(file_path, sr=None)
        else:
            correlation_signal = chirp_signal

        if tf.f_max < tf.F_SAMPLING/2:
            filtered_signal = self.bandpass_filter(signal)
            self.correlation = correlate(filtered_signal, correlation_signal, mode='same')
        else:
            self.correlation = correlate(signal, correlation_signal, mode='same')

        # extract analytic signal and envelope
        self.amplitude_envelope = np.abs(hilbert(self.correlation))

        # thresholding
        mean_corr = np.mean(self.amplitude_envelope)
        corr_std = np.std(self.amplitude_envelope)
        threshold = mean_corr + 3 * corr_std # 3 sigma
        peaks, _ = find_peaks(self.amplitude_envelope, height=threshold)


        if peaks.size == 0:
            threshold = mean_corr + 0.5 * corr_std
            peaks, _ = find_peaks(self.amplitude_envelope, height=threshold)

        # media totale dei picchi
        '''
        mean_peak = np.mean(t_frame[peaks])
        if mean_peak is None:
            # append 00 if no peak found
            self.mean_peak_decoded = np.append(self.mean_peak_decoded, [0, 0])
        # check in which interval the peak is
        for lapse in self.tm.timeInterval:
            if lapse.start * tf.t_slot <= mean_peak <= lapse.end * tf.t_slot:
                self.mean_peak_decoded = np.append(self.mean_peak_decoded, lapse.data)

        # cerca picco massimo
        max_peak_index = np.argmax(self.amplitude_envelope)
        if max_peak_index is None:
            # append 00 if no peak found
            self.max_peak_decoded = np.append(self.max_peak_decoded, [0, 0])
        for lapse in self.tm.timeInterval:
            if lapse.start * tf.chirp_samples <= max_peak_index <= lapse.end * tf.chirp_samples:
                self.max_peak_decoded = np.append(self.max_peak_decoded, lapse.data)

        # media dei picchi per slot per trovare il picco più probabile
        mean_peaks = np.zeros(4)
        peak_density = np.zeros(4)
        for i in range(4):
            peaks_slot = peaks[np.where((peaks >= i * tf.chirp_samples) & (peaks < (i + 1) * tf.chirp_samples))]
            mean_peaks[i] = mean(self.amplitude_envelope, peaks_slot)
            peak_density[i] = len(peaks_slot)

        max_peak = np.argmax(mean_peaks)
        max_density = np.argmax(peak_density)
        if max_peak is None:
            # append 00 if no peak found
            self.slot_peak_decoded = np.append(self.slot_peak_decoded, [0, 0])
        self.slot_peak_decoded = np.append(self.slot_peak_decoded, self.tm.timeInterval[max_peak].data)

        if max_density is None:
            # append 00 if no peak found
            self.peak_density_decoded = np.append(self.peak_density_decoded, [0, 0])
        self.peak_density_decoded = np.append(self.peak_density_decoded, self.tm.timeInterval[max_density].data)
        '''

        # decode signal using different methods
        self.mean_peak_decoded = np.append(self.mean_peak_decoded, self.mean_peak_position(peaks))
        self.max_peak_decoded = np.append(self.max_peak_decoded, self.max_peak())
        self.slot_peak_decoded = np.append(self.slot_peak_decoded, self.mean_peak_height(peaks))
        self.peak_density_decoded = np.append(self.peak_density_decoded, self.peak_density(peaks))

        # plot correlation
        # disable plotting for BER/SNR simulation
        if tf.BER_SNR_SIMULATION:
            return
        if tf.PLOT:
            plt.figure()
            plt.plot(t_frame, self.amplitude_envelope)
            plt.plot(t_frame[peaks], self.amplitude_envelope[peaks], "x", color="red")
            plt.title("Correlation with Chirp")
            plt.xlabel("Time [s]")
            plt.ylabel("Correlation")
            plt.grid(True)
            if SAVE_IMG:
                timestamp = time.time()
                timestamp = str(timestamp).replace(".", "")
                plt.savefig(tf.img_directory + timestamp + ".png")
            else:
                plt.show()

    def save_to_csv(self, filename, counter, data):
        # Calculate the time values for the current slot
        time_values = self.x + counter * tf.T_FRAME
        # Prepare data rows in bulk
        rows = [[t, sig, corr] for t, sig, corr in zip(time_values, data, self.amplitude_envelope)]
        # Append rows to the CSV file
        with open(filename, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(rows)
            file.flush()

    def retrieve_signal(self, db_collection, data):  # retrieve audio data from database
        for i in data:
            # check if exist already in temp_files
            if i in self.temp_files:
                continue
            query = {"_id": int(i)}
            result = db_collection.find_one(query)

            if result and 'audio_data' in result:
                file_path = self.save_to_tempfile(result['_id'], result['audio_data'])
                self.temp_files[i] = file_path

    def save_to_tempfile(self, file_id, audio_data):
        os.makedirs("/tmp/receiver_cache/audio", exist_ok=True)
        file_path = f"/tmp/receiver_cache/audio/{file_id}.flac"
        with open(file_path, 'wb') as f:
            f.write(audio_data)
        return file_path

    def cleanup(self):
        # Deletes all temporary files after processing.
        for file_path in self.temp_files.values():
            if os.path.exists(file_path):
                os.remove(file_path)
        _print("Deleted temporary files")


if __name__ == "__main__":
    rc = Receiver()
    i = 0
    data = np.array([])
    expected_data = []

    with open(animation_file, 'w') as file:
        writer = csv.DictWriter(file, fieldnames=["time", "signal", "correlation"])
        writer.writeheader()

    if tf.ANIMATION:
        animation = multiprocessing.Process(target=run_script, args=("./receiver_animation.py",))
        animation.start()

        # connect to database

    if tf.BIO_SIGNALS:
        client = MongoClient(tf.uri, server_api=ServerApi('1'))
        try:
            client.admin.command('ping')
            _print("Pinged your deployment. You successfully connected to MongoDB!")
        except Exception as e:
            _print(e)
            exit(1)
        database = client['dolphin_database']
        collection = database['short_whistle_files_normalized']
        collection_size = collection.count_documents({})
        random.seed(tf.seed)
        expected_data = [random.randint(1, collection_size) for _ in range(int(tf.num_bits / 2))]
        _print("Expected data: ", expected_data)
        rc.retrieve_signal(collection, expected_data)

    try:
        while i < int(tf.num_bits/2): # or while True: to receive indefinitely
            data = np.array(rc.read())
            if data.size > 0:
                if tf.BIO_SIGNALS:
                    rc.decode_signal(data, expected_data[i])
                else:
                    rc.decode_signal(data)
                rc.save_to_csv(animation_file, i, data)
                #if tf.PLOT: rc.plot_spectrogram(rc.lowpass_filter(data))
                i += 1
            else:
                break
    # print error on stderr
    except KeyboardInterrupt:
        _print("Receiver interrupted", file=sys.stderr)
    except Exception as e:
        _print("Error: ", e, file=sys.stderr)
    finally:
        np.save(tf.res_directory + 'mean_peak.npy', rc.mean_peak_decoded)
        np.save(tf.res_directory + 'max_peak.npy', rc.max_peak_decoded)
        np.save(tf.res_directory + 'slot_peak.npy', rc.slot_peak_decoded)
        np.save(tf.res_directory + 'peak_density.npy', rc.peak_density_decoded)
        rc.channel.close()
        _print("Mean peak height: ", rc.slot_peak_decoded)
        _print("Mean peak position: ", rc.mean_peak_decoded)
        _print("Max peak: ", rc.max_peak_decoded)
        _print("Peak density: ", rc.peak_density_decoded)
        if not tf.BER_SNR_SIMULATION:
            _print("OFF")
        if tf.BIO_SIGNALS:
            rc.cleanup()
        if tf.ANIMATION: animation.join()
