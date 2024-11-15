# receiver deve leggere da fifo in tempo reale...
# aggiugerò in futuro il ritardo, che va calcolato in base a callibrazione
# receiver per ora riceve in tempo reale e deve salvare i dati in un array
# receiver deve riuscire a ricostruire correttamente ilsegnale..più o meno
# sicruamente ci sarà perdita di segnale!
import time
import numpy as np
from matplotlib import pyplot as plt
import channel
import params as tf
from scipy.signal import chirp, spectrogram, correlate, stft, hilbert
from scipy.fft import fftshift
from scipy.signal import butter, lfilter, find_peaks
import sys

if tf.DEBUG:
    import cProfile
    import atexit
    import pstats

    # Set up profiling to save to a file
    profiler = cProfile.Profile()
    profiler.enable()

    # Save profile results to a file on exit
    def save_profile():
        with open('receiver_profile.prof', 'w') as f:  # Use 'receiver_profile.prof' for receiver.py
            ps = pstats.Stats(profiler, stream=f)
            ps.strip_dirs().sort_stats('cumulative').print_stats()

    atexit.register(save_profile)

SAVE_IMG = tf.SAVE_IMG
img_directory = tf.img_directory
METHOD = 1 if tf.MAX_PEAK else 2 if tf.MEAN_PEAK else 3 if tf.SLOT_PEAK else 0

# increase agg.path.chunksize
plt.rcParams['agg.path.chunksize'] = 10000

t_slot = np.linspace(0, tf.t_slot, tf.chirp_samples) # vettore tempo
t_frame = np.linspace(0, tf.T_frame, tf.sig_samples) # vettore tempo
chirp_signal = chirp(t_slot, f0=tf.f_min, f1=tf.f_max, t1=tf.t_slot, method='linear') # segnale chirp

def mean(x, indices):
    if indices.size == 0:
        return 0
    return np.mean(x[indices])

def plot_function(x, y_sig):
    if(len(x) != len(y_sig)):
        print("Errore: dimensioni di x e y non coincidono")
        return
    figure, ax = plt.subplots()
    sig, = ax.plot(x, y_sig, color='r', label='Segnale')  # Crea il grafico

    ax.set_xlabel('Time(s)')  # Aggiunge un'etichetta all'asse x
    ax.set_ylabel('Ampiezza')  # Aggiunge un'etichetta all'asse y1
    ax.set_xlim(0, 4*tf.T_frame)
    ax.set_ylim(-5, 5)
    plt.grid(True)  # Aggiunge una griglia al grafico (opzionale)
    plt.show()  # Mostra il grafico

class Receiver:
    def __init__(self):
        self.channel = channel.Channel('r')
        if not tf.BER_SNR_SIMULATION: print("Receiver ON")
        self.channel.open('r')
        self.correlation = []
        self.deciphered_data = np.array([])
        self.tm = tf.TimeFrame()


    def read(self):
        data = self.channel.read_signal()
        return data

    def plot_data(self, data):
        print("Data length: ", len(data))
        x = np.linspace(0, 4*tf.T_frame, len(data))
        plot_function(x, data)

    def plot_correlation(self, signal: np.array):
        print("Plotting correlation")
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
        print("Plotting spectrogram")
        print("Signal length: ", len(signal))
        f, t, Sxx = spectrogram(signal, fs=tf.f_sampling, window='hamming', nperseg=2048, noverlap=2048 * 0.25,
                                nfft=2048)
        # f, t, Sxx = stft(signal, fs=tf.sig_samples, nperseg=256)
        # Sxx_magnitude = np.abs(Sxx)
        # plt.pcolormesh(t, f, Sxx_magnitude)
        # plt.ylabel('Frequency [Hz]')
        # plt.xlabel('Time [sec]')
        # plt.show()
        plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap="viridis", vmin=-150, vmax=0)
        # plt.pcolormesh(t, f, Sxx_magnitude, shading='gouraud', cmap="viridis", vmin=-150, vmax=0)
        plt.colorbar(label="Power/Frequency (dB/Hz)")
        plt.title("Spectrogram of Noisy Signal")
        plt.ylabel("Frequency [Hz]")
        plt.xlabel("Time [s]")
        plt.tight_layout()  # Adjust layout to make room for the labels
        plt.show()

    # decode signal
    def lowpass_filter(self, data, fs=tf.f_sampling, lowcut=tf.f_max, order=8):
        b, a = butter(order, lowcut, fs=fs, btype='low')

        filtered_data = lfilter(b, a, data)
        return filtered_data

    def decode_signal(self, signal):
        filtered_signal = self.lowpass_filter(signal)
        correlated_signal = correlate(filtered_signal, chirp_signal, mode='same')
        analytic_signal = hilbert(correlated_signal)
        amplitude_envelope = np.abs(analytic_signal)

        if METHOD == 2:
            # media dei picchi per trovare il picco più probabile

            # search for chirp based on mean and std
            mean_corr = np.mean(amplitude_envelope)
            corr_std = np.std(amplitude_envelope)
            threshold = mean_corr + 3 * corr_std # 3 sigma
            peaks, _ = find_peaks(amplitude_envelope, height=threshold)
            mean_peak = np.mean(t_frame[peaks])

            # search for chirp based on peak density

            if tf.DEBUG:
                print("Mean correlation: ", mean_corr)
                print("Correlation std: ", corr_std)
                print("Threshold: ", threshold)
                # decode signal
                print("Mean peak: ", mean_peak)

            # check in which interval the peak is
            if mean_peak < self.tm.lapse1.end*tf.t_slot and mean_peak > self.tm.lapse1.start*tf.t_slot:
                self.deciphered_data = np.append(self.deciphered_data, self.tm.lapse1.data)
            elif mean_peak < self.tm.lapse2.end*tf.t_slot and mean_peak > self.tm.lapse2.start*tf.t_slot:
                self.deciphered_data = np.append(self.deciphered_data, self.tm.lapse2.data)
            elif mean_peak < self.tm.lapse3.end*tf.t_slot and mean_peak > self.tm.lapse3.start*tf.t_slot:
                self.deciphered_data = np.append(self.deciphered_data, self.tm.lapse3.data)
            elif mean_peak < self.tm.lapse4.end*tf.t_slot and mean_peak > self.tm.lapse4.start*tf.t_slot:
                self.deciphered_data = np.append(self.deciphered_data, self.tm.lapse4.data)
            else:
                # print on stderr
                print("No valid peak found", file=sys.stderr)

        # cerca picco massimo
        elif METHOD == 1:
            max_peak_index = np.argmax(amplitude_envelope)
            if max_peak_index < self.tm.lapse1.end*tf.chirp_samples and max_peak_index > self.tm.lapse1.start*tf.chirp_samples:
                self.deciphered_data = np.append(self.deciphered_data, self.tm.lapse1.data)
            elif max_peak_index < self.tm.lapse2.end*tf.chirp_samples and max_peak_index > self.tm.lapse2.start*tf.chirp_samples:
                self.deciphered_data = np.append(self.deciphered_data, self.tm.lapse2.data)
            elif max_peak_index < self.tm.lapse3.end*tf.chirp_samples and max_peak_index > self.tm.lapse3.start*tf.chirp_samples:
                self.deciphered_data = np.append(self.deciphered_data, self.tm.lapse3.data)
            elif max_peak_index < self.tm.lapse4.end*tf.chirp_samples and max_peak_index > self.tm.lapse4.start*tf.chirp_samples:
                self.deciphered_data = np.append(self.deciphered_data, self.tm.lapse4.data)
            else:
                # print on stderr
                print("No valid peak found", file=sys.stderr)

        elif METHOD == 3:
            # media dei picchi per trovare il picco più probabile in base a time slot

            mean_corr = np.mean(amplitude_envelope)
            corr_std = np.std(amplitude_envelope)
            threshold = mean_corr + 3 * corr_std
            peaks, _ = find_peaks(amplitude_envelope, height=threshold)

            mean_peaks = np.zeros(4)
            # slot 1
            peaks_slot1 = peaks[np.where(peaks < tf.chirp_samples)]
            mean_peaks[0] = mean(t_frame, peaks_slot1)
            # slot 2
            peaks_slot2 = peaks[np.where((peaks >= tf.chirp_samples) & (peaks < 2*tf.chirp_samples))]
            mean_peaks[1] = mean(t_frame, peaks_slot2)
            # slot 3
            peaks_slot3 = peaks[np.where((peaks >= 2*tf.chirp_samples) & (peaks < 3*tf.chirp_samples))]
            mean_peaks[2] = mean(t_frame, peaks_slot3)
            # slot 4
            peaks_slot4 = peaks[np.where(peaks >= 3*tf.chirp_samples)]
            mean_peaks[3] = mean(t_frame, peaks_slot4)

            max_peak = np.argmax(mean_peaks)
            if max_peak == 0:
                self.deciphered_data = np.append(self.deciphered_data, self.tm.lapse1.data)
            elif max_peak == 1:
                self.deciphered_data = np.append(self.deciphered_data, self.tm.lapse2.data)
            elif max_peak == 2:
                self.deciphered_data = np.append(self.deciphered_data, self.tm.lapse3.data)
            elif max_peak == 3:
                self.deciphered_data = np.append(self.deciphered_data, self.tm.lapse4.data)
            else:
                # print on stderr
                print("No valid peak found", file=sys.stderr)

        elif METHOD == 0:
            print("No method selected", file=sys.stderr)
            exit(1)

        # plot correlation

        # disable plotting for BER/SNR simulation
        plt.figure()
        plt.plot(t_frame, amplitude_envelope)
        plt.plot(t_frame[peaks], amplitude_envelope[peaks], "x", color="red")
        plt.title("Correlation with Chirp")
        plt.xlabel("Time [s]")
        plt.ylabel("Correlation")
        plt.grid(True)
        if SAVE_IMG:
            timestamp = time.time()
            timestamp = str(timestamp).replace(".", "")
            plt.savefig(img_directory + timestamp + ".png")
        else:
            plt.show()




if __name__ == "__main__":
    rc = Receiver()
    i = 0
    data = np.array([])
    try:
        while True:
            data_str = rc.read()
            if data_str:
                float_data = [float(i) for i in data_str]
                data = np.append(data, float_data)
                rc.decode_signal(float_data)

                if len(data) >= 4 * tf.T_frame * tf.sig_samples and not tf.BER_SNR_SIMULATION:
                    # plot data
                    #rc.plot_data(data)

                    # correlation
                    #rc.plot_correlation(data)

                    # spectrogram
                    #rc.plot_spectrogram(data)
                    #time.sleep(5)
                    data = np.array([])
                i += 1
    # print error on stderr
    except KeyboardInterrupt:
        print("Receiver interrupted", file=sys.stderr)
    except Exception as e:
        print("Error: ", e, file=sys.stderr)
    finally:
        rc.channel.close()
        print(rc.deciphered_data)
        if not tf.BER_SNR_SIMULATION:
            print("Receiver OFF")
        exit(0)
