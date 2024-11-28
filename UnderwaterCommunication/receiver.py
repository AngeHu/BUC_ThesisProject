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
import multiprocessing

SAVE_IMG = tf.SAVE_IMG
img_directory = tf.img_directory


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
        self.tm = tf.TimeFrame()
        manager = multiprocessing.Manager()
        self.deciphered_mean_peak = manager.list()
        self.deciphered_max_peak = manager.list()
        self.deciphered_slot_peak = manager.list()


    def read(self):
        data = self.channel.read_signal()
        if data is None:
            return None
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
        nperseg = min(2048, len(signal))
        f, t, Sxx = spectrogram(signal,
                                fs=tf.sig_samples,
                                window='hamming',
                                nperseg=nperseg,
                                noverlap=2048 * 0.25,
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

    def decode_signal(self, signal, method=0, sigma=2):
        global chirp_signal
        if signal == []:
            print("Empty signal", file=sys.stderr)
            return None
        filtered_signal = self.lowpass_filter(signal)
        correlated_signal = correlate(filtered_signal, chirp_signal, mode='same')
        analytic_signal = hilbert(correlated_signal)
        amplitude_envelope = np.abs(analytic_signal)
        mean_corr = np.mean(amplitude_envelope)
        corr_std = np.std(amplitude_envelope)
        threshold = mean_corr + sigma * corr_std # 3 sigma
        peaks, _ = find_peaks(amplitude_envelope, height=threshold)

        if peaks.size == 0:
            print("No peaks found", file=sys.stderr)

        # Create processes for each of the decoding methods
        mean_peak_process = multiprocessing.Process(target=self._decipher_mean_peak,
                                                    args=(peaks,))
        max_peak_process = multiprocessing.Process(target=self._decipher_max_peak,
                                                   args=(amplitude_envelope,))
        slot_peak_process = multiprocessing.Process(target=self._decipher_slot_peak,
                                                    args=(peaks, amplitude_envelope,))
        mean_peak_process.start()
        max_peak_process.start()
        slot_peak_process.start()

        mean_peak_process.join(timeout=5)
        max_peak_process.join(timeout=5)
        slot_peak_process.join(timeout=5)

        '''
        # if method == 2:
            # print("Mean peak", file=sys.stderr)
            # media dei picchi per trovare il picco più probabile

            # search for chirp based on mean and std
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
            self.deciphered_mean_peak = np.append(self.deciphered_data, self.tm.lapse1.data)
        elif mean_peak < self.tm.lapse2.end*tf.t_slot and mean_peak > self.tm.lapse2.start*tf.t_slot:
            self.deciphered_mean_peak = np.append(self.deciphered_data, self.tm.lapse2.data)
        elif mean_peak < self.tm.lapse3.end*tf.t_slot and mean_peak > self.tm.lapse3.start*tf.t_slot:
            self.deciphered_mean_peak = np.append(self.deciphered_data, self.tm.lapse3.data)
        elif mean_peak < self.tm.lapse4.end*tf.t_slot and mean_peak > self.tm.lapse4.start*tf.t_slot:
            self.deciphered_mean_peak = np.append(self.deciphered_data, self.tm.lapse4.data)
        else:
            # print on stderr
            print("No valid peak found", file=sys.stderr)

        # cerca picco massimo
        # elif method == 1:
            # print("Max peak", file=sys.stderr)
        max_peak_index = np.argmax(amplitude_envelope)
        if max_peak_index < self.tm.lapse1.end*tf.chirp_samples and max_peak_index > self.tm.lapse1.start*tf.chirp_samples:
            self.deciphered_max_peak = np.append(self.deciphered_data, self.tm.lapse1.data)
        elif max_peak_index < self.tm.lapse2.end*tf.chirp_samples and max_peak_index > self.tm.lapse2.start*tf.chirp_samples:
            self.deciphered_max_peak = np.append(self.deciphered_data, self.tm.lapse2.data)
        elif max_peak_index < self.tm.lapse3.end*tf.chirp_samples and max_peak_index > self.tm.lapse3.start*tf.chirp_samples:
            self.deciphered_max_peak = np.append(self.deciphered_data, self.tm.lapse3.data)
        elif max_peak_index < self.tm.lapse4.end*tf.chirp_samples and max_peak_index > self.tm.lapse4.start*tf.chirp_samples:
            self.deciphered_max_peak = np.append(self.deciphered_data, self.tm.lapse4.data)
        else:
            # print on stderr
            print("No valid peak found", file=sys.stderr)

        # elif method == 3:
            # print("Slot peak", file=sys.stderr)
            # media dei picchi per trovare il picco più probabile in base a time slot

        mean_peaks = np.zeros(4)
        # slot 1
        peaks_slot1 = peaks[np.where(peaks < tf.chirp_samples)]
        mean_peaks[0] = mean(amplitude_envelope, peaks_slot1)
        # slot 2
        peaks_slot2 = peaks[np.where((peaks >= tf.chirp_samples) & (peaks < 2*tf.chirp_samples))]
        mean_peaks[1] = mean(amplitude_envelope, peaks_slot2)
        # slot 3
        peaks_slot3 = peaks[np.where((peaks >= 2*tf.chirp_samples) & (peaks < 3*tf.chirp_samples))]
        mean_peaks[2] = mean(amplitude_envelope, peaks_slot3)
        # slot 4
        peaks_slot4 = peaks[np.where(peaks >= 3*tf.chirp_samples)]
        mean_peaks[3] = mean(amplitude_envelope, peaks_slot4)

        max_peak = np.argmax(mean_peaks)
        if max_peak == 0:
            self.deciphered_slot_peak = np.append(self.deciphered_data, self.tm.lapse1.data)
        elif max_peak == 1:
            self.deciphered_slot_peak = np.append(self.deciphered_data, self.tm.lapse2.data)
        elif max_peak == 2:
            self.deciphered_slot_peak = np.append(self.deciphered_data, self.tm.lapse3.data)
        elif max_peak == 3:
            self.deciphered_slot_peak = np.append(self.deciphered_data, self.tm.lapse4.data)
        else:
            # print on stderr
            print("No valid peak found", file=sys.stderr)
        
        elif method == 0:
            print("No method selected", file=sys.stderr)
            exit(1)
        '''

        # plot correlation
        # disable plotting for BER/SNR simulation
        '''
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
        '''

    def _decipher_mean_peak(self, peaks):
        global t_frame
        mean_peak = np.mean(t_frame[peaks])
        if mean_peak is None:
            print("No mean peak found", file=sys.stderr)
            return
        # check in which interval the peak is
        for lapse in self.tm.timeInterval:
            if lapse.start * tf.t_slot <= mean_peak <= lapse.end * tf.t_slot:
                self.deciphered_mean_peak.extend(lapse.data)


    def _decipher_max_peak(self, amplitude_envelope):
        max_peak_index = np.argmax(amplitude_envelope)
        if max_peak_index is None:
            print("No max index peaks found", file=sys.stderr)
            return
        # check in which interval the peak is
        for lapse in self.tm.timeInterval:
            if lapse.start * tf.chirp_samples <= max_peak_index <= lapse.end * tf.chirp_samples:
                self.deciphered_max_peak.extend(lapse.data)

    def _decipher_slot_peak(self, peaks, amplitude_envelope):
        mean_peaks = np.zeros(4)
        for i in range(4):
            peaks_slot = peaks[np.where((peaks >= i * tf.chirp_samples) & (peaks < (i + 1) * tf.chirp_samples))]
            mean_peaks[i] = mean(amplitude_envelope, peaks_slot)

        max_peak = np.argmax(mean_peaks)
        if max_peak is None:
            print("No peaks found", file=sys.stderr)
            return
        self.deciphered_slot_peak.extend(self.tm.timeInterval[max_peak].data)





if __name__ == "__main__":
    rc = Receiver()
    i = 0
    data = np.array([])
    '''
    if tf.BER_SNR_SIMULATION:
        method = int(sys.argv[1])
    else:
        method = 1 if tf.MAX_PEAK else 2 if tf.MEAN_PEAK else 3 if tf.SLOT_PEAK else 0
    '''
    try:
        while True:
            data = rc.read()
            if data is not None:
                rc.decode_signal(data)
            else:
                break
    except Exception as e:
        print("Error: ", e, file=sys.stderr)
    finally:
        print("Mean peak decoded:", rc.deciphered_mean_peak)
        print("Max peak decoded:", rc.deciphered_max_peak)
        print("Slot peak decoded:", rc.deciphered_slot_peak)
        rc.channel.close()
        if not tf.BER_SNR_SIMULATION:
            print("Receiver OFF")
        exit(0)
