# receiver deve leggere da fifo in tempo reale...
# aggiugerò in futuro il ritardo, che va calcolato in base a callibrazione
# receiver per ora riceve in tempo reale e deve salvare i dati in un array
# receiver deve riuscire a ricostruire correttamente ilsegnale..più o meno
# sicruamente ci sarà perdita di segnale!
import time

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import channel
import params as tf
from scipy.signal import chirp, spectrogram, correlate, stft
from scipy.fft import fftshift

# increase agg.path.chunksize
plt.rcParams['agg.path.chunksize'] = 10000

t = np.linspace(0, tf.t_slot, tf.f_sampling) # vettore tempo
chirp_signal = chirp(t, f0=tf.f_min, f1=tf.f_max, t1=tf.t_slot, method='linear') # segnale chirp

def plot_function(x, y_sig):
    if(len(x) != len(y_sig)):
        print("Errore: dimensioni di x e y non coincidono")
        return
    figure, ax = plt.subplots()
    sig, = ax.plot(x, y_sig, color='r', label='Segnale')  # Crea il grafico

    ax.set_xlabel('Time(s)')  # Aggiunge un'etichetta all'asse x
    ax.set_ylabel('Frequenza')  # Aggiunge un'etichetta all'asse y1
    ax.set_xlim(0, 4*tf.T_frame)
    ax.set_ylim(-5, 5)
    plt.grid(True)  # Aggiunge una griglia al grafico (opzionale)
    plt.show()  # Mostra il grafico

class Receiver:
    def __init__(self):
        self.channel = channel.Channel('r')
        print("Receiver ON")
        self.channel.open('r')
        self.correlation = []
        self.deciphered_data = []


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
        lags = np.arange(-len(signal) + 1, len(chirp_signal)) / (4 * tf.f_sampling)
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
        # f, t, Sxx = stft(signal, fs=tf.f_sampling, nperseg=256)
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

    def moving_average(self, signal, window_size):
        return np.convolve(signal, np.ones(window_size) / window_size, mode='same')

    def pulse_detection(self, signal):
        pass

    def decipher(self, signal):
        self.deciphered_data = np.append(self.deciphered_data, 10)


if __name__ == "__main__":
    rc = Receiver()
    i = 0
    data = np.array([])
    data_str = rc.read()
    try:
        while True:
            data_str = rc.read()
            if data_str is not []:
                float_data = [float(i) for i in data_str]
                data = np.append(data, float_data)
                rc.decipher(float_data)

                if len(data) >= 4 * 4 * tf.T_frame * tf.f_sampling and not tf.BER_SNR_SIMULATION:
                    # plot data
                    rc.plot_data(data)

                    # correlation
                    rc.plot_correlation(data)

                    # spectrogram
                    rc.plot_spectrogram(data)
                    #time.sleep(5)
                    data = np.array([])
    finally:
        rc.channel.close()
        if tf.BER_SNR_SIMULATION:
            print(rc.deciphered_data)
        else:
            print("Receiver OFF")
        exit(0)
