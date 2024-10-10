# receiver deve leggere da fifo in tempo reale...
# aggiugerò in futuro il ritardo, che va calcolato in base a callibrazione
# receiver per ora riceve in tempo reale e deve salvare i dati in un array
# receiver deve riuscire a ricostruire correttamente ilsegnale..più o meno
# sicruamente ci sarà perdita di segnale!

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import channel
import params as tf
from scipy.signal import chirp, stft, correlate
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

def plot_correlation(signal: np.array):
    global chirp_signal
    correlation = correlate(signal, chirp_signal, mode='full')
    lags = np.arange(-len(signal) + 1, len(chirp_signal))/(4*tf.f_sampling)
    #lags = np.arange(0, len(signal))
    plt.figure()
    plt.plot(lags, correlation)
    plt.title('Cross-Correlation between Signal and Chirp')
    plt.xlabel('Lag')
    plt.grid(True)
    plt.show()

def plot_spectrogram(signal: np.array):
    f, t, Sxx = stft(signal, fs=tf.f_sampling, window='hamming', nperseg=128, noverlap=64)
    Sxx_magnitude = np.abs(Sxx)
    plt.pcolormesh(t, f, Sxx_magnitude)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

class Receiver:
    def __init__(self):
        self.channel = channel.Channel('r')
        print("Receiver ON")
        self.channel.open('r')

    def read(self):
        data = self.channel.read_signal()
        return data

    def plot_data(self, data):
        print("Data length: ", len(data))
        x = np.linspace(0, 4*tf.T_frame, len(data))
        plot_function(x, data)

    def decipher(self, signal):
        correlated_signal = correlate(signal, chirp_signal, mode='full')
        lags = np.arange(-len(signal) + 1, len(chirp_signal))/(tf.f_sampling)
        threshold = 0.8 * np.max(correlated_signal)  # Set a threshold at 80% of the maximum correlation value
        detected_positions = lags[np.where(correlated_signal > threshold)]  # Find lags where the correlation exceeds the threshold
        # print("Detected Chirp positions (in samples):", detected_positions)


if __name__ == "__main__":
    info = "00100100"
    rc = Receiver()
    i = 0
    data = np.array([])
    while True:
        data_str = rc.read()
        if data_str is not []:
            float_data = [float(i) for i in data_str]
            data = np.append(data, float_data)
            rc.decipher(float_data)
            i += 1
            if i % 4==0:
                print(i)
                # plot data
                rc.plot_data(data)

                # correlation
                plot_correlation(data)

                # spectrogram
                plot_spectrogram(data)
                data = np.array([])
