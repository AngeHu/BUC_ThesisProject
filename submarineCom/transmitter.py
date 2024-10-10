# transmitter
import numpy as np
from matplotlib import pyplot as plt
import channel
import timeframe
import time
from scipy.signal import chirp, spectrogram
import threading
import plot
import queue


shared_data = {
    'signal': queue.Queue(),
    'frequency' : queue.Queue(),
    'data_available' : threading.Event()
}

def run_plot():
    a = plot.Animation()
    a.animate()

def plot_function(x, y, title='Grafico', x_label='X', y_label='Y', time_slice = True):
    plt.figure(figsize=(20, 8))  # Imposta le dimensioni del grafico (opzionale)
    plt.plot(x, y, label='Dati')  # Crea il grafico

    plt.title(title)  # Aggiunge un titolo al grafico
    plt.xlabel(x_label)  # Aggiunge un'etichetta all'asse x
    plt.ylabel(y_label)  # Aggiunge un'etichetta all'asse y
    if time_slice:
        plt.xlim(0, timeframe.T_frame)
        plt.ylim(0, timeframe.f_max)
    plt.legend()  # Aggiunge una legenda al grafico (opzionale)

    plt.grid(True)  # Aggiunge una griglia al grafico (opzionale)

    # animate = FuncAnimation(fig, animate, frames=20, interval=500, repeat=False)
    plt.show()  # Mostra il grafico


def encode_signal(data):
    encoded_signal = []
    for i in range(0, len(data), 2):
        if data[i] == '0' and data[i+1] == '0':
            encoded_signal.append(0)
        elif data[i] == '0' and data[i+1] == '1':
            encoded_signal.append(1)
        elif data[i] == '1' and data[i+1] == '1':
            encoded_signal.append(2)
        elif data[i] == '1' and data[i+1] == '0':
            encoded_signal.append(3)
    print("Encoded signal: ", encoded_signal)
    return encoded_signal


class Transmitter:

    def __init__(self):
        self.channel = channel.Channel('w')
        print("Transmitter ON")
        self.x = np.linspace(0, timeframe.T_frame, timeframe.f_sampling*4)
        self.slot = np.linspace(0, timeframe.t_slot, timeframe.f_sampling)
        self.signal = []
        self.frequency = []

    # must visualize 2 graphs: one for frequency and one for signal
    def generate_frequency(self, interval):
        self.frequency = np.full(timeframe.f_sampling * 4, timeframe.f_min)
        self.frequency[interval.start: interval.end] = np.linspace(timeframe.f_min, timeframe.f_max, timeframe.f_sampling)
        #plot_function(self.x, self.frequency, "Frequenza", "t", "f", False)
        return self.frequency

    def generate_chirp(self, interval):
        self.signal = np.sin(self.frequency * (self.x-interval.start))
        # print("Signal:", self.y_signal)
        #plot_function(self.x, self.signal, "Segnale", "t", "E", False)
        return self.signal

    def send_signal(self):
        self.channel.send_signal(self.signal)

    def generate_signal(self, slot):
        self.slot = np.linspace(slot*timeframe.t_slot, (slot+1)*timeframe.t_slot, timeframe.f_sampling)
        sig_chirp = chirp(self.slot, f0=timeframe.f_min, t1=(slot+1)*timeframe.t_slot, f1=timeframe.f_max, method='linear')
        #plot_function(self.slot, sig_chirp, "Chirp", "t", "E", False)
        #self.signal = np.sin(timeframe.f_min*self.x)
        #self.signal[slot*timeframe.f_sampling: (slot+1)*timeframe.f_sampling] = sig_chirp
        #plot_function(self.x, self.signal, "Segnale", "t", "E", False)
        return self.signal



if __name__ == "__main__":
    # data = open("data.txt", 'r', encoding='utf-8')
    tm = timeframe.TimeFrame()
    transmitter = Transmitter()
    plot = threading.Thread(target=run_plot, args=())
    i = 0
    info = "00100100"
    data = encode_signal(info)

    plot.start()
    while(i < len(data)):
        transmitter.generate_frequency(tm.timeInterval[data[i]])
        transmitter.generate_chirp(tm.timeInterval[data[i]])
        print("Data: ", data[i])
        #transmitter.send_signal()
        i = i + 1
        time.sleep(10)




