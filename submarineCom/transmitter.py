# transmitter
import numpy as np
from matplotlib import pyplot as plt
import channel
import params as tf
import time

# plot every 4*2 bits
def plot_function(x, y_freq, y_sig):
    figure, (ax1, ax2) = plt.subplots(2, 1)
    freq, = ax1.plot(x, y_freq, color='g', label='Frequenza')  # Crea il grafico
    sig, = ax2.plot(x, y_sig, color='r', label='Segnale')  # Crea il grafico

    ax1.set_xlabel('Time(s)')  # Aggiunge un'etichetta all'asse x
    ax1.set_ylabel('Frequenza')  # Aggiunge un'etichetta all'asse y1
    ax2.set_xlabel('Time(s)')  # Aggiunge un'etichetta all'asse x
    ax2.set_ylabel('Segnale')  # Aggiunge un'etichetta all'asse y2
    ax1.set_xlim(0, 4*tf.T_frame)
    ax1.set_ylim(0, 50000)
    ax2.set_xlim(0, 4*tf.T_frame)
    ax2.set_ylim(-1, 1)
    plt.grid(True)  # Aggiunge una griglia al grafico (opzionale)
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


class Transmitter():

    def __init__(self):
        self.channel = channel.Channel('w')
        print("Transmitter ON")
        self.x = np.linspace(0, tf.T_frame, tf.f_sampling*4)
        self.slot = np.linspace(0, tf.t_slot, tf.f_sampling)
        self.signal = []
        self.frequency = []

    # must visualize 2 graphs: one for frequency and one for signal
    def generate_frequency(self, interval):
        self.frequency = np.full(tf.f_sampling * 4, 0)
        self.frequency[interval.start*tf.f_sampling: interval.end*tf.f_sampling] = np.linspace(tf.f_min, tf.f_max, tf.f_sampling)
        #plot_function(self.x, self.frequency, "Frequenza", "t", "f", False)
        return self.frequency

    def generate_chirp(self, interval):
        self.signal = np.sin(2 * np.pi * self.frequency * (self.x - interval.start * tf.t_slot))
        E_signal = sum(self.signal**2)/ tf.f_sampling
        return self.signal, E_signal

    def send_signal(self, noise):
        self.channel.send_signal(self.signal, noise)

    def generate_signal(self, slot):
        self.generate_frequency(slot)
        self.generate_chirp(slot)


if __name__ == "__main__":
    # data = open("data.txt", 'r', encoding='utf-8')
    tm = tf.TimeFrame()
    transmitter = Transmitter()
    file = open("test/test1.txt", 'r')
    # read data from file adn put in a string
    data = file.read()
    print("Data: ", data)

    i = 0
    # info = "00100100"
    data = encode_signal(data)
    frq = np.array([])
    sig = np.array([])
    while(i < len(data)):
        frq = np.append(frq, transmitter.generate_frequency(tm.timeInterval[data[i]]))
        generated_sig, E_signal = transmitter.generate_chirp(tm.timeInterval[data[i]])
        sig = np.append(sig, generated_sig)
        #print("Data: ", data[i])
        print("Power signal: ", E_signal)
        print("Power noise: ", E_signal/tf.SNR)
        # Attenzione al fattore 4!!
        transmitter.send_signal((E_signal)/tf.SNR)
        i = i + 1
        if i % 4 == 0:

            plot_function(np.linspace(0, 4*tf.T_frame, 4*4*tf.f_sampling), frq, sig)
            frq = np.array([])
            sig = np.array([])
    time.sleep(10);







