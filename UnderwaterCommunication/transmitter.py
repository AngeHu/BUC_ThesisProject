# transmitter
import numpy as np
from matplotlib import pyplot as plt
import channel
import params as tf
import time
import sys
from scipy.signal import chirp

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

t_slot = np.linspace(0, tf.t_slot, tf.chirp_samples) # vettore tempo
t_frame = np.linspace(0, tf.T_frame, tf.sig_samples) # vettore tempo
chirp_signal = chirp(t_slot, f0=tf.f_min, f1=tf.f_max, t1=tf.t_slot, method='linear') # segnale chirp
e_signal = sum(chirp_signal**2) / tf.chirp_samples # potenza segnale

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
    plt.tight_layout() # Aggiusta il layout per fare spazio alle etichette
    plt.show()  # Mostra il grafico


def encode_signal(data):
    encoded_signal = []
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
    return encoded_signal


class Transmitter():
    def __init__(self):
        self.channel = channel.Channel("wb")
        if not tf.BER_SNR_SIMULATION: print("Transmitter ON")
        self.x = np.linspace(0, tf.T_frame, tf.sig_samples)
        self.slot = np.linspace(0, tf.t_slot, tf.chirp_samples)
        self.signal = []
        self.frequency = []

    # must visualize 2 graphs: one for frequency and one for signal
    def generate_frequency(self, interval):
        self.frequency = np.full(tf.sig_samples, 0)
        self.frequency[interval.start*tf.chirp_samples: interval.end*tf.chirp_samples] = np.linspace(tf.f_min, tf.f_max, tf.chirp_samples)
        #plot_function(self.x, self.frequency, "Frequenza", "t", "f", False)
        return self.frequency

    def generate_chirp(self, interval):
        self.signal = np.sin(2 * np.pi * self.frequency * (self.x - interval.start * tf.t_slot))
        e_signal = sum(self.signal**2) / tf.chirp_samples
        return self.signal, e_signal

    def send_signal(self, noise):
        self.channel.send_signal(self.signal, noise)

    def generate_signal(self, slot):
        self.generate_frequency(slot)
        self.generate_chirp(slot)


if __name__ == "__main__":
    if tf.BER_SNR_SIMULATION:
        # generate bit sequence
        data = np.random.randint(0, 2, tf.num_bits)
        print(data)
        SNR = float(sys.argv[1])
        # turn snr to linear scale
        SNR = 10 ** (SNR / 10)
    else:
        #file = open("test/test1.txt", 'r')
        # read data from file adn put in a string
        #read_data = file.read()
        #data = [int(char) for char in read_data]
        data = np.random.randint(0, 2, tf.num_bits)
        print("Data: ", data)
        SNR = tf.SNR
    tm = tf.TimeFrame()
    transmitter = Transmitter()

    i = 0
    data = encode_signal(data)
    frq = np.array([])
    sig = np.array([])

    while i < len(data):
        #frq = np.append(frq, transmitter.generate_frequency(tm.timeInterval[data[i]]))
        #generated_sig, E_signal = transmitter.generate_chirp(tm.timeInterval[data[i]])
        #sig = np.append(sig, generated_sig)
        transmitter.generate_signal(tm.timeInterval[data[i]])
        #if tf.DEBUG:
            #print("Power signal: ", E_signal)
            #print("Power noise: ", E_signal / SNR)
        transmitter.send_signal(e_signal / SNR) # send signal with noise
        i += 1
        print("msg ", i)
        '''
        if i % 4 == 0: # plot every 4 signal
            if not tf.BER_SNR_SIMULATION: plot_function(np.linspace(0, 4 * tf.T_frame, 4 * 4 * tf.chirp_samples), frq, sig)
            frq = np.array([])
            sig = np.array([])
        '''

    # transmitter.channel.send_data("EOF")

    time.sleep(1) # wait for receiver to finish

    transmitter.channel.close()

    if not tf.BER_SNR_SIMULATION:
        print("Transmitter OFF")






