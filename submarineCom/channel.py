import os
import params as tf
import numpy as np

fifo_path  = "/tmp/channel"

class Channel:
    def __delete__(self):
        if os.path.exists(fifo_path):
            os.remove(fifo_path)
            if not tf.BER_SNR_SIMULATION:
                print(f"Channel named {fifo_path} is deleted successfully")

    def open(self, mode):
        if mode == 'r' or mode == 'w':
            self.fifo = open(self.channel, mode)
        else:
            print("Mode not valid")
            self.__delete__()
            exit(1)

    def __init__(self, mode):
        # start communication - create pipe channel
        self.channel = fifo_path
        self.fifo = None
        # permission
        self.permission = 0o600
        if not(os.path.exists(self.channel)):
            os.mkfifo(self.channel, self.permission)
            if not tf.BER_SNR_SIMULATION: print(f"FIFO named {self.channel} is created successfully")
        else:
            if not tf.BER_SNR_SIMULATION: print(f"FIFO named {self.channel} already exists")
        self.open(mode)

    def add_noise(self, signal, noise_level):
        noise = np.random.normal(0, 1, len(signal)) * np.sqrt(noise_level)
        # power_noise calcolato prendendo un quarto del rumore e diviso per lunghezza del chirp
        if tf.DEBUG and not tf.BER_SNR_SIMULATION:
            print("Power noise (measured): ", sum(noise[0:tf.f_sampling]**2) / tf.f_sampling) #se diviso per len(signal) da la metà
            print("Check SNR: ", sum(signal**2) / sum(noise[0:tf.f_sampling]**2))
        signal = signal + noise
        return signal
        

    def send_data(self, data):
        try:
            self.fifo.write(data)
            self.fifo.flush()  # Ensure data is written immediately
        except BrokenPipeError:
            if not tf.BER_SNR_SIMULATION: print("BrokenPipeError: The receiver has closed the pipe.")
            self.fifo.close()

    def send_signal(self, signal, noise_level):
        noisy_signal = self.add_noise(signal, noise_level)
        for i in range(tf.f_sampling * 4):
            formatted_data = f'{noisy_signal[i]:.5g}'
            data = str(formatted_data)+'\n'
            self.send_data(data)

    def read_data(self):
        try:
            data = self.fifo.readline().strip()
            return float(data)
        except BrokenPipeError:
            if not tf.BER_SNR_SIMULATION: print("BrokenPipeError: The transmitter has closed the pipe.")
            self.fifo.close()
            return None

    def read_signal(self):
        signal = []
        for i in range(tf.f_sampling * 4):
            data = self.read_data()
            if data is not None:
                signal.append(data)
        return signal

    def close(self):
        self.fifo.close()
