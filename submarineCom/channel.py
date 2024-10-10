import os
import timeframe as tf
import numpy as np

fifo_path  = "/tmp/channel"

class Channel:
    def __delete__(self):
        if os.path.exists(fifo_path):
            os.remove(fifo_path)
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
            print(f"FIFO named {self.channel} is created successfully")
        else:
            print(f"FIFO named {self.channel} already exists")
        self.open(mode)


    def add_noise(self, signal):
        noise = np.random.normal(0, 0.5, len(signal))
        print(noise)
        signal = signal + noise
        return signal

    def send_data(self, data):
        self.fifo.write(data)
        self.fifo.flush() # assicura che i dati vengano scritti immediatamente , permette a receiver di leggere


    def send_signal(self, signal):
        noisy_signal = self.add_noise(signal)
        for i in range(tf.f_sampling * 4):
            formatted_data = f'{noisy_signal[i]:.5g}'
            data = str(formatted_data)+'\n'
            self.send_data(data)

    def read_data(self):
        data = self.fifo.readline().strip()
        return data

    def read_signal(self):
        signal = []
        for i in range(tf.f_sampling * 4):
            data = self.read_data()
            if data is not None:
                signal.append(data)
        return signal

    def close(self):
        self.fifo.close()
        self.__delete__()
