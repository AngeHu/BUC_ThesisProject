import os
import params as tf
import numpy as np
import sys
import struct

fifo_path = "/tmp/channel"

batch_size = tf.chirp_samples * 4

class Channel:
    def __delete__(self):
        if os.path.exists(fifo_path):
            os.remove(fifo_path)
            print(f"Channel named {fifo_path} is deleted successfully", file=sys.stderr)

    def open(self, mode):
        print(f"Opening FIFO named {self.channel} in mode {mode}", file=sys.stderr)
        if mode == "rb" or mode == "wb":
            self.fifo = open(self.channel, mode)
        else:
            print("Mode not valid", file=sys.stderr)
            self.__delete__()
            exit(1)

    def __init__(self, mode):
        # start communication - create pipe channel
        self.channel = fifo_path
        self.fifo = None
        # permission
        self.permission = 0o600
        if not (os.path.exists(self.channel)):
            os.mkfifo(self.channel, self.permission)
            print(f"FIFO named {self.channel} is created successfully", file=sys.stderr)
        else:
            print(f"FIFO named {self.channel} already exists", file=sys.stderr)
        self.open(mode)

    @staticmethod
    def add_noise(signal, noise_level):
        noise = np.random.normal(0, 1, len(signal)) * np.sqrt(noise_level)
        # power_noise calcolato prendendo un quarto del rumore e diviso per lunghezza del chirp
        '''
        if tf.DEBUG:
            print("Power noise (measured): ", sum(noise[0:tf.sig_samples] ** 2) / tf.sig_samples,
                  file=sys.stderr)  # se diviso per len(signal) da la metà
            print("Check SNR: ", sum(signal ** 2) / sum(noise[0:tf.sig_samples] ** 2), file=sys.stderr)
        '''
        signal = signal + noise
        return signal

    def send_data(self, data):
        try:
            self.fifo.write(data)
            self.fifo.flush()  # Ensure data is written immediately
        except BrokenPipeError:
            print("BrokenPipeError: The receiver has closed the pipe.", file=sys.stderr)
            self.fifo.close()

    def send_signal(self, signal, noise_level):
        noisy_signal = self.add_noise(signal, noise_level)
        # signal_str = '\n'.join([f'{x:.5g}' for x in noisy_signal]) + '\n' # Convert signal to string
        signal_str = struct.pack(f'{len(noisy_signal)}f', *noisy_signal)
        self.send_data(signal_str)

    def read_data(self): # deprecated
        try:
            data = self.fifo.readline()
            if data == 'EOF':
                print("End of communication", file=sys.stderr)
                self.fifo.close()
                return None
            else:
                data.strip()
            return float(data)
        except BrokenPipeError:
            if not tf.BER_SNR_SIMULATION: print("BrokenPipeError: The transmitter has closed the pipe.")
            self.fifo.close()
            return None

    '''
    def read_signal(self):

        signal = []
        for i in range(tf.sig_samples):
            data = self.read_data()
            if data is not None:
                signal.append(data)
            else:
                return None
        return signal

    '''

    def read_signal(self, batch_size=batch_size): #batch no more than 400
        signal = []
        try:
            # Read 10 lines at a time until we get `tf.sig_samples` samples or EOF
            while len(signal) < tf.sig_samples:
                '''
                # Read up to 10 lines
                lines = [self.fifo.readline().strip() for _ in range(batch_size)]

                for line in lines:
                    if line == 'EOF':  # Handle EOF
                        print("End of communication", file=sys.stderr)
                        self.fifo.close()
                        return signal  # Return what we have so far
                    if line:  # Skip empty lines
                        try:
                            signal.append(float(line))
                        except ValueError:
                            print(f"Invalid data format: {line}", file=sys.stderr)
                '''
                # read batch of bytes - one value is 4 bytes
                data = self.fifo.read(batch_size)
                if not data:
                    print("End of communication", file=sys.stderr)
                    self.fifo.close()
                    return signal
                # Convert bytes to float
                signal.extend(struct.unpack(f'{len(data) // 4}f', data))

                # Stop reading if we've already collected enough samples
                if len(signal) >= tf.sig_samples:
                    break

            return signal[:tf.sig_samples]  # Return exactly `tf.sig_samples` values

        except BrokenPipeError:
            print("BrokenPipeError: The transmitter has closed the pipe.", file=sys.stderr)
            self.fifo.close()
            return signal

    def close(self):
        self.fifo.close()