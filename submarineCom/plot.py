#TODO: Add the ability to plot the signal and the frequency in real time
#TODO: Add labels to the plots
#TODO: Add titles to the plots

import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import timeframe as tf
from matplotlib.animation import FuncAnimation

BATCH_SIZE = 1000


class Animation:
    # initial data
    def __init__(self):
        self.x = [0]
        self.y_freq = [0]
        self.y_sig = [0]

        # creating the first plot and frame
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1)
        self.freq, = self.ax1.plot(self.x, self.y_freq, color='g')
        self.sig, = self.ax2.plot(self.x, self.y_sig, color='r')

        self.ax1.set_ylim(0,50000)
        self.ax2.set_ylim(-1, 1)

        self.ax1.set_xlim(0, 4*tf.T_frame)
        self.ax1.set_xlim(0, 4*tf.T_frame)
        self.ax2.sharex(self.ax1)
        self.index = 0

        # real time update data structures
        self.new_data_available = False
        self.signal_frequency = []
        self.generated_signal = []
        self.done = True

    def add_data(self):
        print("Adding data")

        self.time = [self.x[-1]]
        for i in range(BATCH_SIZE-1):
            self.time.append(self.time[-1] + tf.T_frame / tf.f_sampling)
        frq_new = self.signal_frequency
        sig_new = self.generated_signal

        print("Time: ", self.time)
        print("Frequency: ", self.freq_new)
        return frq_new, sig_new
        

    # updates the data and freq
    def update(self, frame):
        #self.done = False
        if self.new_data_available:
            print("New data available")
            frq_new, sig_new = self.add_data()
            self.new_data_available = False

            # Update the rolling buffer of x and y data to maintain a fixed window
            if(self.x[-1] < 4*tf.T_frame):
                self.x = np.append(self.x, self.time)
                self.y_freq = np.append(self.y_freq, frq_new)
                self.y_sig = np.append(self.y_sig, sig_new)
            else:
                self.x = np.append(self.x[BATCH_SIZE:], self.time)  # Remove the oldest x value and append the new one
                self.y_freq = np.append(self.y_freq[BATCH_SIZE:], frq_new)  # Remove the oldest y value and append the new one
                self.y_sig = np.append(self.y_sig[BATCH_SIZE:], sig_new)  # Remove the oldest y value and append the new one
                # Adjust the x-axis to always show the latest window
                self.ax1.set_xlim(self.x[0], self.x[-1])

            # Update the plot with new data
            self.freq.set_xdata(self.x)
            self.freq.set_ydata(self.y_freq)
            self.sig.set_xdata(self.x)
            self.sig.set_ydata(self.y_sig)
        #self.done = True

    def animate(self, shared_data):
        self.anim = FuncAnimation(self.fig, self.update, frames=None, fargs=(shared_data,),save_count=100, interval=1)
        plt.show()

if __name__ == "__main__":
    a = Animation()
    a.animate()
