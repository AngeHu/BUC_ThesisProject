import matplotlib; matplotlib.use("TkAgg")
import random
import matplotlib.pyplot as plt
import numpy as np
import timeframe as tf
from matplotlib.animation import FuncAnimation

BATCH_SIZE = 1000

global new_data_available

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

    def add_data(self):



    # updates the data and freq
    def update(self, frame):
        global new_data_available
        if new_data_available:
            self.add_data()
            new_data_available = False
            x_new = [self.x[-1]]
            frq_new = [self.y_freq[-1]]
            sig_new = [self.y_sig[-1]]
            for i in range(BATCH_SIZE-1):
                x_new.append(x_new[-1] + tf.T_frame / tf.f_sampling)
                frq_new.append(np.random.randint(1, 50001))
                sig_new.append(np.sin(x_new[i+1]))  # New random value for y

            # Update the rolling buffer of x and y data to maintain a fixed window
            if(self.x[-1] < 4*tf.T_frame):
                self.x = np.append(self.x, x_new)
                self.y_freq = np.append(self.y_freq, frq_new)
                self.y_sig = np.append(self.y_sig, sig_new)
            else:
                self.x = np.append(self.x[BATCH_SIZE:], x_new)  # Remove the oldest x value and append the new one
                self.y_freq = np.append(self.y_freq[BATCH_SIZE:], frq_new)  # Remove the oldest y value and append the new one
                self.y_sig = np.append(self.y_sig[BATCH_SIZE:], sig_new)  # Remove the oldest y value and append the new one
                # Adjust the x-axis to always show the latest window
                self.ax1.set_xlim(self.x[0], self.x[-1])

            # Update the plot with new data
            self.freq.set_xdata(self.x)
            self.freq.set_ydata(self.y_freq)
            self.sig.set_xdata(self.x)
            self.sig.set_ydata(self.y_sig)


        '''
        # Update the new value in-place
        for i in range(100):
            self.x[self.index + i] = self.x[self.index + i - 1] + tf.T_frame / tf.f_sampling  # Increment time
        # Update the index for circular buffer
        self.index = (self.index + 100) % len(self.x)
        self.y[self.index-100:self.index] = np.random.randint(1, 50001, size=100)  # Random value for y

        # Shift the plot window to show the latest data
        self.freq.set_xdata(self.x)
        self.freq.set_ydata(self.y)

        if(self.x[self.index] > 4*tf.T_frame):
            self.ax1.set_xlim(self.x[self.index - len(self.x)], self.x[self.index])
        else:
            print(self.x[self.index])
        '''

    def animate(self):
        self.anim = FuncAnimation(self.fig, self.update, frames=None, save_count=100, interval=1)
        plt.show()

if __name__ == "__main__":
    a = Animation()
    a.animate()
