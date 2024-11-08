'''
#TODO: Add the ability to plot the signal and the frequency in real time
#TODO: Add labels to the plots
#TODO: Add titles to the plots
import random
import matplotlib
matplotlib.use("TkAgg")
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

    def add_data(self):
        self.time = [self.x[-1]]
        for i in range(BATCH_SIZE-1):
            self.time.append(self.time[-1] + tf.T_frame / tf.f_sampling)
        sig_new = np.sin(np.asarray(self.time) * 2 * np.pi)
        frq_new = np.random.randint(0, 50000, BATCH_SIZE)
        return frq_new, sig_new
        

    # updates the data and freq
    def update(self, frame):
        frq_new, sig_new = self.add_data()
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


    def animate(self):
        self.anim = FuncAnimation(self.fig, self.update, frames=None,save_count=100, interval=1)
        plt.pause(0.01)


if __name__ == "__main__":
    a = Animation()
    a.animate()
'''


import numpy as np
import params as tf
from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource
import multiprocessing as mp

# compute new values in side process


BATCH_SIZE = 1000
UNDERSAMPLING_FACTOR = 100

sampling_rate = tf.f_sampling/UNDERSAMPLING_FACTOR
sampling_time = tf.t_slot/sampling_rate # Time between samples

# Set up data
x = np.array([0])
y = np.sin(x)

# Create a ColumnDataSource (used for updating the plot)
source = ColumnDataSource(data={'x': x, 'y': y})

initial_x_range = (0, 4*tf.T_frame)  # Display the x-axis from 0 to 5 initially
initial_y_range = (-1, 1)  # Display the y-axis from -1 to 1 initially

# Create a figure and line plot
plot = figure(title="Sine Wave Animation", x_axis_label='x', y_axis_label='y', y_range=initial_y_range, x_range=initial_x_range)
plot.line('x', 'y', source=source)

# Define the callback function that will update the plot periodically
def update():
    global x, y
    for i in range(BATCH_SIZE):
        x = np.append(x, x[-1] + sampling_time)
        y = np.append(y, np.sin(x[-1]))
    if x[-1] > 4*tf.T_frame:
        x = x[BATCH_SIZE:]
        y = y[BATCH_SIZE:]
        #update the x-axis to always show the latest window
        plot.x_range.start = x[0]
        plot.x_range.end = x[-1]


    # Update the ColumnDataSource with
    source.data = {'x': x, 'y': y}

# Add a periodic callback that runs every 1 millisecond (0.001 seconds)
curdoc().add_periodic_callback(update, 1)

# Add the plot to the current document
curdoc().add_root(plot)

# Run the application with 'bokeh serve' command
# bokeh serve --show animation.py
