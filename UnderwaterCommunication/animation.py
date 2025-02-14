import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import params
import os
import traceback
matplotlib.use('TkAgg')

NUM_DISPLAYED_FRAMES = 2 # Number of frames to display in animation window
BATCH_SIZE = 800 # Number of new rows to read from the CSV
FRAME_SIZE = NUM_DISPLAYED_FRAMES*params.sig_samples  # Maximum number of data points to display
FRAME_SIZE_DOPPLER = NUM_DISPLAYED_FRAMES*params.sig_samples_doppler  # Maximum number of data points to display
INTERVAL = 1 # Milliseconds between updates
# Global data arrays
time_data, signal_data, frequency_data = [], [], []
transmitter_last_position = 0  # Track the last read position in the CSV

doppler_time, doppler_signal, doppler_frequency = [], [], []
doppler_last_position = 0  # Track the last read position in the CSV

os.makedirs("animation", exist_ok=True)
transmitter_file = "./animation/transmitter.csv"
doppler_file = "./animation/transmitter_doppler.csv"

# Function to read new rows from the CSV
def read_new_data(last_position, batch_size=BATCH_SIZE):
    global time_data, signal_data, frequency_data
    new_lines = []
    with open(transmitter_file, 'r') as file:
        if last_position == 0:
            header = file.readline()  # Skip the header on the first read
            print(f"Header skipped: {header.strip()}")
            last_position = file.tell()
        file.seek(last_position)
        for _ in range(batch_size):
            line = file.readline()
            if not line:
                break
            new_lines.append(line)
        last_position = file.tell()  # Update position after reading

    if new_lines:
        new_data = [line.strip().split(",") for line in new_lines]
        try:
            time_data.extend([float(row[0]) for row in new_data])
            signal_data.extend([float(row[1]) for row in new_data])
            frequency_data.extend([float(row[2]) for row in new_data])
        except (ValueError, IndexError) as e:
            print("Malformed data detected and skipped.")
            print("Detailed Error Information:")
            # Print the exception details
            print(f"Exception Type: {type(e).__name__}")
            print(f"Exception Message: {e}")
            # Print the entire stack trace
            print("Stack Trace:")
            traceback.print_exc()
            # Print the problematic data
            print("Problematic Data:")
            print(new_data)

        # Limit data length for smoother animation
        if len(time_data) > FRAME_SIZE:
            time_data[:] = time_data[-FRAME_SIZE:]
            signal_data[:] = signal_data[-FRAME_SIZE:]
            frequency_data[:] = frequency_data[-FRAME_SIZE:]

    return last_position

def read_doppler_data(last_position, batch_size=BATCH_SIZE):
    global doppler_time, doppler_signal, doppler_frequency
    new_lines = []
    with open(doppler_file, 'r') as file:
        if last_position == 0:
            header = file.readline()  # Skip the header on the first read
            print(f"Header skipped: {header.strip()}")
            last_position = file.tell()
        file.seek(last_position)
        for _ in range(batch_size):
            line = file.readline()
            if not line:
                break
            new_lines.append(line)
        last_position = file.tell()  # Update position after reading

    if new_lines:
        new_data = [line.strip().split(",") for line in new_lines]
        try:
            doppler_time.extend([float(row[0]) for row in new_data])
            doppler_signal.extend([float(row[1]) for row in new_data])
            doppler_frequency.extend([float(row[2]) for row in new_data])
        except (ValueError, IndexError) as e:
            print("Malformed data detected and skipped.")
            print("Detailed Error Information:")
            # Print the exception details
            print(f"Exception Type: {type(e).__name__}")
            print(f"Exception Message: {e}")
            # Print the entire stack trace
            print("Stack Trace:")
            traceback.print_exc()
            # Print the problematic data
            print("Problematic Data:")
            print(new_data)

        # Limit data length for smoother animation
        if len(doppler_time) > FRAME_SIZE:
            doppler_time[:] = doppler_time[-FRAME_SIZE:]
            doppler_signal[:] = doppler_signal[-FRAME_SIZE:]
            doppler_frequency[:] = doppler_frequency[-FRAME_SIZE:]

    return last_position


# Set up the figure and axes for animation
if params.BIO_SIGNALS:
    fig, (ax1, ax2) = plt.subplots(2, 1, num="Underwater Acoustic Communication Simulation", figsize=(8, 6))
else:
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, num="Underwater Acoustic Communication Simulation", figsize=(8, 6))
fig.suptitle("Transmitter", fontsize=16)
fig.tight_layout()
# Signal plot
ax1.set_title("Signal")
ax1.grid(True)
ax1.set_xlim(0, 10)  # Initial x-axis limit
ax1.set_ylim(-1, 1)
line1, = ax1.plot([], [], lw=2, label="Signal", color='blue')
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Amplitude")
ax1.legend()

# Frequency plot
ax2.set_title("Frequency")
ax2.grid(True)
ax2.set_xlim(0, 10)  # Match initial x-axis limit of ax1
ax2.set_ylim(0, 50000)
line2, = ax2.plot([], [], lw=2, label="Frequency", color='orange')
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Frequency (Hz)")
ax2.legend()

if not params.BIO_SIGNALS:
    # Doppler plot
    title = str(params.v_relative) + " m/s relative velocity"
    ax3.set_title("Signal with Doppler Effect" + title)
    ax3.grid(True)
    ax3.set_xlim(0, 10)  # Match initial x-axis limit of ax1
    ax3.set_ylim(-1, 1)
    line3, = ax3.plot([], [], lw=2, label="Amplitude", color='red')
    ax3.set_ylabel("Amplitude")
    ax3.set_xlabel("Time (s)")

    # Doppler Frequency plot
    ax4.set_title("Frequency with Doppler Effect" + title)
    ax4.grid(True)
    ax4.set_xlim(0, 10)  # Match initial x-axis limit of ax1
    ax4.set_ylim(0, 50000)
    line4, = ax4.plot([], [], lw=2, label="Amplitude", color='green')
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Amplitude")


# Animation function
def update(frame):
    global transmitter_last_position, doppler_last_position
    # Fetch new data
    transmitter_position = read_new_data(transmitter_last_position)
    if not params.BIO_SIGNALS:
        doppler_position = read_doppler_data(doppler_last_position)

    if params.BIO_SIGNALS:
        if transmitter_position == transmitter_last_position:
            # No new data, skip update
            return line1, line2
    else:
        if doppler_position == doppler_last_position and transmitter_position == transmitter_last_position:
            # No new data, skip update
            return line1, line2, line3, line4
        doppler_last_position = doppler_position
    transmitter_last_position = transmitter_position

    if time_data:
        # Update the line data
        line1.set_data(time_data, signal_data)
        line2.set_data(time_data, frequency_data)
        # Adjust the x-axis to show the latest data window
        start_time = time_data[0]
        end_time = time_data[-1]
        if params.BIO_SIGNALS:
            max_signal = max(signal_data, key=abs)
            max_signal = max_signal + 0.1 * max_signal
            if max_signal == 0:
                max_signal = 1
            ax1.set_ylim(-max_signal, max_signal)
        ax1.set_xlim(start_time, end_time)
        ax2.set_xlim(start_time, end_time)

    if not params.BIO_SIGNALS and doppler_time:
        line3.set_data(doppler_time, doppler_signal)
        line4.set_data(doppler_time, doppler_frequency)

        start_time = doppler_time[0]
        end_time = doppler_time[-1]
        ax3.set_xlim(start_time, end_time)
        ax4.set_xlim(start_time, end_time)

    if params.BIO_SIGNALS:
        return line1, line2
    else:
        return line1, line2, line3, line4


if __name__ == "__main__":
    # Create the animation
    ani = FuncAnimation(fig, update, interval=INTERVAL)  # Update every 500 ms
    plt.tight_layout()
    plt.show()
