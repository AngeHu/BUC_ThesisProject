import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import params
import traceback
import params  # Ensure this module defines T_frame
matplotlib.use('TkAgg')

BATCH_SIZE = 800  # Number of new rows to read from the CSV
FRAME_SIZE = 4*params.sig_samples  # Maximum number of data points to display
INTERVAL = 1 # Milliseconds between updates
# Global data arrays
time_data, signal_data, frequency_data = [], [], []
last_position = 0  # Track the last read position in the CSV
transmitter_file = "./animation/transmitter.csv"

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


# Set up the figure and axes for animation
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

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

# Animation function
def update(frame):
    global last_position
    # Fetch new data
    new_position = read_new_data(last_position)
    if new_position == last_position:
        # No new data, skip update
        return line1, line2
    last_position = new_position

    if time_data:
        # Update the line data
        line1.set_data(time_data, signal_data)
        line2.set_data(time_data, frequency_data)

        # Adjust the x-axis to show the latest data window
        start_time = time_data[0]
        end_time = time_data[-1]
        ax1.set_xlim(start_time, end_time)
        ax2.set_xlim(start_time, end_time)

    return line1, line2


if __name__ == "__main__":
    # Create the animation
    ani = FuncAnimation(fig, update, interval=INTERVAL)  # Update every 500 ms
    plt.tight_layout()
    plt.show()
