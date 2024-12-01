import params
import subprocess
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# communicate takes too much time
def run_transmitter(script, snr_db):
    subprocess.run(['python3', script, str(snr_db)])

def run_receiver(script):
    subprocess.run(['python3', script])

if params.BER_SNR_SIMULATION:  # to run the simulation set BER_SNR_SIMULATION = True in params.py
    if not os.path.exists("./img"):
        os.makedirs("./img")
    if not os.path.exists(params.img_directory):
        os.makedirs(params.img_directory)
    if not os.path.exists(params.res_directory):
        os.makedirs(params.res_directory)
    i = 0
    snr_db = np.arange(-30, 10, 1)  # SNR range from -25 to 5 dB

    ber_mean_peak = []
    ber_max_peak = []
    ber_slot_peak = []

    # matrix of bit error rate
    for i in range(len(snr_db)):
        print("SNR: ", snr_db[i])
        print("start transmission")
        '''
        process1 = subprocess.Popen(['python3', './transmitter.py', str(snr_db[i])],
                                    #stdout=subprocess.PIPE,
                                    #stderr=subprocess.PIPE,
                                    text=True)
        print("start reception")
        process2 = subprocess.Popen(['python3', './receiver.py'],
                                    #stdout=subprocess.PIPE,
                                    #stderr=subprocess.PIPE,
                                    text=True)
        '''

        # start transmission
        transmitter = multiprocessing.Process(target=run_transmitter, args=('./transmitter.py', snr_db[i]))
        receiver = multiprocessing.Process(target=run_receiver, args=('./receiver.py',))
        transmitter.start()
        receiver.start()
        print("wait for transmission to finish")
        transmitter.join()
        # Capture output and errors
        print("wait for reception to finish")
        receiver.join()

        '''
        # Print output and errors
        print(output1)
        print("transmitter:", error1)
        print(output2)
        print("receiver:", error2)

        # compare output1 and output2 to check if the transmission was successful

        # extract numbers from the output
        output1_list = re.findall(r'-?\d\.?\d*', output1)
        original_data = [float(i) for i in output1_list]

        lines = output2.split("\n")  # Splits by newline
        print(lines)

        for i in range(len(lines)):
            lines[i] = re.findall(r'-?\d\.?\d*', lines[i])
            lines[i] = [float(j) for j in lines[i]]
            print(lines[i], len(lines[i]))
        
        
        # Ensure there are at least three lines of output
        if len(lines) >= 3:
            mean_peak = lines[0]  # First string (Mean peak decoded)
            max_peak = lines[1]  # Second string (Max peak decoded)
            slot_peak = lines[2]  # Third string (Slot peak decoded)
        else:
            print("Error: Receiver output does not have 3 lines.", file=sys.stderr)
            mean_peak = max_peak = slot_peak = None  # Handle missing data
        '''
        # compare the two outputs
        mean_peak_error = 0
        max_peak_error = 0
        slot_peak_error = 0

        # load original data
        original_data = np.load(params.res_directory + 'data.npy')
        # extract numbers from results files
        mean_peak_decoded = np.load(params.res_directory + 'mean_peak.npy')
        max_peak_decoded = np.load(params.res_directory + 'max_peak.npy')
        slot_peak_decoded = np.load(params.res_directory + 'slot_peak.npy')

        for n in range(len(original_data)):
            if original_data[n] != mean_peak_decoded[n]:
                mean_peak_error += 1
            if original_data[n] != max_peak_decoded[n]:
                max_peak_error += 1
            if original_data[n] != slot_peak_decoded[n]:
                slot_peak_error += 1

        # append the ber to the list, if ber > 0.5, set it to 0.5
        ber_mean_peak.append( mean_peak_error/ params.num_bits if mean_peak_error / params.num_bits < 0.5 else 0.5)
        ber_max_peak.append( max_peak_error/ params.num_bits if max_peak_error / params.num_bits < 0.5 else 0.5)
        ber_slot_peak.append( slot_peak_error/ params.num_bits if slot_peak_error / params.num_bits < 0.5 else 0.5)

    # plot ber/snr graph
    plt.figure()
    plt.plot(snr_db, ber_max_peak, label="Max Peak", color='red')
    plt.plot(snr_db, ber_mean_peak, label="Mean Peak", color='blue')
    plt.plot(snr_db, ber_slot_peak, label="Slot Peak", color='green')
    plt.legend()
    plt.yscale('log')
    plt.ylim(None, 0.5)
    plt.xlabel("SNR [dB]")
    plt.ylabel("BER")
    plt.title("BER vs SNR")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    file_name = params.img_directory+'ber_snr'+str(params.num_bits)+'.png'
    if os.path.exists(file_name):
        i = 1
        while os.path.exists(file_name):
            if i == 1:
                file_name = file_name[:-4] + "_" + str(i) + ".png"
            else:
                file_name = file_name[:-5] + str(i) + ".png"
            i += 1
    plt.savefig(file_name)
