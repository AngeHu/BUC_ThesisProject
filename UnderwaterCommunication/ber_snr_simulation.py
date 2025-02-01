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
    snr_db = np.arange(-25, 16, 1)  # SNR range from -25 to 5 dB

    ber_mean_peak = []
    ber_max_peak = []
    ber_slot_peak = []
    ber_peak_density = []

    # matrix of bit error rate
    for i in range(len(snr_db)):
        print("SNR: ", snr_db[i])
        print("start transmission")

        # start transmission
        transmitter = multiprocessing.Process(target=run_transmitter, args=('./transmitter.py', snr_db[i]))
        receiver = multiprocessing.Process(target=run_receiver, args=('./receiver.py',))
        transmitter.start()
        receiver.start()
        transmitter.join()
        receiver.join()

        # compare the two outputs
        mean_peak_error = 0
        max_peak_error = 0
        slot_peak_error = 0
        peak_density_error = 0

        # load original data
        original_data = np.load(params.res_directory + 'data.npy')
        # extract numbers from results files
        mean_peak_decoded = np.load(params.res_directory + 'mean_peak.npy')
        max_peak_decoded = np.load(params.res_directory + 'max_peak.npy')
        slot_peak_decoded = np.load(params.res_directory + 'slot_peak.npy')
        peak_density_decoded = np.load(params.res_directory + 'peak_density.npy')

        print("Original data length: ", len(original_data))
        print("Mean peak decoded length: ", len(mean_peak_decoded))
        print("Max peak decoded length: ", len(max_peak_decoded))
        print("Slot peak decoded length: ", len(slot_peak_decoded))
        print("Peak density decoded length: ", len(peak_density_decoded))

        # calculate the number of errors
        original_lenght = len(original_data)
        mean_peak_length = len(mean_peak_decoded)
        max_peak_length = len(max_peak_decoded)
        slot_peak_length = len(slot_peak_decoded)
        peak_density_length = len(peak_density_decoded)

        for n in range(original_lenght):
            if n >= mean_peak_length:
                mean_peak_error += 1
            elif original_data[n] != mean_peak_decoded[n]:
                mean_peak_error += 1
            if n >= max_peak_length:
                max_peak_error += 1
            elif original_data[n] != max_peak_decoded[n]:
                max_peak_error += 1
            if n >= slot_peak_length:
                slot_peak_error += 1
            elif original_data[n] != slot_peak_decoded[n]:
                slot_peak_error += 1
            if n >= peak_density_length:
                peak_density_error += 1
            elif original_data[n] != peak_density_decoded[n]:
                peak_density_error += 1

        if len(original_data) != len(mean_peak_decoded):
            mean_peak_error += abs(len(original_data) - len(mean_peak_decoded))
        if len(original_data) != len(max_peak_decoded):
            max_peak_error += abs(len(original_data) - len(max_peak_decoded))
        if len(original_data) != len(slot_peak_decoded):
            slot_peak_error += abs(len(original_data) - len(slot_peak_decoded))
        if len(original_data) != len(peak_density_decoded):
            peak_density_error += abs(len(original_data) - len(peak_density_decoded))

        # append the ber to the list, if ber > 0.5, set it to 0.5
        ber_mean_peak.append( mean_peak_error/ params.num_bits if mean_peak_error / params.num_bits < 0.5 else 0.5)
        ber_max_peak.append( max_peak_error/ params.num_bits if max_peak_error / params.num_bits < 0.5 else 0.5)
        ber_slot_peak.append( slot_peak_error/ params.num_bits if slot_peak_error / params.num_bits < 0.5 else 0.5)
        ber_peak_density.append( peak_density_error/ params.num_bits if peak_density_error / params.num_bits < 0.5 else 0.5)

    # plot ber/snr graph
    plt.figure()
    plt.plot(snr_db, ber_max_peak, label="Max Peak", color='red')
    plt.plot(snr_db, ber_mean_peak, label="Total Mean Peak", color='blue')
    plt.plot(snr_db, ber_slot_peak, label="Slot Mean Peak", color='green')
    plt.plot(snr_db, ber_peak_density, label="Peak Density", color='orange')
    plt.legend()
    plt.yscale('log')
    plt.ylim(None, 0.5)
    plt.xlabel("SNR [dB]")
    plt.ylabel("BER")
    title = "BER vs SNR"+" - "+str(params.num_bits)+" bits"
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    #plt.show()
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
