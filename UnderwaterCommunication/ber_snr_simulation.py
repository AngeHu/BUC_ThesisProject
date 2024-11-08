import subprocess
import params
import numpy as np
import re # regular expression
import matplotlib.pyplot as plt
import os

if params.BER_SNR_SIMULATION: # to run the simulation set BER_SNR_SIMULATION = True in params.py
    if not os.path.exists(params.img_directory):
        os.makedirs("./img")
        os.makedirs(params.img_directory)
    ber = []
    i = 0
    snr_db = np.arrange(-10, 10, 1)

    for i in range(len(snr_db)):
        print("SNR: ", snr_db[i])
        print("start transmission")
        process1 = subprocess.Popen(['python3', './transmitter.py', str(snr_db[i])], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print("start reception")
        process2 = subprocess.Popen(['python3', './receiver.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        print("wait for transmission to finish")
        # Capture output and errors
        output1, error1 = process1.communicate()
        output2, error2 = process2.communicate()

        # Print output and errors
        print(output1)
        print(error1)
        print(output2)
        print(error2)

        # todo: implement plot of ber/snr graph
        # compare output1 and output2 to check if the transmission was successful

        # extract numbers from the output
        output1_list = re.findall(r'-?\d\.?\d*', output1)
        output2_list = re.findall(r'-?\d\.?\d*', output2)
        output1 = [float(i) for i in output1_list]
        output2 = [float(i) for i in output2_list]
        '''
        print("Output1: ", output1)
        print("Output2: ", output2)
        '''
        # compare the two outputs
        error_count = 0
        for n in range(len(output1)):
            if output1[n] != output2[n]:
                error_count += 1

        # append the ber to the list, if ber > 0.5, set it to 0.5
        ber.append(error_count / params.num_bits if error_count / params.num_bits < 0.5 else 0.5)

    print("BER: ", ber)

    # plot ber/snr graph
    plt.figure()
    plt.plot(snr_db, ber)
    plt.yscale('log')
    plt.ylim(10**-3, 0.5)
    plt.xlabel("SNR [dB]")
    plt.ylabel("BER")
    plt.title("BER vs SNR")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig(params.img_directory+'ber_snr8.png')

