import subprocess
import params
import numpy as np

if params.BER_SNR_SIMULATION: # to run the simulation set BER_SNR_SIMULATION = True in params.py
    ber = []
    i = 0
    snr_db = np.arange(-100, 20, 10) # SNR range from -100 to 20 dB

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
        error_count = 0
        for n in range(len(output1)):
            if output1[n] != output2[n]:
                error_count += 1

        # error_count = error_count + (len(output2)-params.num_bits) # bits that hasn't been received

        ber.append(error_count/params.num_bits)

    print("BER: ", ber)
