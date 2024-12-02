import threading
import subprocess
import multiprocessing
# Press the green button in the gutter to run the script.
def run_script(script_name):
    subprocess.run(["python3", script_name])

if __name__ == '__main__':

    print("Simulation START: ")


    if __name__ == "__main__":
        transmitter = multiprocessing.Process(target=run_script, args=("./transmitter.py",))
        receiver = multiprocessing.Process(target=run_script, args=("./receiver.py",))

        #transmitter = threading.Thread(target=run_script, args=("./transmitter.py",))
        # receiver = threading.Thread(target=run_script, args=("./receiver.py",))

        transmitter.start()
        receiver.start()

        transmitter.join()
        receiver.join()

        print("Both scripts have finished executing.")

