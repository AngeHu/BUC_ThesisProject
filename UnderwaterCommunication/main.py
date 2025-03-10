import threading
import subprocess
import multiprocessing
import time
# Press the green button in the gutter to run the script.
def run_script(script_name):
    subprocess.run(["python3", script_name])

if __name__ == '__main__':

    print("Simulation START: ")


    if __name__ == "__main__":
        transmitter = multiprocessing.Process(target=run_script, args=("./transmitter.py",))
        receiver = multiprocessing.Process(target=run_script, args=("./receiver.py",))
        # animation = multiprocessing.Process(target=run_script, args=("./animation.py",))
        # receiver_animation = multiprocessing.Process(target=run_script, args=("./receiver_animation.py",))

        #transmitter = threading.Thread(target=run_script, args=("./transmitter.py",))
        #receiver = threading.Thread(target=run_script, args=("./receiver.py",))

        transmitter.start()
        receiver.start()
        time.sleep(5)
        #animation.start()
        #receiver_animation.start()

        transmitter.join()
        receiver.join()
        #animation.join()
        #receiver_animation.join()

        print("Both scripts have finished executing.")

