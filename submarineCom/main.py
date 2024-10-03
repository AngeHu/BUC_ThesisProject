import threading
import subprocess

# Press the green button in the gutter to run the script.
def run_script(script_name):
    subprocess.run(["python", script_name])

if __name__ == '__main__':

    print("Simulation START: ")


    if __name__ == "__main__":
        script1_thread = threading.Thread(target=run_script, args=("./transmitter.py",))
        script2_thread = threading.Thread(target=run_script, args=("./receiver.py",))

        script1_thread.start()
        script2_thread.start()

        script1_thread.join()
        script2_thread.join()

        print("Both scripts have finished executing.")

