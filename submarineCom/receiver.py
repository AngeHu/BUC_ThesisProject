# receiver deve leggere da fifo in tempo reale...
# aggiugerò in futuro il ritardo, che va calcolato in base a callibrazione
# receiver per ora riceve in tempo reale e deve salvare i dati in un array
# receiver deve riuscire a ricostruire correttamente ilsegnale..più o meno
# sicruamente ci sarà perdita di segnale!

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import channel
import timeframe

t = np.linspace(0, timeframe.T_frame, timeframe.f_sampling*4) # vettore tempo

def plot_function(x, y, title='Grafico', x_label='X', y_label='Y', time_slice = True):
    plt.figure(figsize=(10, 8))  # Imposta le dimensioni del grafico (opzionale)
    plt.plot(x, y, label='Dati')  # Crea il grafico

    plt.title(title)  # Aggiunge un titolo al grafico
    plt.xlabel(x_label)  # Aggiunge un'etichetta all'asse x
    plt.ylabel(y_label)  # Aggiunge un'etichetta all'asse y
    if time_slice:
        plt.xlim(0, timeframe.T_frame)
        plt.ylim(0, timeframe.f_max)
    plt.legend()  # Aggiunge una legenda al grafico (opzionale)

    plt.grid(True)  # Aggiunge una griglia al grafico (opzionale)

    # animate = FuncAnimation(fig, animate, frames=20, interval=500, repeat=False)
    plt.show()  # Mostra il grafico


class Receiver:
    def __init__(self):
        self.channel = channel.Channel('r')
        print("Receiver ON")
        self.channel.open('r')
        self.dataCount = 0
        self.data = []

    def read(self):
        '''
        print("Reading data")
        self.data[self.dataCount] = self.channel.read_data()
        print("Finished reading data")
        print(self.data[self.dataCount])
        #decipher(data)
        self.dataCount = self.dataCount + 1
        '''
        data = self.channel.read_signal()
        return data

    def plot_data(self, data):
        print("Data length: ", len(data))
        x = np.linspace(0, timeframe.T_frame, len(data))
        plot_function(x, data, "Received", "t", "E", False)

    # def decipher(self):
        #i = 0
        #while i == self.dataCount:
            #self.deciphered[i] =
            #pass

if __name__ == "__main__":
    rc = Receiver()
    '''
    while True:
        data_str = rc.read()
        data = [float(x) for x in data_str]
        rc.plot_data(data)
    '''
        # see correlation



