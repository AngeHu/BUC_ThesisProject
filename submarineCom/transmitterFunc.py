# transmitter

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import channel

class TimeLapse:
    def __init__(self, begin, end, data):
        self.begin = begin
        self.end = end
        self.data = data


# intervallo di 2.5 sec
lapse1 = TimeLapse(0, 2.5, bin(0))
lapse2 = TimeLapse(2.5, 5, bin(1))
lapse3 = TimeLapse(5, 7.5, bin(3))
lapse4 = TimeLapse(7.5, 10, bin(2))

timeInterval = [lapse1, lapse2, lapse3, lapse4]

# must visualize 2 graphs: one for frequency and one for signal
def plot_function(x, y, title='Grafico', x_label='X', y_label='Y', time_slice = True):
    plt.figure(figsize=(10, 8))  # Imposta le dimensioni del grafico (opzionale)
    plt.plot(x, y, label='Dati')  # Crea il grafico

    plt.title(title)  # Aggiunge un titolo al grafico
    plt.xlabel(x_label)  # Aggiunge un'etichetta all'asse x
    plt.ylabel(y_label)  # Aggiunge un'etichetta all'asse y
    if time_slice:
        plt.xlim(0, 10)
        plt.ylim(0,100)
    plt.legend()  # Aggiunge una legenda al grafico (opzionale)

    plt.grid(True)  # Aggiunge una griglia al grafico (opzionale)

    plt.show()  # Mostra il grafico


def frequency(interval):
    x = np.linspace(interval.begin, interval.end, 1000)
    y = (90/2.5) * x
    plot_function(x, y, "Frequenza", "t", "Hz")
    print(y)
    return y

def signal(interval):
    x = np.linspace(interval.begin, interval.end, 1000)
    y = np.sin(frequency(interval) * x)
    print(y)
    plot_function(x, y, "Segnale", "t", "E", False)
    return y


if __name__ == "__main__":
    channel = channel.Channel()
    signal = signal(timeInterval[0])
    channel.open('w')
    #mandare un segnale (o meglio: pezzi di segnale)
    for i in range(1000):
        if i % 10 == 0:
            channel.send_data(signal[i])

