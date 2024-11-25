#TODO: aggiustare frequenza minima e massima a 18k e 38k

# debug e ber/snr simulation non possono essere contemporaneamente attivi
DEBUG = False
BER_SNR_SIMULATION = True
SAVE_IMG = False

# chirp decoding - only one of the two can be active
MAX_PEAK = False # if false, use MEAN_PEAK
MEAN_PEAK = False
SLOT_PEAK = True

# ber/snr simulation parameters
num_bits = 10000 # Number of bits to transmit
img_directory = "./img/slot_peak/"  # directory

T_frame = 0.1 # periodo totale del segnale - 1 sec
f_min = 18000 # frequenza minima
f_max = 38000 # frequenza massima
f_sampling = 96000 # frequenza campionamento
chirp_samples = int(f_sampling * T_frame/4) # numero di campioni per chirp
sig_samples = chirp_samples * 4 # numero di campioni
t_slot = T_frame/4 # periodo segnale
t_sample = t_slot # tempo di campionamento


SNR = 10 # rapporto segnale rumore

class Period:
    def __init__(self, start, end, data):
        self.start = start
        self.end = end
        self.data = data


class TimeFrame:
    def __init__(self):
        # intervallo di 2.5 sec

        self.lapse1 = Period(0, 1, [0, 0])
        self.lapse2 = Period(1, 2 , [0, 1])
        self.lapse3 = Period(2 , 3 , [1, 1])
        self.lapse4 = Period(3 , 4 , [1, 0])
        self.timeInterval = [self.lapse1, self.lapse2, self.lapse3, self.lapse4]

        self.slot = [0, 1, 2, 3]
