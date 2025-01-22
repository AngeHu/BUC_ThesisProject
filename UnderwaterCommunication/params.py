#TODO: aggiustare frequenza minima e massima a 18k e 38k

# debug e ber/snr simulation non possono essere contemporaneamente attivi
DEBUG = True
BER_SNR_SIMULATION = False
SAVE_IMG = False

# chirp decoding - only one of the two can be active
MAX_PEAK = False # if false, use MEAN_PEAK
MEAN_PEAK = False
SLOT_PEAK = True

# ber/snr simulation parameters
num_bits = 12 # Number of bits to transmit
img_directory = "./img/slot_peak/"  # directory

SNR = 1  # rapporto segnale rumore

# Doppler effect
v_transmitter = 0 # positiva se si avvicina, negativa se si allontana
v_receiver = 0 # positiva se si allontana, negativa se si avvicina
F_SAMPLING = 96000 # frequenza campionamento orginale
T_FRAME = 0.1  # durata frame originale
c = 1500 # velocità suono m/s

v_relative = v_receiver - v_transmitter # velocità relativa m/s
scaling_factor = (c - v_receiver) / (c - v_transmitter)
f_min = 18000 # frequenza minima
f_max = 34000 # frequenza massima
f_min_scaled = f_min * scaling_factor
f_max_scaled = f_max * scaling_factor
wavelength = c / ((f_max+f_min)/2) # lunghezza d'onda
dopp_freq = v_relative/ wavelength # frequenza doppler
f_sampling_doppler = F_SAMPLING + dopp_freq # frequenza campionamento- 96 kHz + frequenza doppler (0 se velocità trasmettitore = velocità ricevitore)

# receiver
chirp_samples = int(F_SAMPLING * T_FRAME/4)  # numero di campioni per chirp
sig_samples =  chirp_samples * 4
t_slot = T_FRAME/4  # periodo slot del segnale
# transmitter
chirp_samples_doppler = int(f_sampling_doppler * T_FRAME/4)  # numero di campioni per chirp
sig_samples_doppler =  int(F_SAMPLING * T_FRAME/4)*3 + chirp_samples_doppler
t_slot_doppler = (T_FRAME / 4) / scaling_factor # periodo slot con effetto doppler
T_frame_doppler = 3 * t_slot + t_slot_doppler


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