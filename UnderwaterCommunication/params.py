from dotenv import load_dotenv
import os

load_dotenv()
USERNAME = os.getenv("DB_USERNAME")
PASSWORD = os.getenv("DB_PASSWORD")

# debug e ber/snr simulation non possono essere contemporaneamente attivi
DEBUG = False
BER_SNR_SIMULATION = False
if DEBUG: # se debug è attivo, non posso fare la simulazione di ber/snr
    BER_SNR_SIMULATION = False
if BER_SNR_SIMULATION: # se la simulazione di ber/snr è attiva, non posso fare il debug
    DEBUG = False
BIO_SIGNALS = True
SAVE_IMG = False # salva le immagini
PLOT = False # mostra i grafici
ANIMATION = False # mostra l'animazione

# chirp decoding - only one of the two can be active
### Not implemented yet
MAX_PEAK = False # if false, use MEAN_PEAK
MEAN_PEAK = False
SLOT_PEAK = True

# ber/snr simulation parameters
num_bits = 32 # Number of bits to transmit
img_directory = "./img/slot_peak/"  # directory

# MongoDB
seed = 42
uri = f"mongodb+srv://{USERNAME}:{PASSWORD}@dolphincleanaudio.q2bsd.mongodb.net/?retryWrites=true&w=majority&appName=DolphinCleanAudio"


SNR = -20  # rapporto segnale rumore

# Doppler effect
v_transmitter = 0 # positive if moving closer, negative if moving away
v_receiver = 0 # positive if moving away, negative if moving closer
F_SAMPLING = 96000 # original sampling frequency
if BIO_SIGNALS:
    T_FRAME = 1.6  # each slot is 0.4 sec
    v_transmitter = 0  # positive if moving closer, negative if moving away
    v_receiver = 0  # positive if moving away, negative if moving closer
else:
    T_FRAME = 0.1  # arbiitrary choice
c = 1500  # speed of sound m/s in water
T_SAMPLING = 1/F_SAMPLING  # sampling period

v_relative = v_receiver - v_transmitter  # relative velocity

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