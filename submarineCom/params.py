#TODO: aggiustare frequenza minima e massima a 18k e 38k
DEBUG = True

T_frame = 1 # periodo totale del segnale - 1 sec
f_min = 18000 # frequenza minima
f_max = 38000 # frequenza massima
f_sampling = 96000 # frequenza campionamento
t_slot = T_frame/4 # periodo segnale
t_sample = t_slot # tempo di campionamento

SNR = 1 # rapporto segnale rumore

class Period:
    def __init__(self, start, end, data):
        self.start = start
        self.end = end
        self.data = data


class TimeFrame:
    def __init__(self):
        # intervallo di 2.5 sec

        self.lapse1 = Period(0, 1, (0, 0))
        self.lapse2 = Period(1, 2 , (0, 1))
        self.lapse3 = Period(2 , 3 , (1, 1))
        self.lapse4 = Period(3 , 4 , (1, 0))
        self.timeInterval = [self.lapse1, self.lapse2, self.lapse3, self.lapse4]

        self.slot = [0, 1, 2, 3]
