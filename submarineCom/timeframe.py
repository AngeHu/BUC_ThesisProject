T_frame = 1 # periodo totale del segnale - 1 sec
f_min = 0 # frequenza minima
f_max = 38000 # frequenza massima
f_sampling = 96000 # frequenza campionamento
t_slot = T_frame/4 # periodo segnale
t_sample = t_slot # tempo di campionamento



class Period:
    def __init__(self, start, end, data):
        self.start = start
        self.end = end
        self.data = data


class TimeFrame:
    def __init__(self):
        # intervallo di 2.5 sec

        self.lapse1 = Period(0, f_sampling, (0, 0))
        self.lapse2 = Period(f_sampling, 2 * f_sampling, (0, 1))
        self.lapse3 = Period(2 * f_sampling, 3 * f_sampling, (1, 1))
        self.lapse4 = Period(3 * f_sampling, 4*f_sampling, (1, 0))
        self.timeInterval = [self.lapse1, self.lapse2, self.lapse3, self.lapse4]

        self.slot = [0, 1, 2, 3]
