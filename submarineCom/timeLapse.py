class Period:
    def __init__(self, begin, end, data):
        self.begin = begin
        self.end = end
        self.data = data


class TimeLapse:
    def __init__(self):
        # intervallo di 2.5 sec
        self.lapse1 = Period(0, 2.5, (0,0))
        self.lapse2 = Period(2.5, 5, (0,1))
        self.lapse3 = Period(5, 7.5, (1,1))
        self.lapse4 = Period(7.5, 10, (1,0))
        self.timeInterval = [self.lapse1, self.lapse2, self.lapse3, self.lapse4]
