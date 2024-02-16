import os


class Channel:
    def __delete__(self):
        if os.path.exists((self.channel)):
            os.remove(self)
            print("FIFO named '%s' is deleted successfully" % self.channel)


    def __init__(self):
        # start communication - create pipe channel
        # self.channel = os.path.join(os.getcwd(), "/mypipe")
        self.channel = "./pipe"
        self.__delete__()
        # permission
        self.mode = 0o600
        os.mkfifo(self.channel, self.mode)
        print("FIFO named '%s' is created successfully" % self.channel)

    def open(self, mode):
        if mode = 'r':
            self.fifo = open(self.channel, "r")
        elif mode = 'w':
            self.fifo = open(self.channel, "w")
        else:
            print("Mode not valid")

    def send_data(self, data):
        self.fifo.write(data)
        self.fifo.flush() # assicura che i dati vengano scritti immediatamente

    def read_data(self):
        data = self.fifo.read()
        return data
