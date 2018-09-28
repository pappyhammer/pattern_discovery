from datetime import datetime


class Parameters:
    def __init__(self, path_results=None, time_str=None, bin_size=1):
        self.path_results = path_results
        self.time_str = time_str
        self.bin_size = bin_size
        if self.time_str is None:
            self.time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")