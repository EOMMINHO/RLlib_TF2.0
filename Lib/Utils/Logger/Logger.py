import json
import os

class EpochLogger(object):
    def __init__(self, logdir, filename, indent=4, verbose=1, width=40):
        self.stat = dict()
        self.logdir = logdir
        self.filename = filename
        self.path = os.path.join(self.logdir, self.filename)
        self.indent = indent
        self.epoch = 0

        self.verbosity = verbose
        self.width = width

    def append_dict(self, dict):
        self.stat["epoch"+str(self.epoch)] = dict
        if self.verbosity == 1:
            self._readable_print(dict)
        self.epoch += 1

    def log_json(self):
        assert self.stat is not None
        with open(self.path, 'w') as json_file:
            json.dump(self.stat, json_file, indent=self.indent)

    def open_json(self):
        with open(self.path, 'r') as json_file:
            stats = json.load(json_file)
        return stats

    def _readable_print(self, dict):
        """
        stdout the dict in a readable way
        """
        cstr = "Epoch" + str(self.epoch)
        print(cstr.center(self.width, '-'))
        for (key, val) in dict.items():
            print("|{:>20s}|{:18.3f}|".format(key, val))
        print("-"*self.width)




if __name__ == "__main__":
    epochlogger = EpochLogger("./", "test.txt")
    dict1 = dict()
    dict1["hello"] = 30
    dict1["AI"] = 20
    dict2 = dict()
    dict2["deep"] = 17.2
    dict2["reinforcement"] = 0.03
    epochlogger.append_dict(dict1)
    epochlogger.append_dict(dict2)
    #epochlogger.log_json()