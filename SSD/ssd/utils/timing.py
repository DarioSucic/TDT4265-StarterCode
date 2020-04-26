from time import perf_counter_ns as pc

class Time:
    def __init__(self, label, level=0, div="ms"):
        self.label = label
        self.div = div
        self.level = level

    def __enter__(self):
        self.st = pc()
    
    def __exit__(self, *args):
        dt = pc() - self.st
        if   self.div == "µs": x = dt / 1e3
        elif self.div == "ms": x = dt / 1e6
        elif self.div == "s" : x = dt / 1e9
        print(f"{'  ' * self.level}{self.label} :: {x:.2f} {self.div}")
        if self.level == 0:
            print()