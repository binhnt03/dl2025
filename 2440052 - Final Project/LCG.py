class LinearCongruentialGenerator:
    def __init__(self, m:int = 2**31-1, a:int = 106542, c:int = 12342986, seed:int = 42) -> None:
        self.m = m
        self.a = a
        self.c = c
        self.seed = seed

    def random(self) -> float:
        self.seed = (self.a * self.seed + self.c) % self.m
        return self.seed / self.m