import random

class RNG:
    def __init__(self, seed: int | None):
        self._seed = seed
        self._rng = random.Random(seed)

    def seed(self) -> int | None:
        return self._seed

    def choice(self, xs):
        return self._rng.choice(xs)

    def uniform(self, a=0.0, b=1.0):
        return self._rng.uniform(a, b)

    def rand(self):
        return self._rng.random()

