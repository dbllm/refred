
class QGen:

    def __init__(self) -> None:
        self.counter = 0
        pass

    def generate(self):
        self.counter += 1
        return self.gen(self.counter)

    def gen(self, counter):
        pass

