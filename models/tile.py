# tile.py

class Tile:
    def __init__(self, value, background):
        self.value = value
        self.background = background

    def clear(self):
        self.value = 0
