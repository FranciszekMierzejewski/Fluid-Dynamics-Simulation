import numpy as np
from grid import Grid
from fields import Fields

class Diffusion():
    def __init__(self, fields: Fields):
        self.fields = fields
        self.grid = fields.grid

    def diffusion_step(self, dt: float):
        pass