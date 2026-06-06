# Given a randomnly generated velocity field, we compute our divergence at each cell, solve for pressure and project velocity field to be divergence-free, hence allowing for incompressible flow.

import numpy as np
from grid import Grid
from fields import Fields

class PressureProjection():
    def __init__(self, fields: Fields):
        self.fields = fields
        self.grid = fields.grid

    def calc_divergence(self) -> np.ndarray:
        return self.grid.divergence(self.fields.x, self.fields.y)