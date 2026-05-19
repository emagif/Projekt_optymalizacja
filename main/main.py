import numpy as np
import matplotlib.pyplot as plt
from draw_func import *
from algorithms import *


if __name__ == "__main__":
    
# 0.9 0.8

    val_2, x5, x6, iter_num_2 = Quasi_Newton_BFGS(-4, -4, [himmelblau_penalty_f, grad_himmelblau_penalty, himmelblau_penalty_draw_contour, himmelblau_penalty_draw_3D], [-6, 6], [-6, 6])
    print(f"val: {val_2}, x1: {x5}, x2: {x6}, iter_num: {iter_num_2}")