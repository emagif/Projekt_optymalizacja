import numpy as np
import matplotlib.pyplot as plt
from draw_func import *
from algorithms import *


if __name__ == "__main__":
    
# 0.9 0.8

    val, x1, x2, iter_num = Quasi_Newton_BFGS(3, -2, [rosenbrock_penalty_f, grad_rosenbrock_penalty, rosenbrock_penalty_draw_contour, rosenbrock_penalty_draw_3D] , [-6, 6], [-6, 6])
    print(f"val: {val}, x1: {x1}, x2: {x2}, iter_num: {iter_num}")
