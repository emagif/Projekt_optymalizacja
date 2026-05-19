import numpy as np
import matplotlib.pyplot as plt
from draw_func import *
from algorithms import *


if __name__ == "__main__":
    
# 0.9 0.8
    # val, x1, x2, iter_num = Quasi_Newton_BFGS(3, -2, [three_hump_camel_penalty_f, grad_three_hump_camel_penalty, three_hump_camel_penalty_draw_contour, three_hump_camel_penalty_draw_3D] , [-6, 6], [-6, 6])
    # print(f"val: {val}, x1: {x1}, x2: {x2}, iter_num: {iter_num}")

    val, x1, x2, iter_num = Quasi_Newton_DFP(-5, 5, [rosenbrock_penalty_f, grad_rosenbrock_penalty, rosenbrock_penalty_draw_contour, rosenbrock_penalty_draw_3D], [-6, 6], [-6, 6])
    print(f"val: {val}, x1: {x1}, x2: {x2}, iter_num: {iter_num}")

    val_1, x3, x4, iter_num_1 = Quasi_Newton_DFP(5, 5, [three_hump_camel_penalty_f, grad_three_hump_camel_penalty, three_hump_camel_penalty_draw_contour, three_hump_camel_penalty_draw_3D], [-6, 6], [-6, 6])
    print(f"val: {val_1}, x1: {x3}, x2: {x4}, iter_num: {iter_num_1}")

    val_2, x5, x6, iter_num_2 = Quasi_Newton_DFP(-5, 0, [himmelblau_penalty_f, grad_himmelblau_penalty, himmelblau_penalty_draw_contour, himmelblau_penalty_draw_3D], [-6, 6], [-6, 6])
    print(f"val: {val_2}, x1: {x5}, x2: {x6}, iter_num: {iter_num_2}")