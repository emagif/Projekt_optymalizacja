import numpy as np
import matplotlib.pyplot as plt
from draw_func import *
from algorithms import *


if __name__ == "__main__":
    
# 0.9 0.8
    val, x1, x2, iter_num = Quasi_Newton_DFP(-2, -4, himmelblau_f, grad_himmelblau, himmelblau_f_draw_contour, himmelblau_f_draw_3D_surf, [-6, 6], [-6, 6])
    print(f"val: {val}, x1: {x1}, x2: {x2}, iter_num: {iter_num}")



    
    
    
    

