import numpy as np
import matplotlib.pyplot as plt
from draw_func import *
from algorithms import *


if __name__ == "__main__":
    

### OPTYMALIZACJA FUNKCJI THREE HUMP CAMEL!!! INNE FUNKCJE W PLIKACH main.py (Himmelblau)  i main1.py (Rosenbrock)

# Z ograniczeniami 
    val_1, x1, x2, iter_num_1, f_values1_o, path1_o = Quasi_Newton_DFP(-2, 0, [three_hump_camel_penalty_f, grad_three_hump_camel_penalty])
    val_2, x3, x4, iter_num_2, f_values2_o, path2_o = Quasi_Newton_DFP(2, 0, [three_hump_camel_penalty_f, grad_three_hump_camel_penalty])
    val_3, x5, x6, iter_num_3, f_values3_o, path3_o = Quasi_Newton_BFGS(-2, 0, [three_hump_camel_penalty_f, grad_three_hump_camel_penalty])
    val_4, x7, x8, iter_num_4, f_values4_o, path4_o = Quasi_Newton_BFGS(2, 0, [three_hump_camel_penalty_f, grad_three_hump_camel_penalty])


# Bez ograniczeń 
    val_5, x9, x10, iter_num_5, f_values1_b, path1_b = Quasi_Newton_DFP(-2, 0, [three_hump_camel_f, grad_three_hump_camel])
    val_6, x11, x12, iter_num_6, f_values2_b, path2_b = Quasi_Newton_DFP(2, 0, [three_hump_camel_f, grad_three_hump_camel])
    val_7, x13, x14, iter_num_7, f_values3_b, path3_b = Quasi_Newton_BFGS(-2, 0, [three_hump_camel_f, grad_three_hump_camel])
    val_8, x15, x16, iter_num_8, f_values4_b, path4_b = Quasi_Newton_BFGS(2, 0, [three_hump_camel_f, grad_three_hump_camel])

# Wykresy  z ograniczeniami
    drawFValues(f_values1_o, f_values2_o, f_values3_o, f_values4_o, 'x=[-2,0] DFP', 'x=[2,0] DFP', 'x=[-2,0] BFGS', 'x=[2,0] BFGS')
    drawFValuesLog(f_values1_o, f_values2_o, f_values3_o, f_values4_o, 'x=[-2,0] DFP', 'x=[2,0] DFP', 'x=[-2,0] BFGS', 'x=[2,0] BFGS')
    three_hump_camel_f_draw_contour_3_z_ograniczeniami([-3, 3], [-3, 2], 400, 1000, path1_o, path2_o, path3_o, path4_o)

# Wykresy bez ograniczeń
    drawFValues(f_values1_b, f_values2_b, f_values3_b, f_values4_b, 'x=[-2,0] DFP', 'x=[2,0] DFP', 'x=[-2,0] BFGS', 'x=[2,0] BFGS')
    drawFValuesLog(f_values1_b, f_values2_b, f_values3_b, f_values4_b, 'x=[-2,0] DFP', 'x=[2,0] DFP', 'x=[-2,0] BFGS', 'x=[2,0] BFGS')
    three_hump_camel_f_draw_contour_3_bez_ograniczen([-3, 3], [-3, 2], 400, 1000, path1_b, path2_b, path3_b, path4_b)