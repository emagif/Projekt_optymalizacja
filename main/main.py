import numpy as np
import matplotlib.pyplot as plt
from draw_func import *
from algorithms import *


if __name__ == "__main__":
    

    val, x1, x2, iter_num = Quasi_Newton_BFGS(0.2, -0.4, 1)
    print(f"val: {val}, x1: {x1}, x2: {x2}, iter_num: {iter_num}")
    
    
    
    
