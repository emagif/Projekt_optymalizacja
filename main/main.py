import numpy as np
import matplotlib.pyplot as plt
from draw_func import *
from algorithms import *


if __name__ == "__main__":
    
    alpha_k = Quasi_Newton_BFGS(1, 2, 1)
    print(alpha_k)
    
    
