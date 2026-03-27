import numpy as np
from scipy.optimize import line_search

def rosenbrock_f(xk):
    return (1-xk[0])**2 + 100*(xk[1] - xk[0]**2)**2

def three_hump_camel_f(xk):
    return 2*xk[0]**2 - 1.05*xk[0]**4 + ((xk[0]**6)/6) + xk[0]*xk[1] + xk[1]**2

def himmelblau_f(xk):
    return (xk[0]**2 + xk[1] - 11)**2 + (xk[0] + xk[1]**2 - 7)**2

def grad_rosenbrock(xk): # liczony gradient dla Rosenbrocka
    grad_x1 = -2 + 2 * xk[0] - 400 * xk[0] * xk[1] + 400 * (xk[0]**3)
    grad_x2 = 200 * xk[1] - 200*(xk[0]**2)
    return np.array([grad_x1, grad_x2])


def grad_three_hump_camel(xk): # liczony gradient dla wielbłądowej 
    grad_x1 = 4 * xk[0] - 4 * 1.05 * (xk[0]**3) + (xk[0]**5) + xk[1]
    grad_x2 = xk[0] + 2 * xk[1]
    return np.array([grad_x1, grad_x2])

def grad_himmelblau(xk): # liczony gradient dla f himmelblaua
    grad_x1 = 4 * xk[0] * ((xk[0]**2) + xk[1] - 11) + 2 * (xk[0] + (xk[1]**2) - 7)
    grad_x2 = 4 * xk[1] * (xk[0] + (xk[1]**2)-7) + 2 * ((xk[0]**2) + xk[1] - 11)
    return np.array([grad_x1, grad_x2])

    
def Quasi_Newton_BFGS(start_x1, start_x2, funkcja_celu):
    

    xk = np.array([start_x1, start_x2])
    Bk = np.eye(2)
    grad_res = np.array([0,0])

    if(funkcja_celu == 1): # tu dla rosenbrocka będzie optymalizacja
        
        grad_res = grad_rosenbrock(xk)
        pk = np.linalg.solve(Bk, -grad_res)
        alpha_k, fc, gc, f1, f2, f3 = line_search(rosenbrock_f, grad_rosenbrock, xk, pk)
        return alpha_k 




#     elif(funkcja_celu == 2): # tu dla wielbłądowej będzie optymalizacja


#     elif(funkcja_celu == 3): # tu dla himmelblaua będzie optymalizacja

