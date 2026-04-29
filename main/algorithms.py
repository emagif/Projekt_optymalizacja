import numpy as np
from scipy.optimize import line_search
from draw_func import *

def rosenbrock_f(xk):
    return (1-xk[0])**2 + 100*(xk[1] - xk[0]**2)**2

def three_hump_camel_f(xk):
    return 2*xk[0]**2 - 1.05*xk[0]**4 + ((xk[0]**6)/6) + xk[0]*xk[1] + xk[1]**2

def himmelblau_f(xk):
    return (xk[0]**2 + xk[1] - 11)**2 + (xk[0] + xk[1]**2 - 7)**2

def grad_rosenbrock(xk): #  gradient liczony dla Rosenbrocka
    grad_x1 = -2*(1 - xk[0]) - 400*xk[0]*(xk[1] - xk[0]**2)
    grad_x2 = 200*(xk[1] - xk[0]**2)
    return np.array([grad_x1, grad_x2])


def grad_three_hump_camel(xk): # liczony gradient dla wielbłądowej 
    grad_x1 = 4 * xk[0] - 4 * 1.05 * (xk[0]**3) + (xk[0]**5) + xk[1]
    grad_x2 = xk[0] + 2 * xk[1]
    return np.array([grad_x1, grad_x2])

def grad_himmelblau(xk): 
    grad_x1 = 4 * xk[0] * ((xk[0]**2) + xk[1] - 11) + 2 * (xk[0] + (xk[1]**2) - 7)
    grad_x2 = 4 * xk[1] * (xk[0] + (xk[1]**2)-7) + 2 * ((xk[0]**2) + xk[1] - 11)
    return np.array([grad_x1, grad_x2])

    
def Quasi_Newton_BFGS(start_x1, start_x2, function, grad, draw_contour, draw_3D, x_bounds, y_bounds):

    xk = np.array([start_x1, start_x2], dtype=float)
    Hk = np.eye(2)
    grad_k = grad(xk)

    tol = 1e-7
    max_iter = 1000
    i = 0

    path = [xk.copy()]

    while np.linalg.norm(grad_k) >= tol and i < max_iter:

        val_k = function(xk)
        pk = -Hk @ grad_k

        alpha_k = 1.0
        c = 1e-4
        while function(xk + alpha_k * pk) > val_k + c * alpha_k * grad_k @ pk:
            alpha_k *= 0.5
            if alpha_k < 1e-8:
                break

        xk_1 = xk + alpha_k * pk
        path.append(xk_1.copy())

        grad_k_1 = grad(xk_1)
        sk = xk_1 - xk
        yk = grad_k_1 - grad_k

        if yk @ sk > 1e-10:
            rho_k = 1.0 / (yk @ sk + 1e-12)
            I = np.eye(2)
            Hk = (I - rho_k * np.outer(sk, yk)) @ Hk @ (I - rho_k * np.outer(yk, sk)) + rho_k * np.outer(sk, sk)

        xk = xk_1
        grad_k = grad_k_1

        if np.any(np.isnan(xk)) or np.any(np.isinf(xk)):
            break

        i += 1
    path = np.array(path)
    draw_contour([x_bounds[0], x_bounds[1]], [y_bounds[0], y_bounds[1]], 50, 100, [start_x1, start_x2], [xk[0], xk[1]], path)
    draw_3D([x_bounds[0], x_bounds[1]], [y_bounds[0], y_bounds[1]], 50, [xk[0], xk[1]], function(xk))
    iter_num = len(path)
    return function(xk), xk[0], xk[1], iter_num


def Quasi_Newton_DFP(start_x1, start_x2, function, grad, draw_contour, draw_3D, x_bounds, y_bounds):
    
    xk = np.array([start_x1, start_x2])
    xk_first = xk.copy()
    Hk = np.eye(2) * 0.1
    grad_k = grad(xk)
    i = 0
    stop_cond = 1e-6
    best_result = float('inf')
    differ = 1
    tol = 1e-7
    max_iter = 1000


    path = []
    path.append(xk.copy())
    while np.linalg.norm(grad_k) >= tol and max_iter >= len(path):
            
        val_k = function(xk)
        pk = -Hk @ grad_k

        alpha_k = 1.0
        while function(xk + alpha_k * pk) > function(xk):
            alpha_k *= 0.5
            if alpha_k < 1e-8:
                break

            
        xk_1 = xk + alpha_k * pk
        path.append(xk_1.copy())
        val_k_1 = function(xk_1)
        grad_k_1 = grad(xk_1)
        sk = xk_1 - xk 
        yk = grad_k_1 - grad_k

        if yk @ sk <= 1e-10:
            Hk = np.eye(2)
            xk = xk_1
            grad_k = grad_k_1
            i += 1
            continue


        Hk_1 = Hk + (np.outer(sk, sk))/(sk @ yk) - (Hk @ np.outer(yk, yk) @ Hk)/(yk @ Hk @ yk) # tutaj np.outer(x,y) to iloczyn zewnętrzny także matma się zgadza

        differ = val_k_1 - val_k
        xk = xk_1
        Hk = Hk_1
        val_k = val_k_1
        i+=1
        best_result = val_k_1
        grad_k = grad_k_1

        if np.any(np.isnan(xk)) or np.any(np.isinf(xk)):
            break
    path = np.array(path)
    draw_contour([x_bounds[0], x_bounds[1]], [y_bounds[0], y_bounds[1]], 50, 100, [start_x1, start_x2], [xk[0], xk[1]], path)
    draw_3D([x_bounds[0], x_bounds[1]], [y_bounds[0], y_bounds[1]], 50, [xk[0], xk[1]], function(xk))
    iter_num = len(path)
    return function(xk), xk_1[0], xk_1[1], iter_num

    
    