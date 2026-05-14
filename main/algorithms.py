import numpy as np
from scipy.optimize import line_search
from draw_func import *



def softplus(g):
    return np.log1p(np.exp(-np.abs(g))) + np.maximum(g, 0)

def sigmoid(g):
    return 1 / (1 + np.exp(-np.clip(g, -50, 50)))


### ROSENBROCK

def rosenbrock_f(xk):
    return (1-xk[0])**2 + 100*(xk[1] - xk[0]**2)**2

def grad_rosenbrock(xk): #  gradient liczony dla Rosenbrocka
    grad_x1 = -2*(1 - xk[0]) - 400*xk[0]*(xk[1] - xk[0]**2)
    grad_x2 = 200*(xk[1] - xk[0]**2)
    return np.array([grad_x1, grad_x2])

def rosenbrock_constrained(xk):
    return (1-xk[0])**2 + 100*(xk[1] - xk[0]**2)**2 + 1.5 - 0.5*xk[0] - xk[1]

def grad_rosenbrock_constrained(xk):
    grad_x1 = -2*(1 - xk[0]) - 400*xk[0]*(xk[1] - xk[0]**2) - 0.5
    grad_x2 = 200*(xk[1] - xk[0]**2) - 1
    return np.array([grad_x1, grad_x2])

def rosenbrock_penalty_f(xk, penalty=100):

    f = (1-xk[0])**2 + 100*(xk[1] - xk[0]**2)**2

    g = 1.5 - 0.5*xk[0] - xk[1]

    g_pos = softplus(g)

    return f + penalty * g_pos**2

def grad_rosenbrock_penalty(xk, penalty=100):
    
    grad_f_x0 = -2*(1-xk[0]) - 400*xk[0]*(xk[1] - xk[0]**2)
    grad_f_x1 = 200*(xk[1] - xk[0]**2)

    g = 1.5 - 0.5*xk[0] - xk[1]

    s = softplus(g)
    ds = sigmoid(g)

    coef = 2 * penalty * s * ds

    grad_x0 = grad_f_x0 + coef * (-0.5)
    grad_x1 = grad_f_x1 + coef * (-1.0)

    return np.array([grad_x0, grad_x1])




### THREE HUMP CAMEL

def three_hump_camel_f(xk):
    return 2*xk[0]**2 - 1.05*xk[0]**4 + ((xk[0]**6)/6) + xk[0]*xk[1] + xk[1]**2

def grad_three_hump_camel(xk): # liczony gradient dla wielbłądowej 
    grad_x1 = 4 * xk[0] - 4 * 1.05 * (xk[0]**3) + (xk[0]**5) + xk[1]
    grad_x2 = xk[0] + 2 * xk[1]
    return np.array([grad_x1, grad_x2])


def constraint_three_hump_camel(xk): # ograniczenia do funkcji three hump camel
    return 1-(xk[0]**2 + xk[1]**2)

def three_hump_camel_constrained(xk): # funkcja z dołożonymi ograniczeniami, policzonymi w funkcji constraint_three_hump_camel
    return 2*xk[0]**2 - 1.05*xk[0]**4 + ((xk[0]**6)/6) + xk[0]*xk[1] + xk[1]**2 + constraint_three_hump_camel(xk)

def constraint_three_hump_camel_grad(xk): # gradient samych ograniczeń !!!
    grad_x1 = -2*xk[0]
    grad_x2 = -2*xk[1]
    return np.array([grad_x1, grad_x2])

def grad_three_hump_camel_constraint(xk): # gradient całej funkcji razem z ograniczeniami
    grad_x1 = 4 * xk[0] - 4 * 1.05 * (xk[0]**3) + (xk[0]**5) + xk[1]-2*xk[0] 
    grad_x2 = xk[0] + 2 * xk[1]-2*xk[1]
    return np.array([grad_x1, grad_x2])


def three_hump_camel_penalty_f(xk, penalty=100):
    violation = max(0, xk[0]**2 + xk[1]**2 - 1)
    f = 2*xk[0]**2 - 1.05*xk[0]**4 + ((xk[0]**6)/6) + xk[0]*xk[1] + xk[1]**2
    return f + penalty * violation**2

def grad_three_hump_camel_penalty(xk, penalty=100):
    g = xk[0]**2 + xk[1]**2 - 1

    if g <= 0:
        return np.zeros(2)

    grad_common = 2 * penalty * g

    grad_x1 = grad_common * (2 * xk[0])
    grad_x2 = grad_common * (2 * xk[1])
    return np.array([grad_x1, grad_x2])


### HIMMEBLAU

def himmelblau_f(xk):
    return (xk[0]**2 + xk[1] - 11)**2 + (xk[0] + xk[1]**2 - 7)**2


def grad_himmelblau(xk): 
    grad_x1 = 4 * xk[0] * ((xk[0]**2) + xk[1] - 11) + 2 * (xk[0] + (xk[1]**2) - 7)
    grad_x2 = 4 * xk[1] * (xk[0] + (xk[1]**2)-7) + 2 * ((xk[0]**2) + xk[1] - 11)
    return np.array([grad_x1, grad_x2])


def himmelblau_constrained(xk):
    return (xk[0]**2 + xk[1] - 11)**2 + (xk[0] + xk[1]**2 - 7)**2 + xk[0] - xk[1] + 2


def grad_himmelblau_constrained(xk):
    grad_x1 = 4 * xk[0] * ((xk[0]**2) + xk[1] - 11) + 2 * (xk[0] + (xk[1]**2) - 7) + 1
    grad_x2 = 4 * xk[1] * (xk[0] + (xk[1]**2)-7) + 2 * ((xk[0]**2) + xk[1] - 11) - 1
    return np.array([grad_x1, grad_x2])

def himmelblau_penalty_f(xk, penalty=100):
    g = xk[0] - xk[1] + 2
    violation = max(0, g)
    f = (xk[0]**2 + xk[1] - 11)**2 + (xk[0] + xk[1]**2 - 7)**2
    return f + penalty * violation**2

def grad_himmelblau_penalty(xk, penalty=100):
    g = xk[0] - xk[1] + 2

    grad_f_x1 = 4 * xk[0] * (xk[0]**2 + xk[1] - 11) + 2 * (xk[0] + xk[1]**2 - 7)
    grad_f_x2 = 4 * xk[1] * (xk[0] + xk[1]**2 - 7) + 2 * (xk[0]**2 + xk[1] - 11)

    if g <= 0:
        return np.array([grad_f_x1, grad_f_x2])

    grad_penalty = 2 * penalty * g * np.array([1.0, -1.0])

    return np.array([
        grad_f_x1 + grad_penalty[0],
        grad_f_x2 + grad_penalty[1]
    ])




### FUNKCJE OPTYMALIZUJĄCE

def Quasi_Newton_BFGS(start_x1, start_x2, functions,  x_bounds, y_bounds): # functions przyjmuje funkcję celu, jej gradient, wykres konturowy, wykres 3D

    xk = np.array([start_x1, start_x2], dtype=float)
    Hk = np.eye(2)
    grad_k = functions[1](xk)

    tol = 1e-7
    max_iter = 1000
    i = 0

    path = [xk.copy()]

    while np.linalg.norm(grad_k) >= tol and i < max_iter:

        val_k = functions[0](xk)
        pk = -Hk @ grad_k

        alpha_k = 1.0
        c = 1e-4
        while functions[0](xk + alpha_k * pk) > val_k + c * alpha_k * grad_k @ pk:
            alpha_k *= 0.5
            if alpha_k < 1e-8:
                break

        xk_1 = xk + alpha_k * pk
        path.append(xk_1.copy())

        grad_k_1 = functions[1](xk_1)
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
    functions[2]([x_bounds[0], x_bounds[1]], [y_bounds[0], y_bounds[1]], 50, 100, [start_x1, start_x2], [xk[0], xk[1]], path)
    functions[3]([x_bounds[0], x_bounds[1]], [y_bounds[0], y_bounds[1]], 50, [xk[0], xk[1]], functions[0](xk))
    iter_num = len(path)
    return functions[0](xk), xk[0], xk[1], iter_num


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

    
    