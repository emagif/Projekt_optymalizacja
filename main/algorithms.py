import numpy as np
from scipy.optimize import line_search
from draw_func import *

### ROSENBROCK

def rosenbrock_f(xk):
    return (1-xk[0])**2 + 100*(xk[1] - xk[0]**2)**2

def grad_rosenbrock(xk):
    grad_x1 = -2*(1 - xk[0]) - 400*xk[0]*(xk[1] - xk[0]**2)
    grad_x2 = 200*(xk[1] - xk[0]**2)
    return np.array([grad_x1, grad_x2])

def rosenbrock_penalty_f(xk, penalty=100):

    f = (1-xk[0])**2 + 100*(xk[1] - xk[0]**2)**2

    g = 1.5 - 0.5*xk[0] - xk[1]

    return f + penalty * g**2

def grad_rosenbrock_penalty(xk, penalty=100):
    
    grad_f_x0 = -2*(1-xk[0]) - 400*xk[0]*(xk[1] - xk[0]**2)
    grad_f_x1 = 200*(xk[1] - xk[0]**2)

    g = 1.5 - 0.5*xk[0] - xk[1]

    coef = 2 * penalty * g
    
    grad_x0 = grad_f_x0 + coef * (-0.5)
    grad_x1 = grad_f_x1 + coef * (-1.0)

    return np.array([grad_x0, grad_x1])




### THREE HUMP CAMEL

def three_hump_camel_f(xk):
    return 2*xk[0]**2 - 1.05*xk[0]**4 + ((xk[0]**6)/6) + xk[0]*xk[1] + xk[1]**2

def grad_three_hump_camel(xk): 
    grad_x1 = 4 * xk[0] - 4 * 1.05 * (xk[0]**3) + (xk[0]**5) + xk[1]
    grad_x2 = xk[0] + 2 * xk[1]
    return np.array([grad_x1, grad_x2])

def three_hump_camel_penalty_f(xk, penalty=100):
    violation = max(0, xk[0]**2 + xk[1]**2 - 1)
    f = 2*xk[0]**2 - 1.05*xk[0]**4 + ((xk[0]**6)/6) + xk[0]*xk[1] + xk[1]**2
    return f + penalty * violation**2

def grad_three_hump_camel_penalty(xk, penalty=100):
    grad_f_x1 = 4*xk[0] - 4*1.05*(xk[0]**3) + (xk[0]**5) + xk[1]
    grad_f_x2 = xk[0] + 2*xk[1]

    g = xk[0]**2 + xk[1]**2 - 1

    if g <= 0:
        return np.array([grad_f_x1, grad_f_x2])

    grad_common = 2 * penalty * g

    penalty_x1 = grad_common * (2*xk[0])
    penalty_x2 = grad_common * (2*xk[1])

    return np.array([
        grad_f_x1 + penalty_x1,
        grad_f_x2 + penalty_x2
    ])


### HIMMEBLAU

def himmelblau_f(xk):
    return (xk[0]**2 + xk[1] - 11)**2 + (xk[0] + xk[1]**2 - 7)**2

def grad_himmelblau(xk): 
    grad_x1 = 4 * xk[0] * ((xk[0]**2) + xk[1] - 11) + 2 * (xk[0] + (xk[1]**2) - 7)
    grad_x2 = 4 * xk[1] * (xk[0] + (xk[1]**2)-7) + 2 * ((xk[0]**2) + xk[1] - 11)
    return np.array([grad_x1, grad_x2])

def himmelblau_penalty_f(xk, penalty=100):

    g = xk[0] - xk[1] + 3

    violation = max(0, g)

    f = (xk[0]**2 + xk[1] - 11)**2 + \
        (xk[0] + xk[1]**2 - 7)**2

    return f + penalty * violation**2

def grad_himmelblau_penalty(xk, penalty=100):

    grad_f_x1 = 4*xk[0]*(xk[0]**2 + xk[1] - 11) + \
                2*(xk[0] + xk[1]**2 - 7)

    grad_f_x2 = 4*xk[1]*(xk[0] + xk[1]**2 - 7) + \
                2*(xk[0]**2 + xk[1] - 11)

    g = xk[0] - xk[1] + 3

    if g <= 0:
        return np.array([grad_f_x1, grad_f_x2])

    grad_penalty = 2 * penalty * g * np.array([1.0, -1.0])

    return np.array([
        grad_f_x1 + grad_penalty[0],
        grad_f_x2 + grad_penalty[1]
    ])


### FUNKCJE OPTYMALIZUJĄCE

def Quasi_Newton_BFGS(start_x1, start_x2, functions):

    xk = np.array([start_x1, start_x2], dtype=float)
    Hk = np.eye(2)

    grad_k = functions[1](xk)

    tol = 1e-7
    max_iter = 1000
    i = 0

    path = [xk.copy()]
    f_values = [functions[0](xk)]

    while np.linalg.norm(grad_k) >= tol and i < max_iter:

        fk = functions[0](xk)

        pk = -Hk @ grad_k

        ls = line_search(
            functions[0],
            functions[1],
            xk,
            pk,
            grad_k,
            c1=1e-4,
            c2=0.9
        )

        alpha_k = ls[0]

        if alpha_k is None:
            alpha_k = 1e-3

        xk_1 = xk + alpha_k * pk

        path.append(xk_1.copy())
        f_values.append(functions[0](xk_1))

        grad_k_1 = functions[1](xk_1)

        sk = xk_1 - xk
        yk = grad_k_1 - grad_k

        if yk @ sk > 1e-10:
            rho_k = 1.0 / (yk @ sk + 1e-12)

            I = np.eye(2)
            Hk = (I - rho_k * np.outer(sk, yk)) @ Hk @ (I - rho_k * np.outer(yk, sk)) \
                 + rho_k * np.outer(sk, sk)

        xk = xk_1
        grad_k = grad_k_1

        if np.any(np.isnan(xk)) or np.any(np.isinf(xk)):
            break

        i += 1

    path = np.array(path)
    iter_num = len(path)

    return functions[0](xk), xk[0], xk[1], iter_num, f_values, path
    
    

def Quasi_Newton_DFP(start_x1, start_x2, functions):

    xk = np.array([start_x1, start_x2], dtype=float)
    Hk = np.eye(2)
    grad_k = functions[1](xk)

    tol = 1e-7
    max_iter = 1000

    path = [xk.copy()]
    f_values = [functions[0](xk)]

    i = 0

    while np.linalg.norm(grad_k) > tol and i < max_iter:

        fk = functions[0](xk)

        pk = -Hk @ grad_k

        if grad_k @ pk >= 0:
            Hk = np.eye(2)
            pk = -grad_k

        ls = line_search(
            functions[0],
            functions[1],
            xk,
            pk,
            grad_k,
            c1=1e-4,
            c2=0.9
        )
        
        alpha = ls[0]

        if alpha is None:
            alpha = 1e-3

        xk_new = xk + alpha * pk

        path.append(xk_new.copy())
        f_values.append(functions[0](xk_new))

        grad_new = functions[1](xk_new)

        sk = xk_new - xk
        yk = grad_new - grad_k

        ys = yk @ sk

        if ys > 1e-10:

            Hy = Hk @ yk
            denom2 = yk @ Hy

            if abs(denom2) > 1e-12:
                Hk = (
                    Hk
                    + np.outer(sk, sk) / ys
                    - np.outer(Hy, Hy) / denom2
                )

        xk = xk_new
        grad_k = grad_new

        if np.any(np.isnan(xk)) or np.any(np.isinf(xk)):
            break

        i += 1

    path = np.array(path)

    return (
        functions[0](xk),
        xk[0],
        xk[1],
        len(path),
        f_values,
        path
    )
