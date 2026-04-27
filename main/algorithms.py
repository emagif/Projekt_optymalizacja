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

    
def Quasi_Newton_BFGS(start_x1, start_x2, function, grad):
    

    xk = np.array([start_x1, start_x2])
    Hk = np.eye(2) * 0.1
    grad_k = grad(xk)
    i = 0
    stop_cond = 1e-6
    best_result = float('inf')
    differ = 1
    tol = 1e-7
    max_iter = 1000

    alpha_max = 0.1

    # grad_k = grad_rosenbrock(xk)

    while np.linalg.norm(grad_k) >= tol and max_iter >= i:
            
        val_k = function(xk)
        # pk = np.linalg.solve(Hk, -grad_k)
        pk = -Hk @ grad_k
        alpha_k, fc, gc, f1, f2, f3 = line_search(function, grad, xk, pk) 

        if(alpha_k is None or alpha_k > alpha_max):
            alpha_k = alpha_max

        print(f"alpha_k: {alpha_k}")
            
        xk_1 = xk + alpha_k * pk
        val_k_1 = function(xk_1)
        grad_k_1 = grad(xk_1)
        sk = xk_1 - xk 
        yk = grad_k_1 - grad_k
        rho_k = 1.0 / (yk @ sk + 1e-12) # algebra siedzi, nawet jak yk nie jest transponowane (sprawdzone) - można unit test napisać; ta dodana wartość jest po to, żeby mianownik się nie wyzerował
        Hk_1 = (np.eye(2) - rho_k * np.outer(sk,yk)) @ Hk @(np.eye(2) - rho_k * np.outer(yk, sk)) + rho_k * np.outer(sk, sk) # tutaj np.outer(x,y) to iloczyn zewnętrzny także matma się zgadza

        differ = val_k_1 - val_k
        xk = xk_1
        Hk = Hk_1
        val_k = val_k_1
        i+=1
        best_result = val_k_1
        grad_k = grad_k_1

        if np.any(np.isnan(xk)) or np.any(np.isinf(xk)):
            break

    return best_result, xk_1[0], xk_1[1], i





def Quasi_Newton_DFP(start_x1, start_x2, function, grad):
    

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

    alpha_max = 0.1

    path = []
    path.append(xk.copy())
    while np.linalg.norm(grad_k) >= tol and max_iter >= i:
            
        val_k = function(xk)
        pk = -Hk @ grad_k

        alpha_k = 1.0
        while function(xk + alpha_k * pk) > function(xk):
            alpha_k *= 0.5
            if alpha_k < 1e-8:
                break

        print(f"alpha_k: {alpha_k}")
            
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
    rosenbrock_f_draw_contour([-2, 2], [-1, 3], 30, 200, [xk_first[0], xk_first[1]], [xk_1[0], xk_1[1]], path)
    return best_result, xk_1[0], xk_1[1], i

    
    