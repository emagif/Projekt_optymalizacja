import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm



def rosenbrock_f_draw_contour(x1, x2, levels, resolution, xk_first, result, path=None):

    x1_new = np.linspace(x1[0], x1[1], resolution)
    x2_new = np.linspace(x2[0], x2[1], resolution)

    X, Y = np.meshgrid(x1_new, x2_new)
    Z = (1 - X)**2 + 100 * (Y - X**2)**2

    plt.figure()
    plt.contour(X, Y, Z, levels=levels)
    plt.colorbar()

    plt.scatter(result[0], result[1], color='red', s=100, label='punkt końcowy')
    plt.scatter(xk_first[0], xk_first[1], color='green', s=100, label='punkt początkowy')

    if path is not None:
        path = np.array(path)
        plt.plot(path[:, 0], path[:, 1], 'o-', color='blue', linewidth=1, label='trajektoria')

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Funkcja Rosenbrocka")
    plt.legend()
    plt.show()

def rosenbrock_f_draw_3D_surf(x1, x2, span, result, func_val):
    x1_new = np.linspace(x1[0], x1[1], span)
    x2_new = np.linspace(x2[0], x2[1], span)

    X, Y = np.meshgrid(x1_new, x2_new)
    Z = (1 - X)**2 + 100 * (Y - X**2)**2

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.scatter(result[0], result[1], func_val, color='red', s=100)

    ax.set_title('Funkcja Rosenbrocka')
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("f(x1, x2)")
    fig.colorbar(surf, shrink=0.5)

    plt.show()

def rosenbrock_penalty_draw_contour(x1, x2, levels, resolution, xk_first, result, path=None, penalty=100):

    x1_new = np.linspace(x1[0], x1[1], resolution)
    x2_new = np.linspace(x2[0], x2[1], resolution)

    X, Y = np.meshgrid(x1_new, x2_new)

    # ROSENBROCK
    f = (1 - X)**2 + 100 * (Y - X**2)**2

    # constraint
    g = 1.5 - 0.5*X - Y
    # g_pos = np.log1p(np.exp(g))  # softplus (stabilne)

    Z = f + penalty * g**2

    plt.figure()
    plt.contour(X, Y, Z, levels=levels)
    plt.colorbar()

    plt.scatter(result[0], result[1], color='red', s=100, label='punkt końcowy')
    plt.scatter(xk_first[0], xk_first[1], color='green', s=100, label='punkt początkowy')

    if path is not None:
        path = np.array(path)
        plt.plot(path[:, 0], path[:, 1], 'o-', color='blue', linewidth=1, label='trajektoria')

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Rosenbrock + penalty")
    plt.legend()
    plt.show()

def rosenbrock_penalty_draw_3D(x1, x2, span, result, func_val, penalty=100):

    x1_new = np.linspace(x1[0], x1[1], span)
    x2_new = np.linspace(x2[0], x2[1], span)

    X, Y = np.meshgrid(x1_new, x2_new)

    f = (1 - X)**2 + 100 * (Y - X**2)**2

    g = 1.5 - 0.5*X - Y
    # g_pos = np.log1p(np.exp(g))

    Z = f + penalty * g**2

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

    ax.scatter(result[0], result[1], func_val, color='red', s=100)

    ax.set_title("Rosenbrock + penalty")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("f(x)")

    fig.colorbar(surf, shrink=0.5)
    plt.show()

# Ta wielbłądowa funkcja 

def three_hump_camel_f_draw_contour(x1, x2, levels, resolution, xk_first, result, path=None):
    x1_new = np.linspace(x1[0], x1[1], resolution)
    x2_new = np.linspace(x2[0], x2[1], resolution)

    X, Y = np.meshgrid(x1_new, x2_new)
    Z = 2*X**2 - 1.05 * X**4 + ((X**6)/6) + X*Y + Y**2


    plt.figure()
    plt.contour(X, Y, Z, levels = levels)
    plt.colorbar()

    plt.scatter(result[0], result[1], color='red', s=100, label='punkt końcowy')
    plt.scatter(xk_first[0], xk_first[1], color='green', s=100, label='punkt początkowy')

    if path is not None:
        path = np.array(path)
        plt.plot(path[:, 0], path[:, 1], 'o-', color='blue', linewidth=1, label='trajektoria')

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Funkcja Three-Hump Camel")
    plt.show()

def three_hump_camel_f_draw_3D_surf(x1, x2, resolution, result, func_val):
    x1_new = np.linspace(x1[0], x1[1], resolution)
    x2_new = np.linspace(x2[0], x2[1], resolution)

    X, Y = np.meshgrid(x1_new, x2_new)
    Z = 2*X**2 - 1.05 * X**4 + ((X**6)/6) + X*Y + Y**2


    fig = plt.figure()
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.scatter(result[0], result[1], func_val, color='red', s=100)

    ax.set_title('Funkcja Three-Hump Camel')
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("f(x1, x2)")
    fig.colorbar(surf, shrink=0.5)

    plt.show()

def three_hump_camel_penalty_draw_contour(x1, x2, levels, resolution, xk_first, result, path=None, penalty=100):
    x1_new = np.linspace(x1[0], x1[1], resolution)
    x2_new = np.linspace(x2[0], x2[1], resolution)

    X, Y = np.meshgrid(x1_new, x2_new)

    # funkcja Three-Hump Camel
    f = 2*X**2 - 1.05*X**4 + (X**6)/6 + X*Y + Y**2

    # constraint
    g = X**2 + Y**2 - 1

    # kara tylko poza okręgiem
    violation = np.maximum(0, g)

    # funkcja z karą
    Z = f + penalty * violation**2

    plt.figure(figsize=(8,6))

    contour = plt.contour(X, Y, Z, levels=levels)
    plt.colorbar(contour)

    # ograniczenie: okrąg jednostkowy
    theta = np.linspace(0, 2*np.pi, 400)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)

    plt.plot(circle_x, circle_y,
             color='black',
             linewidth=2,
             label='x1² + x2² = 1')

    # punkty
    plt.scatter(result[0], result[1],
                color='red',
                s=100,
                label='punkt końcowy')

    plt.scatter(xk_first[0], xk_first[1],
                color='green',
                s=100,
                label='punkt początkowy')

    # trajektoria
    if path is not None:
        path = np.array(path)

        plt.plot(path[:,0], path[:,1],
                 'o-',
                 color='blue',
                 linewidth=1,
                 label='trajektoria')

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Funkcja Three-Hump Camel + penalty")

    plt.axis('equal')
    plt.legend()
    plt.show()

def three_hump_camel_penalty_draw_3D(
        x1, x2, resolution,
        result, func_val,
        penalty=100):

    x1_new = np.linspace(x1[0], x1[1], resolution)
    x2_new = np.linspace(x2[0], x2[1], resolution)

    X, Y = np.meshgrid(x1_new, x2_new)

    # funkcja bazowa
    f = 2*X**2 - 1.05*X**4 + (X**6)/6 + X*Y + Y**2

    # constraint
    g = X**2 + Y**2 - 1

    violation = np.maximum(0, g)

    # funkcja z karą
    Z = f + penalty * violation**2

    fig = plt.figure(figsize=(10,7))

    ax = plt.axes(projection='3d')

    surf = ax.plot_surface(
        X, Y, Z,
        cmap='viridis',
        edgecolor='none'
    )

    ax.scatter(
        result[0],
        result[1],
        func_val,
        color='red',
        s=100
    )

    ax.set_title("Three-Hump Camel + Penalty")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("f(x1,x2)")

    fig.colorbar(surf, shrink=0.5)

    plt.show()

# Tutaj funkcja Himmelblaua 

def himmelblau_f_draw_contour(x1, x2, levels, resolution, xk_first, result, path=None):
    x1_new = np.linspace(x1[0], x1[1], resolution)
    x2_new = np.linspace(x2[0], x2[1], resolution)

    X, Y = np.meshgrid(x1_new, x2_new)
    Z = (X**2 + Y - 11)**2 + (X + Y**2 - 7)**2

    plt.figure()
    plt.contour(X, Y, Z, levels = levels)
    plt.colorbar()


    plt.scatter(result[0], result[1], color='red', s=100, label='punkt końcowy')
    plt.scatter(xk_first[0], xk_first[1], color='green', s=100, label='punkt początkowy')

    if path is not None:
        path = np.array(path)
        plt.plot(path[:, 0], path[:, 1], 'o-', color='blue', linewidth=1, label='trajektoria')


    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Funkcja Himmeblaua")
    plt.show()

def himmelblau_f_draw_3D_surf(x1, x2, resolution, result, func_val):
    x1_new = np.linspace(x1[0], x1[1], resolution)
    x2_new = np.linspace(x2[0], x2[1], resolution)

    X, Y = np.meshgrid(x1_new, x2_new)
    Z = (X**2 + Y - 11)**2 + (X + Y**2 - 7)**2


    fig = plt.figure()
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.scatter(result[0], result[1], func_val, color='red', s=100)

    ax.set_title('Funkcja Himmeblau')
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("f(x1, x2)")
    fig.colorbar(surf, shrink=0.5)

    plt.show()

def himmelblau_penalty_draw_contour(
        x1, x2,
        levels,
        resolution,
        xk_first,
        result,
        path=None,
        penalty=100):

    x1_new = np.linspace(x1[0], x1[1], resolution)
    x2_new = np.linspace(x2[0], x2[1], resolution)

    X, Y = np.meshgrid(x1_new, x2_new)

    # funkcja Himmelblaua
    f = (X**2 + Y - 11)**2 + (X + Y**2 - 7)**2

    # constraint
    g = X - Y + 3

    violation = np.maximum(0, g)

    # funkcja z karą
    Z = f + penalty * violation**2

    plt.figure(figsize=(8,6))

    contour = plt.contour(X, Y, Z, levels=levels)
    plt.colorbar(contour)

    # prosta constraintu:
    # x1 - x2 + 2 = 0
    # x2 = x1 + 2
    line_x = np.linspace(x1[0], x1[1], 400)
    line_y = line_x + 3

    plt.plot(
        line_x,
        line_y,
        color='black',
        linewidth=2,
        label='x1 - x2 + 3 = 0'
    )

    # punkt startowy
    plt.scatter(
        xk_first[0],
        xk_first[1],
        color='green',
        s=100,
        label='punkt początkowy'
    )

    # punkt końcowy
    plt.scatter(
        result[0],
        result[1],
        color='red',
        s=100,
        label='punkt końcowy'
    )

    # trajektoria
    if path is not None:
        path = np.array(path)

        plt.plot(
            path[:,0],
            path[:,1],
            'o-',
            color='blue',
            linewidth=1,
            label='trajektoria'
        )

    plt.xlabel("x1")
    plt.ylabel("x2")

    plt.title("Himmelblau + Penalty")

    plt.legend()
    plt.show()

def himmelblau_penalty_draw_3D(
        x1, x2,
        resolution,
        result,
        func_val,
        penalty=100):

    x1_new = np.linspace(x1[0], x1[1], resolution)
    x2_new = np.linspace(x2[0], x2[1], resolution)

    X, Y = np.meshgrid(x1_new, x2_new)

    # funkcja Himmelblaua
    f = (X**2 + Y - 11)**2 + \
        (X + Y**2 - 7)**2

    # constraint
    g = X - Y + 3

    violation = np.maximum(0, g)

    # penalty
    Z = f + penalty * violation**2

    fig = plt.figure(figsize=(10,7))

    ax = plt.axes(projection='3d')

    surf = ax.plot_surface(
        X, Y, Z,
        cmap='viridis',
        edgecolor='none'
    )

    ax.scatter(
        result[0],
        result[1],
        func_val,
        color='red',
        s=100
    )

    ax.set_title("Himmelblau + Penalty")

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("f(x1,x2)")

    fig.colorbar(surf, shrink=0.5)

    plt.show()