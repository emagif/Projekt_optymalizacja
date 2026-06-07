import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


### ROSENBROCK

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

def rosenbrock_f_draw_contour_3_bez_ograniczen(x1, x2, levels, resolution,
                                             path1, path2, path3, path4):

    x1_new = np.linspace(x1[0], x1[1], resolution)
    x2_new = np.linspace(x2[0], x2[1], resolution)

    X, Y = np.meshgrid(x1_new, x2_new)

    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x = X[i, j]
            y = Y[i, j]

            Z[i, j] = (1 - x)**2 + 100 * (y - x**2)**2

    plt.figure()
    plt.contour(X, Y, Z, levels=levels, linewidths=0.5)
    plt.colorbar()

    colors = ['blue', 'orange', 'green', 'red']
    
    labels = [
        'DFP start [1.75,3]',
        'DFP start [0,3]',
        'BFGS start [1.75,3]',
        'BFGS start [0,3]'
    ]
    markers = ['o', 's', '^', 'x']
    for i, path in enumerate([path1, path2, path3, path4]):
        path = np.array(path)
        plt.plot(path[:, 0], path[:, 1],
                 marker=markers[i],
                 color=colors[i],
                 linewidth=1,
                 label=labels[i])

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Funkcja Rosenbrocka bez ograniczeń")
    plt.legend()
    plt.grid(True)
    plt.show()


def rosenbrock_f_draw_contour_3_z_ograniczeniami(x1, x2, levels, resolution,
                                             path1, path2, path3, path4):

    x1_new = np.linspace(x1[0], x1[1], resolution)
    x2_new = np.linspace(x2[0], x2[1], resolution)

    X, Y = np.meshgrid(x1_new, x2_new)

    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x = X[i, j]
            y = Y[i, j]

            Z[i, j] = (1 - x)**2 + 100 * (y - x**2)**2

    plt.figure()
    plt.contour(X, Y, Z, levels=levels, linewidths=0.5)
    plt.colorbar()

    Xg, Yg = np.meshgrid(x1_new, x2_new)

    line_x = x1_new
    line_y = 1.5 - 0.5 * line_x

    plt.plot(
        line_x,
        line_y,
        color='black',
        linewidth=1.5,
        label='g(x)=0: 1.5 - 0.5x1 - x2 = 0'
    )

    colors = ['blue', 'orange', 'green', 'red']

    labels = [
        'DFP start [2,3.5]',
        'DFP start [0,3]',
        'BFGS start [2,3.5]',
        'BFGS start [0,3]'
    ]
    markers = ['o', 's', '^', 'x']
    for i, path in enumerate([path1, path2, path3, path4]):
        path = np.array(path)
        plt.plot(path[:, 0], path[:, 1],
                 marker=markers[i],
                 color=colors[i],
                 linewidth=1,
                 label=labels[i])

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Funkcja Rosenbrocka z ograniczeniem liniowym")
    plt.legend()
    plt.grid(True)
    plt.show()



# THREE HUMP CAMEL

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

def three_hump_camel_f_draw_contour_3_bez_ograniczen(x1, x2, levels, resolution,
                                    path1, path2, path3, path4):

    x1_new = np.linspace(x1[0], x1[1], resolution)
    x2_new = np.linspace(x2[0], x2[1], resolution)

    X, Y = np.meshgrid(x1_new, x2_new)

    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x = X[i, j]
            y = Y[i, j]

            Z[i, j] = (2*x**2
                       - 1.05*x**4
                       + (x**6)/6
                       + x*y
                       + y**2)

    plt.figure()
    plt.contour(X, Y, Z, levels=levels, linewidths=0.5)
    plt.colorbar()

    colors = ['blue', 'orange', 'green', 'red']

    labels = [
        'DFP start [-2,0]',
        'DFP start [2,0]',
        'BFGS start [-2,0]',
        'BFGS start [2,0]'
    ]

    for i, path in enumerate([path1, path2, path3, path4]):
        path = np.array(path)
        plt.plot(path[:, 0], path[:, 1],
                 'o-',
                 color=colors[i],
                 linewidth=1,
                 label=labels[i])

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Funkcja Three-Hump Camel bez ograniczeń")
    plt.legend()
    plt.grid(True)
    plt.show()

def three_hump_camel_f_draw_contour_3_z_ograniczeniami(x1, x2, levels, resolution,
                                    path1, path2, path3, path4):

    x1_new = np.linspace(x1[0], x1[1], resolution)
    x2_new = np.linspace(x2[0], x2[1], resolution)

    X, Y = np.meshgrid(x1_new, x2_new)

    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x = X[i, j]
            y = Y[i, j]

            Z[i, j] = (2*x**2
                       - 1.05*x**4
                       + (x**6)/6
                       + x*y
                       + y**2)

    plt.figure()
    plt.contour(X, Y, Z, levels=levels, linewidths=0.5)
    plt.colorbar()

    theta = np.linspace(0, 2*np.pi, 400)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)

    Xg, Yg = np.meshgrid(x1_new, x2_new)

    mask = Xg**2 + Yg**2 > 1

    plt.contourf(
    Xg,
    Yg,
    mask,
    levels=[0.5, 1],
    colors=['gray'],
    alpha=0.2
)
    plt.plot(
    circle_x,
    circle_y,
    color='black',
    linewidth=1.5,
    label='x1² + x2² = 1'
)
    colors = ['blue', 'orange', 'green', 'red']

    labels = [
        'x = [-2, 0] DFP',
        'x = [2, 0] DFP',
        'x = [-2, 0] BFGS',
        'x = [2, 0] BFGS'
    ]

    for i, path in enumerate([path1, path2, path3, path4]):
        path = np.array(path)
        plt.plot(path[:, 0], path[:, 1],
                 'o-',
                 color=colors[i],
                 linewidth=1,
                 label=labels[i])

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Funkcja Three-Hump Camel z ograniczeniami")
    plt.legend()
    plt.grid(True)
    plt.show()




### HIMMELBLAU

def himmelblau_f_draw_contour(x1, x2, levels, resolution, xk_first, result, path=None):
    x1_new = np.linspace(x1[0], x1[1], resolution)
    x2_new = np.linspace(x2[0], x2[1], resolution)

    X, Y = np.meshgrid(x1_new, x2_new)
    Z = (X**2 + Y - 11)**2 + (X + Y**2 - 7)**2

    plt.figure()
    plt.contour(X, Y, Z, levels = levels, linewidths=0.5)
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


def himmelblau_f_draw_contour_3_z_ograniczeniami(x1, x2, levels, resolution,
                                path1, path2, path3, path4):

    x1_new = np.linspace(x1[0], x1[1], resolution)
    x2_new = np.linspace(x2[0], x2[1], resolution)

    X, Y = np.meshgrid(x1_new, x2_new)

    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = (X[i, j]**2 + Y[i, j] - 11)**2 + (X[i, j] + Y[i, j]**2 - 7)**2

    plt.figure()
    plt.contour(X, Y, Z, levels=levels, linewidths=0.5)
    plt.colorbar()

    xmin, xmax = x1[0], x1[1]
    ymin, ymax = x2[0], x2[1]

    candidates = []

    y = xmin + 3
    if ymin <= y <= ymax:
        candidates.append([xmin, y])

    y = xmax + 3
    if ymin <= y <= ymax:
        candidates.append([xmax, y])

    x = ymin - 3
    if xmin <= x <= xmax:
        candidates.append([x, ymin])

    x = ymax - 3
    if xmin <= x <= xmax:
        candidates.append([x, ymax])

    candidates = np.array(candidates)

    xmin, xmax = x1[0], x1[1]
    ymin, ymax = x2[0], x2[1]

    x_fill = np.linspace(xmin, xmax, 400)
    y_line = x_fill + 3

    y_upper = np.minimum(y_line, ymax)

    plt.fill_between(
    x_fill,
    ymin,
    y_upper,
    color='gray',
    alpha=0.2
)

    if len(candidates) >= 2:
        candidates = candidates[np.argsort(candidates[:, 0])]

        plt.plot(
            candidates[:, 0],
            candidates[:, 1],
            color='black',
            linewidth=1.5,  
            label='x1 - x2 + 3 = 0'
        )

    colors = ['blue', 'orange', 'green', 'red']
    labels = ['x = [0,0] DFP', 'x = [-0.39,-0.64] DFP', 'x = [0,0] BFGS', 'x = [-0.39,-0.64] BFGS']
    for i, path in enumerate([path1, path2, path3, path4]):
        path = np.array(path)
        plt.plot(path[:, 0], path[:, 1],
                 'o-',
                 color=colors[i],
                 linewidth=1,
                 label=labels[i])
        


    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Funkcja Himmelblau z ograniczeniami")
    plt.legend()
    plt.grid(True)
    plt.show()

def himmelblau_f_draw_contour_3_bez_ograniczen(x1, x2, levels, resolution,
                                path1, path2, path3, path4):

    x1_new = np.linspace(x1[0], x1[1], resolution)
    x2_new = np.linspace(x2[0], x2[1], resolution)

    X, Y = np.meshgrid(x1_new, x2_new)

    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = (X[i, j]**2 + Y[i, j] - 11)**2 + (X[i, j] + Y[i, j]**2 - 7)**2

    plt.figure()
    plt.contour(X, Y, Z, levels=levels, linewidths=0.5)
    plt.colorbar()


    colors = ['blue', 'orange', 'green', 'red']
    labels = ['x = [0,0] DFP', 'x = [-0.39,-0.64] DFP', 'x = [0,0] BFGS', 'x = [-0.39,-0.64] BFGS']
    for i, path in enumerate([path1, path2, path3, path4]):
        path = np.array(path)
        plt.plot(path[:, 0], path[:, 1],
                 'o-',
                 color=colors[i],
                 linewidth=1,
                 label=labels[i])

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Funkcja Himmelblau bez ograniczeń")
    plt.legend()
    plt.grid(True)
    plt.show()


### ITERACJE

def drawFValuesLog(f_values1, f_values2, f_values3, f_values4,
                   label1='Metoda 1',
                   label2='Metoda 2',
                   label3='Metoda 3', 
                   label4='Metoda 4'):

    plt.figure(figsize=(8, 5))

    plt.semilogy(np.arange(len(f_values1)), f_values1, marker='o', label=label1, color='blue')
    plt.semilogy(np.arange(len(f_values2)), f_values2, marker='s', label=label2, color='orange')
    plt.semilogy(np.arange(len(f_values3)), f_values3, marker='^', label=label3, color='green')
    plt.semilogy(np.arange(len(f_values4)), f_values4, marker='x', label=label4, color='red')

    plt.xlabel('Numer iteracji')
    plt.ylabel('Wartość funkcji celu')
    plt.title('Porównanie zbieżności w zależności od punktu początkowego (skala logarytmiczna)')
    plt.grid(True, which='both')
    plt.legend()

    plt.show()


def drawFValues(f_values1, f_values2, f_values3, f_values4,
                label1='Metoda 1',
                label2='Metoda 2',
                label3='Metoda 3', 
                label4='Metoda 4'):

    plt.figure(figsize=(8, 5))

    plt.plot(np.arange(len(f_values1)), f_values1, marker='o', label=label1, color='blue')
    plt.plot(np.arange(len(f_values2)), f_values2, marker='s', label=label2, color='orange')
    plt.plot(np.arange(len(f_values3)), f_values3, marker='^', label=label3, color='green')
    plt.plot(np.arange(len(f_values4)), f_values4, marker='x', label=label4, color='red')

    plt.xlabel('Numer iteracji')
    plt.ylabel('Wartość funkcji celu')
    plt.title('Porównanie zbieżności w zależności od punktu początkowego')
    plt.grid(True)
    plt.legend()

    plt.show()

