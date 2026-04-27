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


def rosenbrock_f_draw_3D_surf(x1, x2, span, result):
    x1_new = np.linspace(x1[0], x1[1], span)
    x2_new = np.linspace(x2[0], x2[1], span)

    X, Y = np.meshgrid(x1_new, x2_new)
    Z = (1 - X)**2 + 100 * (Y - X**2)**2

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

    ax.set_title('Funkcja Rosenbrocka')
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("f(x1, x2)")
    fig.colorbar(surf, shrink=0.5)

    plt.show()



# Ta wielbłądowa funkcja 

def three_hump_camel_f_draw_contour(x1, x2, levels, span, result, path=None):
    x1_new = np.linspace(x1[0], x1[1], span)
    x2_new = np.linspace(x2[0], x2[1], span)

    X, Y = np.meshgrid(x1_new, x2_new)
    Z = 2*X**2 - 1.05 * X**4 + ((X**6)/6) + X*Y + Y**2

    plt.figure()
    plt.contour(X, Y, Z, levels = levels)
    plt.colorbar()
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Funkcja Three-Hump Camel")
    plt.show()


def three_hump_camel_f_draw_3D(x1, x2, span):
    x1_new = np.linspace(x1[0], x1[1], span)
    x2_new = np.linspace(x2[0], x2[1], span)

    X, Y = np.meshgrid(x1_new, x2_new)
    Z = 2*X**2 - 1.05 * X**4 + ((X**6)/6) + X*Y + Y**2


    fig = plt.figure()
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

    ax.set_title('Funkcja Three-Hump Camel')
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("f(x1, x2)")
    fig.colorbar(surf, shrink=0.5)

    plt.show()



# Tutaj funkcja Himmelblaua 

def himmelblau_f_draw_contour(x1, x2, levels, span, result, path=None):
    x1_new = np.linspace(x1[0], x1[1], span)
    x2_new = np.linspace(x2[0], x2[1], span)

    X, Y = np.meshgrid(x1_new, x2_new)
    Z = (X**2 + Y - 11)**2 + (X + Y**2 - 7)**2

    plt.figure()
    plt.contour(X, Y, Z, levels = levels)
    plt.colorbar()
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Funkcja Himmeblaua")
    plt.show()



def himmelblau_f_draw_3D(x1, x2, span):
    x1_new = np.linspace(x1[0], x1[1], span)
    x2_new = np.linspace(x2[0], x2[1], span)

    X, Y = np.meshgrid(x1_new, x2_new)
    Z = (X**2 + Y - 11)**2 + (X + Y**2 - 7)**2


    fig = plt.figure()
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

    ax.set_title('Funkcja Himmeblau')
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("f(x1, x2)")
    fig.colorbar(surf, shrink=0.5)

    plt.show()