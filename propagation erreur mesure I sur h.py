import numpy as np
from matplotlib import pyplot as plt

n0 = 1  # indice de réfraction de l'air
n1 = 1.52  # indice de réfraction du biopolymère qui compose le nanofilm dont on veut mesurer l'épaisseur : idéalement ce n'est pas une valeur fixe mais un indice dépendant de la longueur d'onde, comme celui importé pour le silicium
n2_455 = 4.615
n2_730 = 3.752

h = np.arange(0, 300, 0.5)
I_455 = (np.cos(4 * np.pi * h * 0.000000001 * n1 / 0.000000455) + 1) / 2
I_730 = (np.cos(4 * np.pi * h * 0.000000001 * n1 / 0.000000730) + 1) / 2
plt.plot(h, I_455)
plt.plot(h, I_730)
plt.title('Intensités normalisées en 455 et 730 nm')
plt.xlabel('épaisseurs (nm)')
plt.ylabel('Intensité normalisée')
plt.show()

I_455_B = 2807 * (np.cos(4 * np.pi * h * 0.000000001 * n1 / 0.000000455) + 1) / 2 + 1150
I_730_B = 3133 * (np.cos(4 * np.pi * h * 0.000000001 * n1 / 0.000000730) + 1) / 2 + 657
plt.plot(h, I_455_B)
plt.plot(h, I_730_B)
plt.title('Intensités Brutes mesurées en 455 et 730 nm')
plt.xlabel('épaisseurs (nm)')
plt.ylabel('Intensité (niveaux de gris)')
plt.show()


def calcul_h(I, lambdaa):
    return lambdaa * np.arccos(2 * I - 1) / (4 * np.pi * n1)


epaisseur = calcul_h(0.95, 455)


def calcul_Delta_h(I, lambdaa, delta_I):
    h1 = calcul_h(I + 0.5 * delta_I, lambdaa)
    h2 = calcul_h(I - 0.5 * delta_I, lambdaa)
    delta_h = abs(h1 - h2)
    return delta_h


map_variation_h_455 = np.zeros([10, 10])
for i in range(len(map_variation_h_455)):
    for j in range(len(map_variation_h_455[0])):
        map_variation_h_455[len(map_variation_h_455) - i - 1, j] = calcul_Delta_h(0.05 + i * 0.1, 455,
                                                                                  0.111 / 10 + j * 0.111 / 10)

map_condition_I = np.zeros([10, 10])
for i in range(len(map_condition_I)):
    for j in range(len(map_condition_I[0])):
        map_condition_I[len(map_condition_I) - i - 1, j] = 0.05 + i * 0.1

map_condition_delta_I = np.zeros([10, 10])
for i in range(len(map_condition_delta_I)):
    for j in range(len(map_condition_delta_I[0])):
        map_condition_delta_I[len(map_condition_delta_I) - i - 1, j] = 0.111 / 10 + j * 0.111 / 10

map_variation_h_730 = np.zeros([10, 10])
for i in range(len(map_variation_h_730)):
    for j in range(len(map_variation_h_730[0])):
        map_variation_h_730[len(map_variation_h_730) - i - 1, j] = calcul_Delta_h(0.05 + i * 0.1, 730,
                                                                                  0.111 / 10 + j * 0.111 / 10)