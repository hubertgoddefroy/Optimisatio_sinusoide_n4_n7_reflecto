from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.animation as animation
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import csv
import os

# plaque gamme large -11
# mesures_RSB = np.asarray([4,4,4,5,4,144,25.6,36.4,41.4,45.4,53.9,71.9,135.4,126.1,117.7,134.9,106.5,98.8,93.6,87.3,79.4,71.5,69.3,68,74.1,77.9,81.4,84.1,86.4,90.6,97.4,130.1,103,109.7,115.5,128.8,68.9,140.1,51.7,47.7,38.5,31.7,20.5,20,5,4,4,4,4,4,4,4,4,130.3,18.8,32.1,30.7,36.2,40.4,58.4,134,125.1,116.4,134.8,101.8,96.2,91.7,85.9,80.5,71.8,65.6,61,78.7,78.6,83.2,87.5,89,92.1,99.3,132,104.9,110.3,118.4,130.8,82.2,155.4,70.1,63.4,49.6,35.7,27.7,15.9,4,4,4,4])
# plaque gamme large -20
# mesures_RSB = np.asarray([4,4,4,17.3,25.8,150.9,30.9,37.4,42.4,54.6,55,71.5,138.2,129,119.9,139.9,111,104,100.2,93.1,87.4,82.4,77.4,64.3,75.8,74.7,80.9,86.1,91,97.9,104.5,136.3,108.6,116.1,122.8,133.5,70.3,142.2,55.9,56.4,45,40.6,34.2,22.6,21.6,4,4,4,4,4,10.3,15.8,15.6,137.4,32.3,36.3,39.2,45.9,53.5,66,138.8,128.6,118.7,140.9,110.7,103.8,94.8,89.3,83.9,75.6,68.8,64.6,79.1,77.2,81.4,84.6,92.9,96.5,104.4,141.2,110.3,117.5,125.2,134.3,84.2,158.7,53.3,54.2,46.5,33.2,31,23.4,14.5,4,4,4])
# plaque post -05
# mesures_RSB = np.asarray([154.9,147.4,144.9,144.5,143.6,143.3,144.1,142.3,142,142.7,142.3,149.4,143.3,135.5,133.1,132.2,131.1,130.5,132.1,130.6,130.2,130,130.5,139.2,142.9,134.3,131.9,130.5,130.3,130.1,131,129.5,129.1,129,129.7,137.7,143.7,134.6,132.5,131.1,131.2,130.7,131,130.1,129,129.7,130.6,138,143.9,134.6,133,130.8,130.7,131.6,132.2,130.2,129.3,129.4,130.5,138.2,144.3,135.5,133.6,131.3,130.9,131.7,132.2,131.5,130.4,130.1,130.5,138.3,146.4,137.6,134.9,132.8,133,133.6,134,134.1,133,131.7,132.5,138.7,157.3,150.5,148.2,147.2,147.3,147.5,146.1,145.9,147.1,145.9,145.3,151.7])
# plaque post -06
mesures_RSB = np.asarray(
    [152.2, 146, 145.7, 144.6, 144.2, 142.6, 146.2, 148.4, 146.7, 144.6, 145.3, 153.8, 142.5, 136.2, 134.5, 133.9,
     132.2, 131.6, 133.5, 136.1, 134.8, 132.4, 134.3, 143.2, 140.7, 135, 134.5, 134.1, 132, 130, 132.4, 135.1, 133.5,
     131.4, 132.6, 141.9, 140.6, 135.1, 134.5, 133.5, 132.3, 130.4, 131.4, 135, 133.6, 131.8, 132.9, 142.2, 140, 134.5,
     133, 133, 132.1, 130.5, 131.5, 134.5, 133.7, 131.8, 132.1, 142.2, 140.3, 135.1, 133, 133.3, 132.2, 130.9, 131,
     133.6, 133.6, 132.7, 132.5, 142.1, 142.2, 136.1, 134.2, 134.1, 133.7, 132.8, 131.9, 134.7, 134.3, 134.8, 135,
     143.5, 152, 146.3, 145.8, 146, 145.5, 144.7, 143.8, 147, 148.6, 148.4, 149, 156.1])

identite = [0, 50, 100, 150, 200]
for i in range(len(mesures_RSB)):
    mesures_RSB[i] = float(mesures_RSB[i])

path = 'F:\\optimisation_plate_type_BG_NF\\Cas_ZC_rfl_v1_SCBG250423-06_post_ZC22_19052025\\post'
liste_acquisitions = os.listdir(path)

liste_propre = []
for dossier in liste_acquisitions:
    if dossier[:5] == 'n4=1.':
        liste_propre.append(dossier)

liste_excel = []

for reconstruction in range(len(liste_propre)):
    print(liste_propre[reconstruction])
    liste_csv = []
    chemin_csv = path + '\\' + liste_propre[reconstruction] + '\\Synthese\\synthese_interferometric_data.csv'
    with open(chemin_csv, mode='r', newline='', encoding='utf-8') as fichier_csv:
        lecteur_csv = csv.reader(fichier_csv, delimiter=';')
        for ligne in lecteur_csv:
            liste_csv.append(ligne)

    for i in range(len(liste_csv) - 1):
        if liste_csv[i + 1][4] == 'nan':
            liste_excel.append([liste_propre[reconstruction], str(liste_propre[reconstruction][3:7]),
                                str(liste_propre[reconstruction][11:]), liste_csv[i + 1][0], np.nan])
        else:
            liste_excel.append([liste_propre[reconstruction], str(liste_propre[reconstruction][3:7]),
                                str(liste_propre[reconstruction][11:]), liste_csv[i + 1][0],
                                float(liste_csv[i + 1][4])])

np_liste_excel = np.asarray(liste_excel)

# # Configuration de la figure
fig, ax = plt.subplots()
ax.set_xlim(120, 170)
ax.set_ylim(120, 170)
scat = ax.scatter([], [], lw=2)

y_indentite = identite
ax.plot(identite, y_indentite, 'r-')


# Initialisation de la ligne
def init():
    scat.set_offsets(np.empty((0, 2)))
    return scat,


# Fonction d'animation
def animate(i):
    x = mesures_RSB
    y = np_liste_excel[i * 96:(i + 1) * 96, 4]
    x_trace = []
    y_trace = []
    for j in range(96):
        if np_liste_excel[i * 96 + j, 4] != np.nan:
            x_trace.append(mesures_RSB[j])
            y_trace.append(float(np_liste_excel[i * 96 + j, 4]))
    # y = 0.5*i*mesures_RSB
    # line.set_data(x_trace,y_trace)
    # return line,
    points = np.column_stack((x_trace, y_trace))
    scat.set_offsets(points)
    ax.set_title(np_liste_excel[i * 96, 0])
    return scat,


def RMS(liste1, liste2):
    liste1_sans_nan = []
    liste2_sans_nan = []
    # print(liste1)
    # print(liste2)
    for i in range(len(liste1)):
        # if liste2[i] != 'nan':
        # if isinstance(liste2[i], float):
        if not (np.isnan(liste2[i])):
            liste1_sans_nan.append(liste1[i])
            liste2_sans_nan.append(liste2[i])
            # print(type(liste2_sans_nan[0]))
    # print(liste2_sans_nan)
    MSE = mean_squared_error(liste1_sans_nan, liste2_sans_nan)
    RMSE = math.sqrt(MSE)
    return RMSE


def R_carre(liste1, liste2):
    liste1_sans_nan = []
    liste2_sans_nan = []
    for i in range(len(liste1)):
        if not (np.isnan(liste2[i])):
            liste1_sans_nan.append(liste1[i])
            liste2_sans_nan.append(liste2[i])
    liste1_sans_nan = np.asarray(liste1_sans_nan)
    liste2_sans_nan = np.asarray(liste2_sans_nan)
    print('liste1_sans_nan = \n', liste1_sans_nan)
    x = liste1_sans_nan.reshape(-1, 1)
    print('liste1_sans_nan.reshape = \n', x)
    model = LinearRegression()
    model.fit(x, liste2_sans_nan)
    y_pred = model.predict(x)
    r2 = r2_score(liste2_sans_nan, y_pred)
    return r2


# Création de l'animation
ani = animation.FuncAnimation(fig, animate, init_func=init, frames=108, interval=300, blit=True)

# Sauvegarde de l'animation en fichier GIF
ani.save('C:\\Users\\Admin\\Documents\\plaque_post_SCBG250423-06.gif', writer='pillow', fps=6)
plt.show()

liste_R_carre = []
liste_RMS = []
for j in range(int(len(liste_excel) / 96)):
    list_excel_float = []
    for h in range(96):
        list_excel_float.append(float(np_liste_excel[j * 96 + h, 4]))
    list_excel_float = np.asarray(list_excel_float)
    # print(list_excel_float)
    liste_R_carre.append([liste_excel[96 * j][0], R_carre(mesures_RSB, list_excel_float)])
    liste_RMS.append([liste_excel[96 * j][0], RMS(mesures_RSB, list_excel_float)])
liste_RMS = np.asarray(liste_RMS)
liste_R_carre = np.asarray(liste_R_carre)
print('®')
