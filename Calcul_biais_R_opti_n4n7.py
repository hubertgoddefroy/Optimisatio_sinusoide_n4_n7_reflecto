# Lorsqu'on a lancé une série de reconstruction avec Reflecto Batch, sur une variation de n4 et n7, ce code permet de
# calculer pour chaque reconstruction les biais moyen, min et max sur toute l'acquisition par rapport à une référence RSB
# en permettant d'exclure certains puits, et calcul du R².

from tracer_animation_find_optimal_n4_n7 import R_carre
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

# MESURE DE REFERENCE AU RSB : expo SCExploPhy250410-30
mesures_RSB = np.asarray([12.3,11.7,121.7,13,11.4,10.9,123.9,10.1,33.6,42.8,57.2,72.1,122.5,120.4,121.1,122.8,123.8,122.4,119.3,116.3,125.8,114.3,104.3,86.5,74.8,62.5,127.6,49.4,37.9,11.8,130.3,8.7,6.9,5,5,3,94.7,128.3,109.1,117,130.4,124.2,128,114.5,132.2,131.7,133.1,132.7,10.8,10.9,126.4,10.6,9.6,8.7,134.6,7.7,39.6,55.1,72.2,84.5,121.6,121.7,120.9,121.3,128.1,125.4,125.4,124,135.3,125.4,116,98.3,76.7,62.9,126.7,51.9,37.3,29.5,129.9,3,3,3,3,3,90.5,126.2,105.3,112.8,128,123.7,122.7,125.3,126.6,123.8,124.9,126.9])

# LISTE DES PUITS A EXCLURE
exclusions = np.asarray(['A3','D8','F1','F10','H7','H9'])

# Seuil d'épaisseur mesurée au RSB en dessous duquel le puits doit être exclu (20 nm par défaut)
seuil_epaisseur_RSB = 20.0

for i in range(len(mesures_RSB)):
    mesures_RSB[i] = float(mesures_RSB[i])

path = 'F:\\optimisation_plate-type_Phy\\SCExploPhy250410-30\\weight_0_sans_restr'
liste_acquisitions = os.listdir(path)

liste_propre = []
for dossier in liste_acquisitions:
    if dossier[:5] == 'n4=1.':
        liste_propre.append(dossier)

biais_R_carre = [['condition testée',
                 'Moyenne biais absolu signé (nm)', 'Min biais absolu signé (nm)', 'Max biais absolu signé (nm)',
                 'Moyenne biais relatif signé (%)', 'Min biais relatif signé (%)', 'Max biais relatif signé (%)',
                 'Moyenne biais absolu absolu (nm)', 'Min biais absolu absolu (nm)', 'Max biais absolu absolu (nm)',
                 'Moyenne biais relatif absolu (%)', 'Min biais relatif absolu (%)', 'Max biais relatif absolu (%)',
                 'R²']]

for reconstruction in range(len(liste_propre)):
    print(liste_propre[reconstruction])
    liste_csv = []
    calcul_biais = []
    chemin_csv = path + '\\' + liste_propre[reconstruction] + '\\Synthese\\synthese_interferometric_data.csv'
    with open(chemin_csv, mode='r', newline='', encoding='utf-8') as fichier_csv:
        lecteur_csv = csv.reader(fichier_csv, delimiter=';')
        for ligne in lecteur_csv:
            liste_csv.append(ligne)
    for i in range(len(liste_csv) - 1):
        if not(liste_csv[i + 1][0] in exclusions) and (mesure_RSB[i] > seuil_epaisseur_RSB) and (liste_csv[i + 1][4] != 'nan'):
            biais_absolu_signe = float(mesures_RSB[i] - float(liste_csv[i + 1][4]))
            biais_relatif_signe = biais_absolu_signe*100/mesures_RSB[i]
            biais_absolu_absolu = abs(biais_absolu_signe)
            biais_relatif_absolu = biais_absolu_absolu*100/mesures_RSB[i]
            calcul_biais.append([float(liste_csv[i + 1][4]), biais_absolu_signe, biais_relatif_signe, biais_absolu_absolu, biais_relatif_absolu])
    moyenne_biais_absolu_signe = np.mean(calcul_biais[:,1])
    moyenne_biais_relatif_signe = np.mean(calcul_biais[:,2])
    moyenne_biais_absolu_absolu = np.mean(calcul_biais[:,3])
    moyenne_biais_relatif_absolu = np.mean(calcul_biais[:,4])
    min_biais_absolu_signe = np.min(calcul_biais[:,1])
    max_biais_absolu_signe = np.max(calcul_biais[:,1])
    min_biais_relatif_signe = np.min(calcul_biais[:,2])
    max_biais_relatif_signe = np.max(calcul_biais[:,2])
    min_biais_absolu_absolu = np.min(calcul_biais[:,3])
    max_biais_absolu_absolu = np.max(calcul_biais[:,3])
    min_biais_relatif_absolu = np.min(calcul_biais[:,4])
    max_biais_relatif_absolu = np.max(calcul_biais[:,4])
    r_carre = R_carre(mesures_RSB, calcul_biais[:,0])
    biais_R_carre.append([liste_propre[reconstruction],
                          moyenne_biais_absolu_signe, min_biais_absolu_signe, max_biais_absolu_signe,
                          moyenne_biais_relatif_signe, min_biais_relatif_signe, max_biais_relatif_signe,
                          moyenne_biais_absolu_absolu, min_biais_absolu_absolu, max_biais_absolu_absolu,
                          moyenne_biais_relatif_absolu, min_biais_relatif_absolu, max_biais_relatif_absolu,
                          r_carre])
