#coding:utf8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Source des données : https://www.data.gouv.fr/datasets/election-presidentielle-des-10-et-24-avril-2022-resultats-definitifs-du-1er-tour/

# Sources des données : production de M. Forriez, 2016-2023
#Etape 5
#Condition
import os
contenu = pd.read_csv(
    "data/resultats-elections-presidentielles-2022-1er-tour.csv",
    sep=None,
    engine="python",
    encoding="utf-8"
)

colonnes_quanti = ["Inscrits", "Votants", "Blancs", "Nuls", "Exprimés", "Abstentions"]
num = contenu[colonnes_quanti].copy()
#Calcul
moyennes = num.mean(axis=0).round(2)
medianes = num.median(axis=0).round(2)
modes = num.mode().iloc[0].round(2)
ecarts_type = num.std(ddof=0, axis=0).round(2)          # écart-type population[web:12]
ecarts_abs_moy = num.apply(lambda s: np.abs(s - s.mean()).mean()).round(2)
etendues = (num.max() - num.min()).round(2)
#Affichage
print("\nMoyennes :\n", moyennes)
print("\nMédianes :\n", medianes)
print("\nModes :\n", modes)
print("\nÉcarts type :\n", ecarts_type)
print("\nÉcarts absolus moyens :\n", ecarts_abs_moy)
print("\nÉtendues :\n", etendues)
# Étape 6 – Construire un tableau récapitulatif et l’afficher en colonnes
resume = pd.DataFrame({
    "Moyenne": moyennes,
    "Médiane": medianes,
    "Mode": modes,
    "Écart-type": ecarts_type,
    "Écart abs. moyen": ecarts_abs_moy,
    "Étendue": etendues
})

print("\n--- Paramètres (par colonne quantitative) ---\n")
print(resume.to_string())
#Etape 7
#Calcul des distances interquartile et interdécile 
Q1 = num.quantile(0.25)
Q3 = num.quantile(0.75)
dist_interquartile = (Q3 - Q1).round(2)

D1 = num.quantile(0.10)
D9 = num.quantile(0.90)
dist_interdecile = (D9 - D1).round(2)
#Affichage sous forme de colonnes
print("\n--- Distances interquartiles (par colonne) ---\n")
print(dist_interquartile.to_string())

print("\n--- Distances interdéciles (par colonne) ---\n")
print(dist_interdecile.to_string())
#Etape 8
#Créer le dossier img 
os.makedirs("img", exist_ok=True)

# Boucle sur chaque colonne quantitative
for col in num.columns:
    plt.figure(figsize=(4, 6))
    plt.boxplot(num[col].dropna(), vert=True)
    plt.title(f"Boîte à moustaches - {col}")
    plt.ylabel(col)

    # Enregistrer l'image dans le dossier img
    plt.savefig(f"img/boxplot_{col}.png", bbox_inches="tight")
    plt.close()
#Etape 10
#Lecture du fichier
df_island = pd.read_csv("data/island-index.csv", encoding="utf-8")
#Sélection de la colonne Surface (km2)
surfaces = df_island["Surface (km²)"]
#Définition des bornes de classes
bins = [0, 10, 25, 50, 100, 2500, 5000, 10000, float("inf")]
#Étiquettes des classes
labels = [
    "]0, 10]",
    "]10, 25]",
    "]25, 50]",
    "]50, 100]",
    "]100, 2500]",
    "]2500, 5000]",
    "]5000, 10000]",
    "]10000, +∞["
]
#Catégorisation des surfaces
classes = pd.cut(surfaces, bins=bins, labels=labels, right=True, include_lowest=False)
#Dénombrement des îles par classe
compte_classes = classes.value_counts().sort_index()

print("\nNombre d'îles par intervalle de surface :\n")
print(compte_classes.to_string())
