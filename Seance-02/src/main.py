#coding:utf8

import pandas as pd
import matplotlib.pyplot as plt

# Source des données : https://www.data.gouv.fr/datasets/election-presidentielle-des-10-et-24-avril-2022-resultats-definitifs-du-1er-tour/
with open("./data/resultats-elections-presidentielles-2022-1er-tour.csv","r") as fichier:
    contenu = pd.read_csv(fichier)

# Mettre dans un commentaire le numéro de la question
# Question 1
# ...
print(contenu)
# Question 6
print("Nombre de lignes :", len(contenu))
print("Nombre de colonnes:", len(contenu.columns))
#Question 7
print(contenu.dtypes)
#Question 8
print("Nom des colonnes:")
print(contenu.head)
#Question 9
print("Colonne des inscrits:")
print(contenu.Inscrits)
#Question 10
#Création d'une liste vide
somme_colonnes = []
#Condition
for col in contenu.columns:
    if contenu[col].dtypes in ["int64" , "float64"]:
         total = contenu[col].sum()
         somme_colonnes.append((col, total))
#Résultat
print("Effectifs de chaque colonne:")
for col, total in somme_colonnes:
     print(f"-{col} : {total}")
#Question 11
import os
os.makedirs("images", exist_ok=True)
colonnes = ["Votants" , "Inscrits"]
for col in colonnes:
    plt.figure(figsize=(15,10))
    plt.bar(contenu["Libellé du département"], contenu[col], color="red")
    plt.xticks(rotation=90)
    plt.title(f"Nombre de {col} par département")
    plt.ylabel("Nombre d'électeurs")
    plt.xlabel("Départements")
    plt.tight_layout()
    plt.savefig(f"images/{col}.png", dpi=250)
    plt.close()
#Question 12
Création du dossier
import os
os.makedirs("./images_pie", exist_ok=True)
colonnes = ["Abstentions", "Blancs", "Nuls", "Exprimés"]
for idx, row in contenu.iterrows():
    valeurs = [row[col] for col in colonnes]
     labels = colonnes
     plt.pie(valeurs, labels=labels, autopct='%1.1f%%', startangle=90)
     plt.title(f"Répartition des votes - {row['Libellé du département']}")
     plt.savefig(f"images_pie/{row['Code du département']}_{row['Libellé du département']}.png", dpi=300)
     plt.close()
print("Diagrammes circulaires")
#Question 13
import os
os.makedirs("images_histogramme", exist_ok=True)
plt.hist(contenu["Inscrits"], bins=25, color="#C8A2C8", edgecolor="black", density=True)
plt.title("Histogramme de la distribution des Inscrits")
plt.xlabel("Nombre d'inscrits par département")
plt.ylabel("Densité (aire totale=1)")
plt.tight_layout()
plt.savefig("images_histogramme/histogramme_inscrits.png", dpi=250)
plt.close()
print("Histogramme")












