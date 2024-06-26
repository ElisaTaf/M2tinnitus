import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro
import seaborn as sns
from scipy.stats import spearmanr
from scipy.stats import spearmanr, kruskal
from scipy.stats import wilcoxon
from scipy.stats import chi2_contingency
import pingouin as pg
from scipy.stats import levene

# Chemin vers le fichier Excel
chemin_fichier_excel = "C:/Users/Elisa/OneDrive/Bureau/M2 Stage Tinnitus/Tableau M2 Tinnitus Data.xlsx"
# Charger le fichier Excel dans un DataFrame pandas
df = pd.read_excel(chemin_fichier_excel)
# Afficher les premières lignes du DataFrame pour vérification
print(df.head())

# Effectuer un test du Chi-2 entre les variables 'Latéralité' et 'TTT'
crosstab = pd.crosstab(df['Latéralité'], df['TTT'])
chi2, p_value, dof, expected = chi2_contingency(crosstab)
# Afficher les résultats du test
print(f"Chi-square statistic Latéralité/TTT: {chi2}")
print(f"P-value: {p_value}")
print(f"Degrees of freedom: {dof}")

# Créer une table de contingence entre les deux variables
crosstab = pd.crosstab(df['Latéralité'], df['TTT'])
print(crosstab)
# Tracer un histogramme avec les barres côte à côte
crosstab.plot(kind='bar', stacked=False, color=['blue', 'red', 'green'], position=0.5)
plt.xlabel('Latéralité')
plt.ylabel('Nombre de sujets')
plt.legend(title='Traitements sonores')
plt.grid(True)
plt.xticks(rotation=0)
plt.show()

# Effectuer un test du Chi-2 entre les variables 'Uni/Bilat' et 'TTT'
crosstab = pd.crosstab(df['Uni/Bilat'], df['TTT'])
chi2, p_value, dof, expected = chi2_contingency(crosstab)
# Afficher les résultats du test
print(f"Chi-square statistic Uni/Bilat/TTT: {chi2}")
print(f"P-value: {p_value}")
print(f"Degrees of freedom: {dof}")

# Effectuer un test du Chi-2 entre les variables 'Gamme ACA' et 'TTT'
crosstab = pd.crosstab(df['Gamme ACA'], df['TTT'])
chi2, p_value, dof, expected = chi2_contingency(crosstab)
# Afficher les résultats du test
print(f"Chi-square statistic Gamme ACA/TTT: {chi2}")
print(f"P-value: {p_value}")
print(f"Degrees of freedom: {dof}")

# Créer une table de contingence entre les deux variables
crosstab = pd.crosstab(df['Gamme ACA'], df['TTT'])
# Tracer un histogramme avec les barres côte à côte
crosstab.plot(kind='bar', stacked=False, color=['blue', 'red', 'green'], position=0.5)
plt.xlabel('Gammes des appareils auditifs')
plt.ylabel('Nombre de sujets')
plt.legend(title='Traitements sonores')
plt.xticks(rotation=0)
plt.grid(True)
plt.show()

# Effectuer un test du Chi-2 entre les variables 'Perte auditive' et 'TTT'
crosstab = pd.crosstab(df['Perte combi'], df['TTT'])
chi2, p_value, dof, expected = chi2_contingency(crosstab)
# Afficher les résultats du test
print(f"Chi-square statistic Perte combi/TTT: {chi2}")
print(f"P-value: {p_value}")
print(f"Degrees of freedom: {dof}")

# Créer une table de contingence entre les deux variables
crosstab = pd.crosstab(df['Perte combi'], df['TTT'])
print(crosstab)
# Tracer un histogramme avec les barres côte à côte
crosstab.plot(kind='bar', stacked=False, color=['blue', 'red', 'green'], position=0.5)
plt.xlabel('Perte auditive')
plt.ylabel('Nombre de sujets')
plt.legend(title='Traitements sonores')
plt.grid(True)
plt.xticks(rotation=0)
plt.show()