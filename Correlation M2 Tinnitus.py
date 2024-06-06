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

# Sélectionner les colonnes 'Age' et 'THIB'
age_thib_df = df[['Age', 'THIB']].dropna()
# Calculer le coefficient de corrélation
correlation_spearman, p_value_spearman = spearmanr(age_thib_df['Age'], age_thib_df['THIB'])
# Créer un graphique en nuage de points
plt.scatter(df['Age'], df['THIB'])
# Ajouter des labels et un titre
plt.xlabel('Âge')
plt.ylabel('Score du THI avant traitement sonore')
plt.title(f'Âge des sujets en fonction du THI avant traitement (corrélation : {correlation_spearman:.2f})')
# Données
xage = df['Age']
yTHIB = df['THIB']
# Calcul de la droite de régression linéaire
coefficients = np.polyfit(xage, yTHIB, 1)
polynomial = np.poly1d(coefficients)
# Tracer la droite de régression linéaire
plt.plot(xage, polynomial(xage), color='black', label=f'Droite de régression linéaire: y = {coefficients[0]:.2f}x + {coefficients[1]:.2f}')
# Calcul du coefficient de détermination (R²)
y_pred = polynomial(xage)
y_mean = np.mean(yTHIB)
SST = np.sum((yTHIB - y_mean) ** 2)
SSR = np.sum((y_pred - y_mean) ** 2)
R_squared = SSR / SST
# Affichage du R carré
print("Coefficient de détermination (R²) :", R_squared)
# Affichage de l'équation complète de la droite de régression linéaire
print("Equation de la droite de régression linéaire :", polynomial)
# Ajouter des labels et un titre
plt.xlabel('Âge du sujet')
plt.ylabel('THI avant traitement')
plt.title(f'Corrélation Spearman : {correlation_spearman:.2f}, R² : {R_squared:.2f}')
plt.legend()
plt.show()

# Calculer le coefficient de corrélation de Spearman et la valeur p associée
correlation_spearman, p_value_spearman = spearmanr(age_thib_df['Age'], age_thib_df['THIB'])
# Afficher le coefficient de corrélation de Spearman et la valeur p
print("Coefficient de corrélation de Spearman entre l'âge et le THIB:", correlation_spearman)
print("Valeur p:", p_value_spearman)

# Supprimer les lignes avec des valeurs manquantes dans les colonnes d'intérêt
dfthiblateralite = df.dropna(subset=['THIB', 'Latéralité'])
# Calculer le coefficient de corrélation de Spearman
correlation, p_value = spearmanr(dfthiblateralite['THIB'], dfthiblateralite['Latéralité'])
print(f"Coefficient de corrélation de Spearman entre THIB et latéralité de l'acouphène: {correlation}")
print(f"P-value : {p_value}")

# Effectuer le test de Kruskal-Wallis pour voir si les différences entre les catégories sont significatives
result = kruskal(*[dfthiblateralite[dfthiblateralite['Latéralité'] == category]['THIB'] for category in dfthiblateralite['Latéralité'].unique()])
print("Test de Kruskal-Wallis :")
print("Statistique de test :", result.statistic)
print("P-value :", result.pvalue)
# Supprimer les lignes avec des valeurs manquantes dans les colonnes d'intérêt
dfthiblateralite = df.dropna(subset=['THIB', 'Latéralité'])
# Calculer le coefficient de corrélation de Spearman
correlation_spearman, p_value = spearmanr(dfthiblateralite['THIB'], dfthiblateralite['Latéralité'])
print(f"Coefficient de corrélation de Spearman THIB/Latéralité: {correlation_spearman}")
print(f"P-value THIB/Latéralité: {p_value}")

# Supprimer les lignes avec des valeurs manquantes dans les colonnes d'intérêt
df1mois = df.dropna(subset=['THIB', 'THIA 1 mois'])
# Calculer la différence entre THIB et THIA
df1mois['Diff_THIB_THIA1mois'] = df1mois['THIB'] - df1mois['THIA 1 mois']
# Calculer la moyenne de la différence
diff_mean = df1mois['Diff_THIB_THIA1mois'].mean()
print(f"Différence moyenne entre THIB et THIA 1 mois : {diff_mean}")

# Supprimer les lignes avec des valeurs manquantes dans les colonnes d'intérêt
df1mois = df.dropna(subset=['THIB', 'THIA 1 mois'])
# Calculer la différence entre THIB et THIA
df1mois['Diff_THIB_THIA1mois'] = df1mois['THIB'] - df1mois['THIA 1 mois']
# Calculer la moyenne de la différence
diff_mean = df1mois['Diff_THIB_THIA1mois'].mean()
print(f"Différence moyenne entre THIB et THIA 1 mois : {diff_mean}")
# Effectuer un test de Wilcoxon pour échantillons appariés
wilcoxon_statistic, p_value_wilcoxon = wilcoxon(df1mois['THIB'], df1mois['THIA 1 mois'])
print("Test de Wilcoxon pour échantillons appariés :")
print("Statistique de test :", wilcoxon_statistic)
print("P-value :", p_value_wilcoxon)
# Vérifier si la différence est significative (avec un niveau de confiance de 95%)
alpha = 0.05
if p_value_wilcoxon < alpha:
    print("La différence entre THIB et THIA 1 mois est statistiquement significative.")
else:
    print("La différence entre THIB et THIA 1 mois n'est pas statistiquement significative.")

# Supprimer les lignes avec des valeurs manquantes dans les colonnes d'intérêt
df3mois = df.dropna(subset=['THIB', 'THIA 3 mois'])
# Calculer la différence entre THIB et THIA
df3mois['Diff_THIB_THIA3mois'] = df3mois['THIB'] - df3mois['THIA 3 mois']
# Calculer la moyenne de la différence
diff_meanB3 = df3mois['Diff_THIB_THIA3mois'].mean()
print(f"Différence moyenne entre THIB et THIA 3 mois : {diff_meanB3}")
# Effectuer un test de Wilcoxon pour échantillons appariés
wilcoxon_statistic, p_value_wilcoxon = wilcoxon(df3mois['THIB'], df3mois['THIA 3 mois'])
print("Test de Wilcoxon pour échantillons appariés :")
print("Statistique de test :", wilcoxon_statistic)
print("P-value :", p_value_wilcoxon)
# Vérifier si la différence est significative (avec un niveau de confiance de 95%)
alpha = 0.05
if p_value_wilcoxon < alpha:
    print("La différence entre THIB et THIA 3 mois est statistiquement significative.")
else:
    print("La différence entre THIB et THIA 3 mois n'est pas statistiquement significative.")

# Supprimer les lignes avec des valeurs manquantes dans les colonnes d'intérêt
df1mois = df.dropna(subset=['THIB', 'THIA 1 mois'])
# Calculer la différence entre THIB et THIA
df1mois['Diff_THIB_THIA1mois'] = df1mois['THIB'] - df1mois['THIA 1 mois']
# Calculer la différence moyenne entre THIB et THIA
diff_mean = df1mois['Diff_THIB_THIA1mois'].mean()
print(f"Différence moyenne entre THIB et THIA 1 mois : {diff_mean}")
# Effectuer un test de Wilcoxon pour échantillons appariés
wilcoxon_statistic, p_value_wilcoxon = wilcoxon(df1mois['THIB'], df1mois['THIA 1 mois'])
print("Test de Wilcoxon pour échantillons appariés :")
print("Statistique de test :", wilcoxon_statistic)
print("P-value :", p_value_wilcoxon)
# Vérifier si la différence est significative (avec un niveau de confiance de 95%)
alpha = 0.05
if p_value_wilcoxon < alpha:
    print("La différence entre THIB et THIA 1 mois est statistiquement significative.")
else:
    print("La différence entre THIB et THIA 1 mois n'est pas statistiquement significative.")

# Supprimer les lignes avec des valeurs manquantes dans les colonnes d'intérêt
df3mois = df.dropna(subset=['THIB', 'THIA 3 mois'])
# Calculer la différence entre THIB et THIA
df3mois['Diff_THIB_THIA3mois'] = df3mois['THIB'] - df3mois['THIA 3 mois']
# Calculer la différence moyenne entre THIB et THIA
diff_mean = df3mois['Diff_THIB_THIA3mois'].mean()
print(f"Différence moyenne entre THIB et THIA 3 mois : {diff_mean}")
# Effectuer un test de Wilcoxon pour échantillons appariés
wilcoxon_statistic, p_value_wilcoxon = wilcoxon(df3mois['THIB'], df3mois['THIA 3 mois'])
print("Test de Wilcoxon pour échantillons appariés :")
print("Statistique de test :", wilcoxon_statistic)
print("P-value :", p_value_wilcoxon)
# Vérifier si la différence est significative (avec un niveau de confiance de 95%)
alpha = 0.05
if p_value_wilcoxon < alpha:
    print("La différence entre THIB et THIA 3 mois est statistiquement significative.")
else:
    print("La différence entre THIB et THIA 3 mois n'est pas statistiquement significative.")

# Tracer les boxplots pour THIB, THIA 1 mois et THIA 3 mois
dfthibs = df.dropna(subset=['THIB'])['THIB']
dfthi1mois = df.dropna(subset=['THIA 1 mois'])['THIA 1 mois']
dfthi3mois = df.dropna(subset=['THIA 3 mois'])['THIA 3 mois']
plt.boxplot([dfthibs, dfthi1mois, dfthi3mois], tick_labels=['THIB', 'THIA 1 mois', 'THIA 3 mois'])
plt.ylabel('Scores')
# Calculer les moyennes pour THIB, THIA 1 mois et THIA 3 mois
mean_THIB = df['THIB'].mean()
mean_THIA_1_month = df['THIA 1 mois'].mean()
mean_THIA_3_month = df['THIA 3 mois'].mean()
# Placer les points des moyennes en noir dans les boxplots
plt.scatter(1, mean_THIB, color='black', zorder=2)  # THIB
plt.scatter(2, mean_THIA_1_month, color='black', zorder=2)  # THIA 1 mois
plt.scatter(3, mean_THIA_3_month, color='black', zorder=2)  # THIA 3 mois
plt.grid(True)
plt.gca().yaxis.set_label_coords(-0.075, 0.5)  # Titre de l'axe des ordonnées
plt.gca().xaxis.set_label_coords(0.075, -0.085)  # Titre de l'axe des abscisses
plt.show()

# Extraire les colonnes THIN, THI 1 mois et THI 3 mois
thib = df['THIB']
thi_1_mois = df['THIA 1 mois']
thi_3_mois = df['THIA 3 mois']
data_clean = df[['THIB', 'THIA 1 mois', 'THIA 3 mois']].dropna()
# Vérifier les valeurs manquantes
print(f"Valeurs manquantes dans THIB: {thib.isna().sum()}")
print(f"Valeurs manquantes dans THIA 1 mois: {thi_1_mois.isna().sum()}")
print(f"Valeurs manquantes dans THIA 3 mois: {thi_3_mois.isna().sum()}")
# Extraire les colonnes après suppression des valeurs manquantes
thib_clean = data_clean['THIB']
thi_1_mois_clean = data_clean['THIA 1 mois']
thi_3_mois_clean = data_clean['THIA 3 mois']
# Effectuer le test de Levene
stat, p_value = levene(thib_clean, thi_1_mois_clean, thi_3_mois_clean)
# Afficher les résultats
print(f"Statistique de Levene: {stat:.4f}")
print(f"p-value: {p_value:.4f}")

# Supprimer les lignes avec des valeurs manquantes dans les colonnes d'intérêt
dfeva1mois = df.dropna(subset=['EVAB gêne', 'EVAA gêne 1 mois'])
# Calculer la différence entre EVAB et EVAA
dfeva1mois['Diff_EVAB_EVAA1mois'] = dfeva1mois['EVAB gêne'] - dfeva1mois['EVAA gêne 1 mois']
# Calculer la différence moyenne
diff_mean = dfeva1mois['Diff_EVAB_EVAA1mois'].mean()
print(f"Différence moyenne entre EVAB et EVAA 1 mois : {diff_mean}")
# Effectuer un test de Wilcoxon pour échantillons appariés
wilcoxon_statistic, p_value_wilcoxon = wilcoxon(dfeva1mois['EVAB gêne'], dfeva1mois['EVAA gêne 1 mois'])
print("Test de Wilcoxon pour échantillons appariés :")
print("Statistique de test :", wilcoxon_statistic)
print("P-value :", p_value_wilcoxon)
# Vérifier si la différence est significative (avec un niveau de confiance de 95%)
alpha = 0.05
if p_value_wilcoxon < alpha:
    print("La différence entre EVAB gêne et EVAA 1 mois gêne est statistiquement significative.")
else:
    print("La différence entre EVAB gêne et EVAA 1 mois gêne n'est pas statistiquement significative.")

# Supprimer les lignes avec des valeurs manquantes dans les colonnes d'intérêt
dfeva3mois = df.dropna(subset=['EVAB gêne', 'EVAA gêne 3 mois'])
# Calculer la différence entre EVAB et EVAA
dfeva3mois['Diff_EVAB_EVAA3mois'] = dfeva3mois['EVAB gêne'] - dfeva3mois['EVAA gêne 3 mois']
# Calculer la différence moyenne
diff_mean = dfeva3mois['Diff_EVAB_EVAA3mois'].mean()
print(f"Différence moyenne entre EVAB et EVAA 3 mois : {diff_mean}")
# Effectuer un test de Wilcoxon pour échantillons appariés
wilcoxon_statistic, p_value_wilcoxon = wilcoxon(dfeva3mois['EVAB gêne'], dfeva3mois['EVAA gêne 3 mois'])
print("Test de Wilcoxon pour échantillons appariés :")
print("Statistique de test :", wilcoxon_statistic)
print("P-value :", p_value_wilcoxon)
# Vérifier si la différence est significative (avec un niveau de confiance de 95%)
alpha = 0.05
if p_value_wilcoxon < alpha:
    print("La différence entre EVAB gêne et EVAA 3 mois gêne est statistiquement significative.")
else:
    print("La différence entre EVAB gêne et EVAA 3 mois gêne n'est pas statistiquement significative.")

# Tracer les deux boxplots sur le même graphique
dfevag = df.dropna(subset=['EVAB gêne'])['EVAB gêne']
dfevag1mois = df.dropna(subset=['EVAA gêne 1 mois'])['EVAA gêne 1 mois']
dfevag3mois = df.dropna(subset=['EVAA gêne 3 mois'])['EVAA gêne 3 mois']
plt.boxplot([dfevag, dfevag1mois, dfevag3mois], tick_labels=['EVA gêne avant traitement', 'EVA gêne à 1 mois', 'EVA gêne à 3 mois'])
# Calculer les moyennes
mean_evab = df['EVAB gêne'].mean()
mean_eva1mois = df['EVAA gêne 1 mois'].mean()
mean_eva3mois = df['EVAA gêne 3 mois'].mean()
# Placer les moyennes en rouge dans les boxplots
plt.scatter(1, mean_evab, color='black', zorder=2)
plt.scatter(2, mean_eva1mois, color='black', zorder=2)
plt.scatter(3, mean_eva3mois, color='black', zorder=2)
plt.ylabel('Scores de l\'EVA gêne')
plt.xticks(ticks=[1, 2, 3], labels=['EVAB', 'EVAA gêne 1 mois', 'EVAA gêne 3 mois'])
plt.grid(True)
plt.show()

#THIB en fonction de la surdité Perte combi
plt.figure(figsize=(8, 6))
# Sélectionner les données pour chaque catégorie de perte auditive
subnormale_data = df[df['Perte combi'] == 'Subnormale']['THIB']
legere_data = df[df['Perte combi'] == 'Légère']['THIB']
moyenne_data = df[df['Perte combi'] == 'Moyenne']['THIB']
#Calculs moyennes
meansub= subnormale_data.mean()
meanleg= legere_data.mean()
meanmoy= moyenne_data.mean()
plt.scatter(1, meansub, color='black', zorder=2)
plt.scatter(2, meanleg, color='black', zorder=2)
plt.scatter(3, meanmoy, color='black', zorder=2)
# Créer le boxplot
plt.boxplot([subnormale_data, legere_data, moyenne_data], tick_labels=['Subnormale', 'Légère', 'Moyenne'])
plt.title('Boxplot du THIB pour chaque degré de surdité')
plt.xlabel('Catérogies de perte auditive')
plt.ylabel('Score du THI avant traitement')
plt.grid(True)
plt.gca().yaxis.set_label_coords(-0.075, 0.5)  # Titre de l'axe des ordonnées
plt.gca().xaxis.set_label_coords(0.5, -0.085)  # Titre de l'axe des abscisses
plt.show()

# Histogramme des degres de surdités OD (inutile pour le mémoire donc #)
# Compter le nombre d'occurrences de chaque degré de surdité
# dfperteOD = df.dropna(subset=['Perte auditive OD'])
# counts = dfperteOD['Perte auditive OD'].value_counts()
# Créer un histogramme
# counts.plot(kind='bar', color=['blue', 'green', 'red'], width=0.5)  # Ajuster la largeur des barres
# plt.title('Histogramme des Degrés de surdité OD')
# plt.xlabel('Degré de surdité')
# plt.ylabel('Nombre d\'occurrences')
# plt.ylim(0, 30)  # Définir l'échelle de l'axe y à 5
# plt.xticks(rotation=0)  # Rotation des étiquettes sur l'axe des abscisses
# plt.show()

# Histogramme des degres de surdités OG (inutile pour le mémoire donc #)
# Compter le nombre d'occurrences de chaque degré de surdité
# dfperteOG = df.dropna(subset=['Perte auditive OG'])
# counts = dfperteOG['Perte auditive OG'].value_counts()
# Créer un histogramme
# counts.plot(kind='bar', color=['blue', 'green', 'red'], width=0.5)  # Ajuster la largeur des barres
# plt.title('Histogramme des Degrés de surdité OG')
# plt.xlabel('Degré de surdité')
# plt.ylabel('Nombre d\'occurrences')
# plt.ylim(0, 30)  # Définir l'échelle de l'axe y à 5
# plt.xticks(rotation=0)  # Rotation des étiquettes sur l'axe des abscisses
# plt.show()

# Histogramme des degres de surdités OG et OD
# Compter le nombre d'occurrences de chaque degré de surdité pour OD
counts_od = df['Perte auditive OD'].value_counts()
# Compter le nombre d'occurrences de chaque degré de surdité pour OG
counts_og = df['Perte auditive OG'].value_counts()
# Créer un DataFrame pour stocker les deux séries de données
df_combined = pd.DataFrame({'droite': counts_od, 'gauche': counts_og})
# Tracer l'histogramme combiné
df_combined.plot(kind='bar', color=['red', 'blue'], width=0.4)
# Ajouter des labels et un titre
plt.xlabel('Catérogies de perte auditive')
plt.ylabel('Nombre de sujets')
plt.ylim(0, 25)  # Définir l'échelle de l'axe y à 30
plt.xticks(rotation=0)  # Rotation des étiquettes sur l'axe des abscisses
plt.legend(title='Oreille')  # Ajouter une légende avec le type de surdité (OD ou OG)
plt.grid(True)
plt.show()

# Supprimer les lignes avec des valeurs NaN dans les colonnes 'TTT' et 'Amelioration THI'
dfTTTamelnorm = df.dropna(subset=['TTT', 'Amelioration THI'])
# Séparer les données en groupes basés sur les catégories de la variable discrète
classe8 = dfTTTamelnorm[dfTTTamelnorm['TTT'] == 'correction auditive']['Amelioration THI']
classe9 = dfTTTamelnorm[dfTTTamelnorm['TTT'] == 'GBB + correction auditive']['Amelioration THI']
classe10 = dfTTTamelnorm[dfTTTamelnorm['TTT'] == 'GBB']['Amelioration THI']
# Effectuer le test de Kruskal-Wallis
test_kruskal, p_value_kruskal = kruskal(classe8, classe9, classe10)
# Afficher le résultat du test
print("Résultat du test de Kruskal-Wallis TTT  et amélioration THI ratio:")
print("Statistique de test :", test_kruskal)
print("P-value :", p_value_kruskal)
# Créer un boxplot pour chaque classe
plt.boxplot([classe8, classe9, classe10], tick_labels=['correction auditive', 'GBB + correction auditive', 'GBB'])
# Calculer les moyennes pour chaque classe
mean_classe8 = classe8.mean()
mean_classe9 = classe9.mean()
mean_classe10 = classe10.mean()
# Ajouter les moyennes en noir dans les boxplots
plt.scatter([1,2,3], [mean_classe8, mean_classe9, mean_classe10],color='black')
plt.xlabel('Traitements sonores')
plt.ylabel('Amelioration du THI')
plt.title('Boxplot de l amélioration du THI en ratio par classe de traitement')
plt.grid(True)
plt.show()

dfetiologieratio = df.dropna(subset=['Etiologie', 'Diff norm'])
# Séparer les données en groupes basés sur les catégories de la variable discrète
classe3ratio = dfetiologieratio[dfetiologieratio['Etiologie'] == 'Idiopatique']['Diff norm']
classe4ratio = dfetiologieratio[dfetiologieratio['Etiologie'] == 'Traumatisme sonore']['Diff norm']

dfetiologie = df.dropna(subset=['Etiologie', 'Amelioration THI'])
# Séparer les données en groupes basés sur les catégories de la variable discrète
classe3 = dfetiologie[dfetiologie['Etiologie'] == 'Idiopatique']['Amelioration THI']
classe4 = dfetiologie[dfetiologie['Etiologie'] == 'Traumatisme sonore']['Amelioration THI']

# Effectuer le test de Kruskal-Wallis ratio
test_kruskal, p_value_kruskal = kruskal(classe3ratio, classe4ratio)
# Afficher le résultat du test
print("Résultat du test de Kruskal-Wallis :")
print("Statistique de test :", test_kruskal)
print("P-value :", p_value_kruskal)

# Créer un boxplot pour chaque classe
plt.boxplot([classe3ratio, classe4ratio], tick_labels=['Idopatique','Traumatisme sonore'])
# Calculer les moyennes pour chaque classe
mean_classe3ratio = classe3ratio.mean()
mean_classe4ratio = classe4ratio.mean()
# Ajouter les moyennes en noir dans les boxplots
plt.scatter([1, 2], [mean_classe3ratio, mean_classe4ratio], color='black')
plt.xlabel('Etiologies')
plt.ylabel('Amelioration du THI')
plt.title('Analyse de le ratio de l amelioration THI par classe d étiologie')
plt.grid(True)
plt.show()

from scipy.stats import kruskal
# Effectuer le test de Kruskal-Wallis
test_kruskal, p_value_kruskal = kruskal(classe3, classe4)
# Afficher le résultat du test
print("Résultat du test de Kruskal-Wallis :")
print("Statistique de test :", test_kruskal)
print("P-value :", p_value_kruskal)

# Créer un boxplot pour chaque classe
plt.boxplot([classe3, classe4], tick_labels=['Idopatique','Traumatisme sonore'])
# Calculer les moyennes pour chaque classe
mean_classe3 = classe3.mean()
mean_classe4 = classe4.mean()
# Ajouter les moyennes en noir dans les boxplots
plt.scatter([1, 2], [mean_classe3, mean_classe4], color='black')
plt.xlabel('Etiologies')
plt.ylabel('Amelioration du THI')
plt.title('Analyse de l amelioration THI par classe d étiologie')
plt.grid(True)
plt.show()

# Supprimer les lignes avec des valeurs NaN dans les colonnes 'Gamme ACA' et 'Amelioration THI'
dfgammeamel = df.dropna(subset=['Gamme ACA', 'Amelioration THI'])
# Séparer les données en groupes basés sur les catégories de la variable discrète
classe1 = dfgammeamel[dfgammeamel['Gamme ACA'] == 'Classe 1']['Amelioration THI']
classe2 = dfgammeamel[dfgammeamel['Gamme ACA'] == 'Classe 2']['Amelioration THI']
# Effectuer le test de Kruskal-Wallis
test_kruskal, p_value_kruskal = kruskal(classe1, classe2)
# Afficher le résultat du test
print("Résultat du test de Kruskal-Wallis Gamme/Amel :")
print("Statistique de test :", test_kruskal)
print("P-value :", p_value_kruskal)

# Créer un boxplot pour chaque classe
plt.boxplot([classe1, classe2], tick_labels=['Classe 1','Classe 2'])
# Calculer les moyennes pour chaque classe
mean_classe1 = classe1.mean()
mean_classe2 = classe2.mean()
# Ajouter les moyennes en noir dans les boxplots
plt.scatter([1, 2], [mean_classe1, mean_classe2], color='black')
plt.xlabel('Gammes des appareils auditifs')
plt.ylabel('Amelioration THI')
plt.title('Analyse de l Amelioration THI par classe d''appareil')
plt.grid(True)
plt.show()

# Sélectionner les colonnes 'Amelioration THI' et 'THIB'
amelthi_thib_df = df[['THIB', 'Amelioration THI']].dropna()
# Calculer le coefficient de corrélation
correlation_spearman, p_value_spearman = spearmanr(amelthi_thib_df['THIB'], amelthi_thib_df['Amelioration THI'])
# Créer un graphique en nuage de points
plt.scatter(df['THIB'], df['Amelioration THI'])
# Données
xthib = df['THIB']
yamelthi = df['Amelioration THI']
# Calcul de la droite de régression linéaire
coefficients = np.polyfit(xthib, yamelthi, 1)
polynomial = np.poly1d(coefficients)
# Tracer la droite de régression linéaire
plt.plot(xthib, polynomial(xthib), color='black', label=f'Droite de régression linéaire: y = {coefficients[0]:.2f}x + {coefficients[1]:.2f}')
# Calcul du coefficient de détermination (R²)
y_pred = polynomial(xthib)
y_mean = np.mean(yamelthi)
SST = np.sum((yamelthi - y_mean) ** 2)
SSR = np.sum((y_pred - y_mean) ** 2)
R_squared = SSR / SST
# Affichage du R carré
print("Coefficient de détermination (R²) :", R_squared)
# Affichage de l'équation complète de la droite de régression linéaire
print("Equation de la droite de régression linéaire :", polynomial)
# Ajouter des labels et un titre
plt.xlabel('Score du THI avant traitement')
plt.ylabel('Amelioration du THI')
plt.title(f'Corrélation Spearman: {correlation_spearman:.2f}, R² : {R_squared:.2f}')
plt.legend()
plt.show()

# Sélectionner les colonnes 'Amelioration THI' et 'THIB'
amelthi_thib_df = df[['THIB', 'Amelioration THI']].dropna()
# Calculer le coefficient de corrélation de Pearson et sa valeur p
correlation_spearman, p_value_spearman = spearmanr(amelthi_thib_df['THIB'], amelthi_thib_df['Amelioration THI'])
# Afficher le coefficient de corrélation de Spearman et la valeur p
print("Coefficient de corrélation de Spearman THIB/Amel THI:", correlation_spearman)
print("Valeur p:", p_value_spearman)

# Supprimer les lignes avec des valeurs manquantes dans les colonnes "Perte auditive OD" et "Perte auditive OG"
dfperteODG = df.dropna(subset=['Perte auditive OD', 'Perte auditive OG'])
# Sélectionner les données pour chaque catégorie de perte auditive pour OD
subnormaleOD_data = dfperteODG[dfperteODG['Perte auditive OD'] == 'Subnormale']['Amelioration THI']
legereOD_data = dfperteODG[dfperteODG['Perte auditive OD'] == 'Légère']['Amelioration THI']
moyenneOD_data = dfperteODG[dfperteODG['Perte auditive OD'] == 'Moyenne']['Amelioration THI']
# Sélectionner les données pour chaque catégorie de perte auditive pour OG
subnormaleOG_data = dfperteODG[dfperteODG['Perte auditive OG'] == 'Subnormale']['Amelioration THI']
legereOG_data = dfperteODG[dfperteODG['Perte auditive OG'] == 'Légère']['Amelioration THI']
moyenneOG_data = dfperteODG[dfperteODG['Perte auditive OG'] == 'Moyenne']['Amelioration THI']
# Calculer les moyennes pour chaque catégorie de perte auditive pour OD
subnormaleOD_mean = subnormaleOD_data.mean()
legereOD_mean = legereOD_data.mean()
moyenneOD_mean = moyenneOD_data.mean()
# Calculer les moyennes pour chaque catégorie de perte auditive pour OG
subnormaleOG_mean = subnormaleOG_data.mean()
legereOG_mean = legereOG_data.mean()
moyenneOG_mean = moyenneOG_data.mean()

# Concaténer les données de 'Perte auditive OD' et 'Perte auditive OG' pour chaque catégorie de perte auditive
combined_subnormale_data = pd.concat([subnormaleOD_data, subnormaleOG_data])
combined_legere_data = pd.concat([legereOD_data, legereOG_data])
combined_moyenne_data = pd.concat([moyenneOD_data, moyenneOG_data])
# Calculer la moyenne combinée pour chaque catégorie de perte auditive
combined_subnormale_mean = combined_subnormale_data.mean()
combined_legere_mean = combined_legere_data.mean()
combined_moyenne_mean = combined_moyenne_data.mean()
# Créer un graphique
plt.figure(figsize=(8, 6))
# Ajouter les moyennes combinées avec des points noirs
plt.scatter([1, 3, 5], [combined_subnormale_mean, combined_legere_mean, combined_moyenne_mean], color='black')
# Tracer les boxplots combinés
plt.boxplot([combined_subnormale_data, combined_legere_data, combined_moyenne_data], positions=[1, 3, 5], widths=0.3, tick_labels=['Subnormale', 'Légère', 'Moyenne'], boxprops=dict(color='black'))
# Définir les étiquettes des axes
plt.xlabel('Degrés de surdité')
plt.ylabel('Amelioration du THI')
plt.title('Boxplot de l amelioration THI pour chaque degré de surdité')
plt.ylim(-5, 80)
plt.xlim(0, 6)
plt.grid(True)
# Afficher le graphique
plt.show()
# Effectuer le test de Kruskal-Wallis
test_kruskal, p_value_kruskal = kruskal(combined_subnormale_data, combined_legere_data, combined_moyenne_data)
# Afficher le résultat du test
print("Résultat du test de Kruskal-Wallis degrés de surdité et amélioration THI:")
print("Statistique de test :", test_kruskal)
print("P-value :", p_value_kruskal)

# Sélectionner les colonnes 'Amelioration THI' et 'Amelioration EVA gêne'
amelthi_eva_df = df[['Amelioration EVA gêne', 'Amelioration THI']].dropna()
# Calculer le coefficient de corrélation de Pearson et sa valeur p
correlation_spearman, p_value_spearman = spearmanr(amelthi_eva_df['Amelioration EVA gêne'], amelthi_eva_df['Amelioration THI'])
# Afficher le coefficient de corrélation de Spearman et la valeur p
print("Coefficient de corrélation de Spearman 'Amelioration  THI/Amelioration EVA gêne:", correlation_spearman)
print("Valeur p:", p_value_spearman)

# Calculer le coefficient de corrélation
correlation_spearman, p_value_spearman = spearmanr(amelthi_eva_df['Amelioration EVA gêne'], amelthi_eva_df['Amelioration THI'])
# Créer un graphique en nuage de points
plt.scatter(amelthi_eva_df['Amelioration EVA gêne'], amelthi_eva_df['Amelioration THI'])
# Données
xameleva = amelthi_eva_df['Amelioration EVA gêne']
yamelthi = amelthi_eva_df['Amelioration THI']
# Calcul de la droite de régression linéaire
coefficients = np.polyfit(xameleva, yamelthi, 1)
polynomial = np.poly1d(coefficients)
# Tracer la droite de régression linéaire
plt.plot(xameleva, polynomial(xameleva), color='black', label=f'Droite de régression linéaire: y = {coefficients[0]:.2f}x + {coefficients[1]:.2f}')
# Calcul du coefficient de détermination (R²)
y_pred = polynomial(xameleva)
y_mean = np.mean(yamelthi)
SST = np.sum((yamelthi - y_mean) ** 2)
SSR = np.sum((y_pred - y_mean) ** 2)
R_squared = SSR / SST
# Affichage du R carré
print("Coefficient de détermination (R²) :", R_squared)
# Affichage de l'équation complète de la droite de régression linéaire
print("Equation de la droite de régression linéaire .polynomial :", polynomial)
# Ajouter des labels et un titre
plt.xlabel('Amelioration de l\'EVA gêne')
plt.ylabel('Amelioration du THI')
plt.title(f'Corrélation Spearman : {correlation_spearman:.2f}, R² : {R_squared:.2f}')
plt.legend()
# Afficher le graphique
plt.show()

# Supprimer les lignes avec des valeurs NaN dans les colonnes 'Uni/Bilat' et 'Amelioration THI'
dflateralitevsamel = df.dropna(subset=['Uni/Bilat', 'Amelioration THI'])
# Séparer les données en groupes basés sur les catégories de la variable discrète
classe19 = dflateralitevsamel[dflateralitevsamel['Uni/Bilat'] == 'Unilatéral']['Amelioration THI']
classe20 = dflateralitevsamel[dflateralitevsamel['Uni/Bilat'] == 'Bilatéral']['Amelioration THI']
# Effectuer le test de Kruskal-Wallis
test_kruskal, p_value_kruskal = kruskal(classe19, classe20)
# Afficher le résultat du test
print("Résultat du test de Kruskal-Wallis latéralité  et amélioration THI:")
print("Statistique de test :", test_kruskal)
print("P-value :", p_value_kruskal)

# Créer un boxplot pour chaque classe
plt.boxplot([classe19, classe20], tick_labels=['Unilatéral', 'Bilatéral'])
# Calculer les moyennes pour chaque classe
mean_classe19 = classe19.mean()
mean_classe20 = classe20.mean()
# Ajouter les moyennes en noir dans les boxplots
plt.scatter([1,2], [mean_classe19, mean_classe20],color='black')
plt.xlabel('Latéralité de l\'acouphène')
plt.ylabel('Amelioration du THI')
plt.title('Boxplot de l amélioration du THI par latéralité')
plt.grid(True)
plt.show()

# Sélectionner les colonnes 'Amelioration THI' et 'Amelioration EVA gêne'
amelthi_dl_df = df[['Durée de port (h/j)', 'Amelioration THI']].dropna()
# Calculer le coefficient de corrélation de Pearson et sa valeur p
correlation_spearman, p_value_spearman = spearmanr(amelthi_dl_df['Durée de port (h/j)'], amelthi_dl_df['Amelioration THI'])
# Afficher le coefficient de corrélation de Spearman et la valeur p
print("Coefficient de corrélation de Spearman Datalog/Amelioration THI:", correlation_spearman)
print("Valeur p:", p_value_spearman)

# Nuage de point "Durée de port et amelioration THI"
# Calculer le coefficient de corrélation
correlation_spearman, p_value_spearman = spearmanr(amelthi_dl_df['Durée de port (h/j)'], amelthi_dl_df['Amelioration THI'])
# Créer un graphique en nuage de points
plt.scatter(amelthi_dl_df['Durée de port (h/j)'], amelthi_dl_df['Amelioration THI'])
# Données
xport = amelthi_dl_df['Durée de port (h/j)']
yamelthi = amelthi_dl_df['Amelioration THI']
# Calcul de la droite de régression linéaire
coefficients = np.polyfit(xport, yamelthi, 1)
polynomial = np.poly1d(coefficients)
# Calcul du coefficient de détermination (R²)
y_pred = polynomial(xport)
y_mean = np.mean(yamelthi)
SST = np.sum((yamelthi - y_mean) ** 2)
SSR = np.sum((y_pred - y_mean) ** 2)
R_squared = SSR / SST
# Affichage du R carré
print("Coefficient de détermination (R²) :", R_squared)
# Affichage de l'équation complète de la droite de régression linéaire
print("Equation de la droite de régression linéaire .polynomial :", polynomial)
# Tracer la droite de régression linéaire
plt.plot(xport, polynomial(xport), color='black', label=f'Droite de régression linéaire: y = {coefficients[0]:.2f}x + {coefficients[1]:.2f}')
# Ajouter des labels et un titre
plt.xlabel('Durée de port (h/j)')
plt.ylabel('Amelioration du THI')
plt.title(f'Corrélation Spearman: {correlation_spearman:.2f}, R² : {R_squared:.2f}')
plt.show()

