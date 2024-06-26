import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Chemin vers le fichier Excel
chemin_fichier_excel = "C:/Users/Elisa/OneDrive/Bureau/M2 Stage Tinnitus/testaudiometrieboxplot.xlsx"
# Charger le fichier Excel dans un DataFrame pandas
df = pd.read_excel(chemin_fichier_excel)
# Afficher les premières lignes du DataFrame pour vérification
print(df.head())

# Création d'un DataFrame à partir des données
columns = ["125", "250", "500", "1000", "1500", "2000", "3000", "4000", "6000", "8000"]
# Convertir les colonnes d'intensités en nombres
for col in columns[0:]:
    df[col] = pd.to_numeric(df[col], errors='coerce')
# Calcul de la moyenne des fréquences pour chaque ID
df['Mean_Frequency'] = df[['500', '1000', '2000', '4000']].mean(axis=1)
# Filtrer les lignes pour ne garder que celles avec une surdité subnormale
df_subnormal_hearing = df[df['Mean_Frequency'] < 25]
# Création du boxplot avec la couleur rouge pour les boîtes
plt.figure(figsize=(10, 6))
bp = df_subnormal_hearing[columns[0:]].boxplot(boxprops=dict(color='black'), flierprops=dict(markerfacecolor='black', markersize=4), widths=0.3)
# Tracer tous les points
for i, col in enumerate(columns, start=1):
    # Calcul de Q1 (premier quartile) et Q3 (troisième quartile)
    Q1 = df_subnormal_hearing[col].quantile(0.25)
    Q3 = df_subnormal_hearing[col].quantile(0.75)
    # Calcul de l'écart interquartile (IQR)
    IQR = Q3 - Q1
    # Définition des limites pour détecter les valeurs aberrantes
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Filtrer les données pour supprimer les valeurs aberrantes
    filtered_data = df_subnormal_hearing[(df_subnormal_hearing[col] >= lower_bound) & (df_subnormal_hearing[col] <= upper_bound)]
    # Calculer les nouveaux indices et les valeurs correspondantes après filtrage
    num_data_subnorm = len(filtered_data)
    subnorm_index = num_data_subnorm // 2  # Indice du milieu des données
    x_values = [i + (j - subnorm_index) * 0.02 for j in range(num_data_subnorm)]
    plt.scatter(x_values, filtered_data[col], color='grey', edgecolor='grey', s=10, alpha=0.9)
# Ajout des moyennes avec un point rouge
means = df_subnormal_hearing[columns[0:]].mean()
for i, mean in enumerate(means):
    plt.scatter(i + 1, mean, color='red', s=20, zorder=2)  # zorder pour placer les moyennes au-dessus des boîtes
plt.xlabel('Fréquence (Hz)')
plt.ylabel('Intensité (dB HL)')
plt.title('Boxplot de l\'intensité en fonction de la fréquence pour audition subnormale')
plt.ylim(-5, 101)
plt.grid(True)
plt.gca().invert_yaxis()
plt.show()

# Création d'un DataFrame à partir des données
columns = ["125", "250", "500", "1000", "1500", "2000", "3000", "4000", "6000", "8000"]
# Convertir les colonnes d'intensités en nombres
for col in columns[0:]:
    df[col] = pd.to_numeric(df[col], errors='coerce')
# Calcul de la moyenne des fréquences pour chaque ID
df['Mean_Frequency'] = df[['500', '1000', '2000', '4000']].mean(axis=1)
# Filtrer les lignes pour ne garder que celles avec une surdité légère entre 25 et 40 en intensité
df_mild_hearing_loss = df[(df['Mean_Frequency'] >= 25) & (df['Mean_Frequency'] < 40)]
# Création du boxplot avec la couleur bleue pour les boîtes
plt.figure(figsize=(10, 6))
bp = df_mild_hearing_loss[columns[0:]].boxplot(boxprops=dict(color='black'), flierprops=dict(marker='o', markerfacecolor='black', markersize=4), widths=0.3)
# Tracer tous les points
for i, col in enumerate(columns, start=1):
    # Calcul de Q1 (premier quartile) et Q3 (troisième quartile)
    Q1 = df_mild_hearing_loss[col].quantile(0.25)
    Q3 = df_mild_hearing_loss[col].quantile(0.75)
    # Calcul de l'écart interquartile (IQR)
    IQR = Q3 - Q1
    # Définition des limites pour détecter les valeurs aberrantes
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Filtrer les données pour supprimer les valeurs aberrantes
    filtered_data = df_mild_hearing_loss[(df_mild_hearing_loss[col] >= lower_bound) & (df_mild_hearing_loss[col] <= upper_bound)]
    # Calculer les nouveaux indices et les valeurs correspondantes après filtrage
    num_data_mild = len(filtered_data)
    mild_index = num_data_mild // 2  # Indice du milieu des données
    x_values = [i + (j - mild_index) * 0.02 for j in range(num_data_mild)]
    plt.scatter(x_values, filtered_data[col], color='grey', edgecolor='grey', s=10, alpha=0.9)
# Ajout des moyennes avec un point rouge
means = df_mild_hearing_loss[columns[0:]].mean()
for i, mean in enumerate(means):
    plt.scatter(i + 1, mean, color='red', s=20, zorder=2)  # zorder pour placer les moyennes au-dessus des boîtes
plt.xlabel('Fréquence (Hz)')
plt.ylabel('Intensité (dB HL)')
plt.title('Boxplot de l\'intensité en fonction de la fréquence pour perte auditive légère')
plt.ylim(-5, 101)
plt.grid(True)
plt.gca().invert_yaxis()
plt.show()


# Création d'un DataFrame à partir des données
columns = ["125", "250", "500", "1000", "1500", "2000", "3000", "4000", "6000", "8000"]
# Convertir les colonnes d'intensités en nombres
for col in columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')
# Calcul de la moyenne des fréquences pour chaque ID
df['Mean_Frequency'] = df[['500', '1000', '2000', '4000']].mean(axis=1)
# Filtrer les lignes pour ne garder que celles avec une surdité moyenne supérieure à 40 en intensité
df_higher_than_40 = df[df['Mean_Frequency'] > 40]
plt.figure(figsize=(10, 6))
bp = df_higher_than_40[columns].boxplot(boxprops=dict(color='black'), flierprops=dict(marker='o', markerfacecolor='black', markersize=4), widths=0.3)
# Tracer tous les points
for i, col in enumerate(columns, start=1):
    # Calcul de Q1 (premier quartile) et Q3 (troisième quartile)
    Q1 = df_higher_than_40[col].quantile(0.25)
    Q3 = df_higher_than_40[col].quantile(0.75)
    # Calcul de l'écart interquartile (IQR)
    IQR = Q3 - Q1
    # Définition des limites pour détecter les valeurs aberrantes
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Filtrer les données pour supprimer les valeurs aberrantes
    filtered_data = df_higher_than_40[(df_higher_than_40[col] >= lower_bound) & (df_higher_than_40[col] <= upper_bound)]
    # Calculer les nouveaux indices et les valeurs correspondantes après filtrage
    num_data_high = len(filtered_data)
    high_index = num_data_high // 2  # Indice du milieu des données
    x_values = [i + (j - high_index) * 0.02 for j in range(num_data_high)]
    plt.scatter(x_values, filtered_data[col], color='grey', edgecolor='grey', s=10, alpha=0.9)
# Ajout des moyennes avec un point rouge
means = df_higher_than_40[columns].mean()
for i, mean in enumerate(means, start=1):
    plt.scatter(i, mean, color='red', s=20, zorder=2)
plt.xlabel('Fréquence (Hz)')
plt.ylabel('Intensité (dB HL)')
plt.title('Boxplot de l\'intensité en fonction de la fréquence pour perte auditive moyenne')
plt.ylim(-5, 101)
plt.gca().invert_yaxis()
plt.grid(True)
plt.show()
