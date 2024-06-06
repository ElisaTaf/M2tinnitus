import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro
import seaborn as sns
from scipy.stats import spearmanr
from scipy.stats import spearmanr, kruskal
from scipy.stats import wilcoxon
from scipy.stats import chi2_contingency

# Chemin vers le fichier Excel
chemin_fichier_excel = "C:/Users/Elisa/OneDrive/Bureau/M2 Stage Tinnitus/Tableau M2 Tinnitus Data.xlsx"
# Charger le fichier Excel dans un DataFrame pandas
df = pd.read_excel(chemin_fichier_excel)
# Afficher les premières lignes du DataFrame pour vérification
print(df.head())

# Calculer la moyenne d'âge
moyenne_age = df['Age'].mean()
print(f"La moyenne d'âge est : {moyenne_age}")

# Calculer la moyenne de la colonne "age"
moyenne_age = df['Age'].mean()
# Calculer l'écart type de la colonne "age"
ecart_type_age = df['Age'].std()
# Afficher les résultats
print(f"La moyenne de la colonne 'age' est : {moyenne_age}")
print(f"L'écart type de la colonne 'age' est : {ecart_type_age}")

# Sélectionner la colonne 'Age' et supprimer les valeurs manquantes
age_data = df['Age'].dropna()
# Effectuer le test de normalité de Shapiro-Wilk
statistic, p_value = shapiro(age_data)
# Afficher les résultats du test
print("Statistique de test de Shapiro-Wilk:", statistic)
print("Valeur p:", p_value)
# Interpréter les résultats du test
if p_value < 0.05:
    print("La distribution de l age n'est pas normale (rejet de l'hypothèse nulle).")
else:
    print("La distribution de l age est normale (l'hypothèse nulle n'est pas rejetée).")

# Sélectionner la colonne 'THIB' et supprimer les valeurs manquantes
thib_data = df['THIB'].dropna()
# Effectuer le test de normalité de Shapiro-Wilk
statistic, p_value = shapiro(thib_data)
# Afficher les résultats du test
print("Statistique de test de Shapiro-Wilk:", statistic)
print("Valeur p:", p_value)
# Interpréter les résultats du test
if p_value < 0.05:
    print("La distribution du THIB n'est pas normale (rejet de l'hypothèse nulle).")
else:
    print("La distribution du THIB est normale (l'hypothèse nulle n'est pas rejetée).")

# Sélectionner la colonne 'THIA 1 mois' et supprimer les valeurs manquantes
thia1mois_data = df['THIA 1 mois'].dropna()
# Effectuer le test de normalité de Shapiro-Wilk
statistic, p_value = shapiro(thia1mois_data)
# Afficher les résultats du test
print("Statistique de test de Shapiro-Wilk:", statistic)
print("Valeur p:", p_value)
# Interpréter les résultats du test
if p_value < 0.05:
    print("La distribution du THIA 1 mois n'est pas normale (rejet de l'hypothèse nulle).")
else:
    print("La distribution du THIA 1 mois est normale (l'hypothèse nulle n'est pas rejetée).")

# Sélectionner la colonne 'THIA 3 mois' et supprimer les valeurs manquantes
thia3mois_data = df['THIA 3 mois'].dropna()
# Effectuer le test de normalité de Shapiro-Wilk
statistic, p_value = shapiro(thia3mois_data)
# Afficher les résultats du test
print("Statistique de test de Shapiro-Wilk:", statistic)
print("Valeur p:", p_value)
# Interpréter les résultats du test
if p_value < 0.05:
    print("La distribution du THIA 3 mois n'est pas normale (rejet de l'hypothèse nulle).")
else:
    print("La distribution du THIA 3 mois est normale (l'hypothèse nulle n'est pas rejetée).")

# Sélectionner la colonne 'Amelioration THI' et supprimer les valeurs manquantes
amelthi_data = df['Amelioration THI'].dropna()
# Effectuer le test de normalité de Shapiro-Wilk
statistic, p_value = shapiro(amelthi_data)
# Afficher les résultats du test
print("Statistique de test de Shapiro-Wilk:", statistic)
print("Valeur p:", p_value)
# Interpréter les résultats du test
if p_value < 0.05:
    print("La distribution de l amelioration THI n'est pas normale (rejet de l'hypothèse nulle).")
else:
    print("La distribution de l amelioration THI est normale (l'hypothèse nulle n'est pas rejetée).")

# Sélectionner la colonne 'EVAB gêne' et supprimer les valeurs manquantes
evabg_data = df['EVAB gêne'].dropna()
# Effectuer le test de normalité de Shapiro-Wilk
statistic, p_value = shapiro(evabg_data)
# Afficher les résultats du test
print("Statistique de test de Shapiro-Wilk:", statistic)
print("Valeur p:", p_value)
# Interpréter les résultats du test
if p_value < 0.05:
    print("La distribution de l EVAB gêne n'est pas normale (rejet de l'hypothèse nulle).")
else:
    print("La distribution de l EVAB gêne est normale (l'hypothèse nulle n'est pas rejetée).")

# Sélectionner la colonne 'EVAA gêne 1 mois' et supprimer les valeurs manquantes
evag1mois_data = df['EVAA gêne 1 mois'].dropna()
# Effectuer le test de normalité de Shapiro-Wilk
statistic, p_value = shapiro(evag1mois_data)
# Afficher les résultats du test
print("Statistique de test de Shapiro-Wilk:", statistic)
print("Valeur p:", p_value)
# Interpréter les résultats du test
if p_value < 0.05:
    print("La distribution de l EVA gêne 1 mois n'est pas normale (rejet de l'hypothèse nulle).")
else:
    print("La distribution de l EVA gêne 1 mois est normale (l'hypothèse nulle n'est pas rejetée).")

# Sélectionner la colonne 'EVAA gêne 3 mois' et supprimer les valeurs manquantes
evag3mois_data = df['EVAA gêne 3 mois'].dropna()
# Effectuer le test de normalité de Shapiro-Wilk
statistic, p_value = shapiro(evag3mois_data)
# Afficher les résultats du test
print("Statistique de test de Shapiro-Wilk:", statistic)
print("Valeur p:", p_value)
# Interpréter les résultats du test
if p_value < 0.05:
    print("La distribution de l EVA gêne 3 mois n'est pas normale (rejet de l'hypothèse nulle).")
else:
    print("La distribution de l EVA gêne 3 mois est normale (l'hypothèse nulle n'est pas rejetée).")

# Sélectionner la colonne 'Amelioration EVA gêne' et supprimer les valeurs manquantes
amelevag_data = df['Amelioration EVA gêne'].dropna()
# Effectuer le test de normalité de Shapiro-Wilk
statistic, p_value = shapiro(amelevag_data)
# Afficher les résultats du test
print("Statistique de test de Shapiro-Wilk:", statistic)
print("Valeur p:", p_value)
# Interpréter les résultats du test
if p_value < 0.05:
    print("La distribution de l amelioration EVA gêne n'est pas normale (rejet de l'hypothèse nulle).")
else:
    print("La distribution de l amelioration EVA gêne est normale (l'hypothèse nulle n'est pas rejetée).")

# Sélectionner la colonne 'EVAB intensité' et supprimer les valeurs manquantes
evabi_data = df['EVAB intensité'].dropna()
# Effectuer le test de normalité de Shapiro-Wilk
statistic, p_value = shapiro(evabi_data)
# Afficher les résultats du test
print("Statistique de test de Shapiro-Wilk:", statistic)
print("Valeur p:", p_value)
# Interpréter les résultats du test
if p_value < 0.05:
    print("La distribution de l EVAB intensité n'est pas normale (rejet de l'hypothèse nulle).")
else:
    print("La distribution de l EVAB intensité est normale (l'hypothèse nulle n'est pas rejetée).")

# Sélectionner la colonne 'EVAA intensité 1 mois' et supprimer les valeurs manquantes
evai1mois_data = df['EVAA intensité 1 mois'].dropna()
# Effectuer le test de normalité de Shapiro-Wilk
statistic, p_value = shapiro(evai1mois_data)
# Afficher les résultats du test
print("Statistique de test de Shapiro-Wilk:", statistic)
print("Valeur p:", p_value)
# Interpréter les résultats du test
if p_value < 0.05:
    print("La distribution de l EVA intensité 1 mois n'est pas normale (rejet de l'hypothèse nulle).")
else:
    print("La distribution de l EVA intensité 1 mois est normale (l'hypothèse nulle n'est pas rejetée).")

# Sélectionner la colonne 'EVAA intensité 3 mois' et supprimer les valeurs manquantes
evai3mois_data = df['EVAA intensité 3 mois'].dropna()
# Effectuer le test de normalité de Shapiro-Wilk
statistic, p_value = shapiro(evai3mois_data)
# Afficher les résultats du test
print("Statistique de test de Shapiro-Wilk:", statistic)
print("Valeur p:", p_value)
# Interpréter les résultats du test
if p_value < 0.05:
    print("La distribution de l EVA intensité 3 mois n'est pas normale (rejet de l'hypothèse nulle).")
else:
    print("La distribution de l EVA intensité 3 mois est normale (l'hypothèse nulle n'est pas rejetée).")

# Sélectionner la colonne 'Amelioration EVA intensité' et supprimer les valeurs manquantes
amelevai_data = df['Amelioration EVA intensité'].dropna()
# Effectuer le test de normalité de Shapiro-Wilk
statistic, p_value = shapiro(amelevai_data)
# Afficher les résultats du test
print("Statistique de test de Shapiro-Wilk:", statistic)
print("Valeur p:", p_value)
# Interpréter les résultats du test
if p_value < 0.05:
    print("La distribution de l amelioration EVA intensité n'est pas normale (rejet de l'hypothèse nulle).")
else:
    print("La distribution de l amelioration EVA intensité est normale (l'hypothèse nulle n'est pas rejetée).")

# Sélectionner la colonne 'Durée de port (h/j)' et supprimer les valeurs manquantes
datalog_data = df['Durée de port (h/j)'].dropna()
# Effectuer le test de normalité de Shapiro-Wilk
statistic, p_value = shapiro(datalog_data)
# Afficher les résultats du test
print("Statistique de test de Shapiro-Wilk:", statistic)
print("Valeur p:", p_value)
# Interpréter les résultats du test
if p_value < 0.05:
    print("La distribution de la durée de port n'est pas normale (rejet de l'hypothèse nulle).")
else:
    print("La distribution de la durée de port est normale (l'hypothèse nulle n'est pas rejetée).")

# Sélectionner la colonne 'Durée depuis apparition (années)' et supprimer les valeurs manquantes
apparition_data = df['Durée depuis apparition (années)'].dropna()
# Effectuer le test de normalité de Shapiro-Wilk
statistic, p_value = shapiro(apparition_data)
# Afficher les résultats du test
print("Statistique de test de Shapiro-Wilk:", statistic)
print("Valeur p:", p_value)
# Interpréter les résultats du test
if p_value < 0.05:
    print("La distribution de la durée depuis apparition n'est pas normale (rejet de l'hypothèse nulle).")
else:
    print("La distribution de la durée depuis apparition est normale (l'hypothèse nulle n'est pas rejetée).")

# Sélectionner la colonne 'Pitch OD (Hz)' et supprimer les valeurs manquantes
pitchod_data = df['Pitch OD (Hz)'].dropna()
# Effectuer le test de normalité de Shapiro-Wilk
statistic, p_value = shapiro(pitchod_data)
# Afficher les résultats du test
print("Statistique de test de Shapiro-Wilk:", statistic)
print("Valeur p:", p_value)
# Interpréter les résultats du test
if p_value < 0.05:
    print("La distribution du Pitch OD (Hz) n'est pas normale (rejet de l'hypothèse nulle).")
else:
    print("La distribution du Pitch OD (Hz) est normale (l'hypothèse nulle n'est pas rejetée).")

# Sélectionner la colonne 'Pitch OG (Hz)' et supprimer les valeurs manquantes
pitchog_data = df['Pitch OG (Hz)'].dropna()
# Effectuer le test de normalité de Shapiro-Wilk
statistic, p_value = shapiro(pitchog_data)
# Afficher les résultats du test
print("Statistique de test de Shapiro-Wilk:", statistic)
print("Valeur p:", p_value)
# Interpréter les résultats du test
if p_value < 0.05:
    print("La distribution du Pitch OG (Hz) n'est pas normale (rejet de l'hypothèse nulle).")
else:
    print("La distribution du Pitch OG (Hz) est normale (l'hypothèse nulle n'est pas rejetée).")

# Compter le nombre de valeurs "M" et "F" dans la colonne "Sexe"
sexe_counts = df['Sexe'].value_counts()
print(sexe_counts)

# Compter le nombre de valeurs dans la colonne "Perte combi"
Pertecombi_counts = df['Perte combi'].value_counts()
print(Pertecombi_counts)

# Compter le nombre de valeurs dans la colonne "Uni/Bilat"
lateral_counts = df['Uni/Bilat'].value_counts()
print(lateral_counts)

# Compter le nombre de valeurs dans la colonne "Gamme ACA"
gamme_counts = df['Gamme ACA'].value_counts()
print(gamme_counts)

# Compter le nombre de valeurs dans la colonne "Audio"
audio_counts = df['Audio'].value_counts()
print(audio_counts)

# Compter le nombre de valeurs dans la colonne "TTT"
ttt_counts = df['TTT'].value_counts()
print(ttt_counts)

# Créer des catégories d'âge
jeunes = df[(df['Age'] >= 18) & (df['Age'] <= 34)]
age_moyen = df[(df['Age'] >= 35) & (df['Age'] <= 59)]
sujets_ages = df[df['Age'] >= 60]
# Nombre de personnes dans chaque catégorie
nb_jeunes = len(jeunes)
nb_age_moyen = len(age_moyen)
nb_sujets_ages = len(sujets_ages)
# Créer l'histogramme
plt.figure(figsize=(8, 6))
# Création des barres pour chaque catégorie
plt.bar('Sujets jeunes (18-34)', nb_jeunes, color='blue', label='Sujets jeunes (18-34)')
plt.bar('Sujets d\'âge moyen (35-59)', nb_age_moyen, color='green', label='Sujets d\'âge moyen (35-59)')
plt.bar('Sujets âgés (60+)', nb_sujets_ages, color='red', label='Sujets âgés (60+)')
plt.xlabel('Catégories d\'âge')
plt.ylabel('Nombre de sujets')
plt.yticks(range(0, 26, 5))
plt.legend()
plt.gca().yaxis.set_label_coords(-0.075, 0.5)  # Titre de l'axe des ordonnées
plt.gca().xaxis.set_label_coords(0.5, -0.085)  # Titre de l'axe des abscisses
plt.grid(True)
plt.show()

# Définir les intervalles et les étiquettes des catégories
intervalles = [18, 36, 56, 76, 100]
categories = ['Handicap léger', 'Handicap modéré', 'Handicap sévère', 'Handicap catastrophique']
# Ajouter une nouvelle colonne "Catégories THIB" avec les catégories correspondantes
df['Catégories THIB'] = pd.cut(df['THIB'], bins=intervalles, labels=categories, right=False)
# Grouper les données par catégorie et compter le nombre d'occurrences dans chaque catégorie
cat_counts = df['Catégories THIB'].value_counts()
cat_counts = cat_counts.reindex(categories)
# Créer un histogramme des catégories THIB
plt.figure(figsize=(10, 6))
plt.bar(cat_counts.index, cat_counts.values, color=['plum', 'lightskyblue', 'aquamarine', 'tomato'])
plt.xlabel('Catégories du THI')
plt.ylabel('Nombre de sujets')
plt.gca().yaxis.set_label_coords(-0.075, 0.5)  # Titre de l'axe des ordonnées
plt.gca().xaxis.set_label_coords(0.5, -0.085)  # Titre de l'axe des abscisses
plt.ylim(0, 20)  # Définir l'échelle de l'axe y à 5
plt.grid(True)
plt.show()

# Compter le nombre de valeurs dans la colone "Etiologie"
etiology_counts = df['Etiologie'].value_counts()
# Créer un histogramme
plt.bar(etiology_counts.index, etiology_counts.values)
# Ajouter des labels et un titre
plt.xlabel('Etiologie')
plt.ylabel('Nombre de sujets')
# Faire pivoter les étiquettes de l'axe des abscisses
plt.xticks(rotation=45, ha='right')
# Afficher l'histogramme
plt.grid(True)
plt.tight_layout()  # Ajuster automatiquement la disposition pour éviter les coupures
plt.gca().yaxis.set_label_coords(-0.075, 0.5)  # Titre de l'axe des ordonnées
plt.show()

df = pd.read_excel(chemin_fichier_excel)
# Définir les intervalles et les catégories correspondantes
intervalles = [18, 36, 56, 76, 100]
categories = ['Handicap léger', 'Handicap modéré', 'Handicap sévère', 'Handicap catastrophique']
# Créer une nouvelle colonne "Catégorie THIB" en fonction des intervalles
df['Catégorie THIB'] = pd.cut(df['THIB'], bins=intervalles, labels=categories, right=False)
# Groupement des données par catégorie THIB et comptage du nombre de sujets dans chaque catégorie
etiology_counts_grouped = df.groupby(['Etiologie', 'Catégorie THIB']).size().unstack(fill_value=0)
# Création du diagramme en bâtons empilés
etiology_counts_grouped.plot(kind='bar', stacked=True, figsize=(10, 6), color=['violet', 'skyblue', 'lightgreen', 'salmon'], width=0.8)
# Ajout des labels et du titre
plt.xlabel('Etiologies')
plt.ylabel('Nombre de sujets')
plt.title('Plotbar des étiologies en fonction des catégories du THI avant traitement sonore par nombre de sujets')
# Rotation des étiquettes sur l'axe des abscisses
plt.xticks(rotation=45, ha='right')
# Affichage de la légende et du graphique
plt.legend(title='Catégories THI')
plt.grid(True)
plt.tight_layout()
plt.show()

# Tracer l'histogramme de densité avec des barres plus larges et de couleur skyblue
sns.histplot(data=df, x='Amelioration THI', bins=100, kde=True, color='darkblue', binwidth=2)
# Ajouter des labels et un titre
plt.xlabel("Amélioration du THI")
plt.ylabel("Nombre de sujets")
# Afficher le graphique
plt.show()

# Tracer l'histogramme de densité avec des barres plus larges et de couleur skyblue
sns.histplot(data=df, x='Amelioration EVA gêne', bins=len(df['Amelioration EVA gêne'].unique()), kde=True, color='skyblue', binwidth=1)
# Ajouter des labels et un titre
plt.xlabel("Amélioration de l\'EVA gêne")
plt.ylabel("Nombre de sujets")
# Afficher le graphique
plt.show()

#Faire un graph de l'evolution du THI en fonction de chaque sujet
thib_avant_traitement = df['THIB']
thia_apres_traitement = df['THIA 1 mois']
# Créer le graphique
plt.figure(figsize=(8, 6))
# Boucle sur chaque sujet pour tracer ses données avant et après traitement
for i, (a, b) in enumerate(zip(thib_avant_traitement, thia_apres_traitement)):
    plt.plot([1, 2], [a, b], marker='o')
# Ajouter des étiquettes et un titre
plt.xticks([1, 2], ['THI avant traitement', 'THIA à 1 mois'])
plt.ylabel('Score')
plt.title('Évolution avant et après traitement sonore pour chaque sujet')
plt.ylim(0, 100)
# Afficher la légende et le graphique
plt.legend()
plt.grid(True)
plt.show()

#Faire un graph de l'evolution du THI en fonction de chaque sujet
thib_avant_traitement = df['THIB']
thia_apres_traitement = df['THIA 3 mois']
# Créer le graphique
plt.figure(figsize=(8, 6))
# Boucle sur chaque sujet pour tracer ses données avant et après traitement
for i, (a, b) in enumerate(zip(thib_avant_traitement, thia_apres_traitement)):
    plt.plot([1, 2], [a, b], marker='o')
# Ajouter des étiquettes et un titre
plt.xticks([1, 2], ['THI avant traitement', 'THIA à 3 mois'])
plt.ylabel('Score')
plt.title('Évolution avant et après traitement sonore pour chaque sujet')
plt.ylim(0, 100)
# Afficher la légende et le graphique
plt.legend()
plt.grid(True)
plt.show()

# Compter le nombre de valeurs dans la colone "TTT"
ttt_counts = df['TTT'].value_counts()
# Créer un histogramme
plt.bar(ttt_counts.index, ttt_counts.values, color=['lightsalmon', 'skyblue', 'limegreen'])
# Ajouter des labels et un titre
plt.xlabel('Traitements sonores')
plt.ylabel('Nombre de sujets')
# Afficher l'histogramme
plt.grid(True)
plt.tight_layout()  # Ajuster automatiquement la disposition pour éviter les coupures
plt.gca().yaxis.set_label_coords(-0.075, 0.5)  # Titre de l'axe des ordonnées
plt.gca().xaxis.set_label_coords(0.075, -0.085)  # Titre de l'axe des abscisses
plt.show()

# Compter le nombre de valeurs dans la colone "Gamme ACA"
gammeaca_counts = df['Gamme ACA'].value_counts()

df = pd.read_excel(chemin_fichier_excel)
# Filtrer les données pour ne conserver que les sujets ayant une "Perte auditive" légère
df_subnorm = df[df['Perte combi'] == 'Subnormale']
# Combiner les colonnes "Pitch OD (en Hz)" et "Pitch OG (en Hz)" en une seule colonne
pitch_data = pd.concat([df_subnorm['Pitch OD (Hz)'], df_subnorm['Pitch OG (Hz)']]).dropna()
# Créer un histogramme de la fréquence des acouphènes
plt.figure(figsize=(10, 6))
plt.hist(pitch_data, bins=30, color='navy', edgecolor='black')
# Ajouter des labels et un titre
plt.xlabel('Fréquence des acouphènes (en Hz)')
plt.ylabel('Nombre de sujets avec une audition normale')
plt.xlim(250, 8000)
plt.xticks([250, 500, 1000, 1500, 2000, 3000, 4000, 6000, 8000],
           labels=["250", "500", "1000", "1500", "2000", "3000", "4000", "6000", "8000"])
plt.show()

# Filtrer les données pour ne conserver que les sujets ayant une "Perte auditive" légère
df_legere = df[df['Perte combi'] == 'Légère']
# Combiner les colonnes "Pitch OD (en Hz)" et "Pitch OG (en Hz)" en une seule colonne
pitch_data = pd.concat([df_legere['Pitch OD (Hz)'], df_legere['Pitch OG (Hz)']]).dropna()
# Créer un histogramme de la fréquence des acouphènes
plt.figure(figsize=(10, 6))
plt.hist(pitch_data, bins=30, color='navy', edgecolor='black')
# Ajouter des labels et un titre
plt.xlabel('Fréquence des acouphènes (en Hz)')
plt.ylabel('Nombre de sujets avec une surdité légère')
plt.xlim(250, 8000)
plt.xticks([250, 500, 1000, 1500, 2000, 3000, 4000, 6000, 8000],
           labels=["250", "500", "1000", "1500", "2000", "3000", "4000", "6000", "8000"])
plt.show()

# Filtrer les données pour ne conserver que les sujets ayant une "Perte auditive" légère
df_moyenne = df[df['Perte combi'] == 'Moyenne']
# Combiner les colonnes "Pitch OD (en Hz)" et "Pitch OG (en Hz)" en une seule colonne
pitch_data = pd.concat([df_moyenne['Pitch OD (Hz)'], df_moyenne['Pitch OG (Hz)']]).dropna()
# Créer un histogramme de la fréquence des acouphènes
plt.figure(figsize=(10, 6))
plt.hist(pitch_data, bins=30, color='navy', edgecolor='black')
# Ajouter des labels et un titre
plt.xlabel('Fréquence des acouphènes (en Hz)')
plt.ylabel('Nombre de sujets avec une surdité moyenne')
plt.xlim(250, 8000)
plt.xticks([250, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000],
           labels=["250", "500", "1000", "1500", "2000", "3000", "4000", "5000", "6000", "7000", "8000"])
plt.show()

# Filtrer les données pour ne conserver que les sujets ayant une "Perte auditive" moyenne
df_moyenne = df[df['Perte combi'] == 'Moyenne']
# Combiner les colonnes "Pitch OD (en Hz)" et "Pitch OG (en Hz)" en une seule colonne
pitch_data = pd.concat([df_moyenne['Pitch OD (Hz)'], df_moyenne['Pitch OG (Hz)']]).dropna()
# Définir les sous-catégories de fréquences
categories = {
    'Bas médium': (200, 500),
    'Médium': (500, 1500),
    'Haut médium': (1500, 3000),
    'Aigu': (3000, 8000)
}
# Créer un histogramme pour chaque sous-catégorie
plt.figure(figsize=(10, 6))
# Initialiser une liste pour stocker les données catégorisées
categorized_data = []
for category, (low, high) in categories.items():
    subset = pitch_data[(pitch_data >= low) & (pitch_data < high)]
    categorized_data.append(subset)
# Tracer les histogrammes avec 4 bins pour chaque catégorie
plt.hist(categorized_data, bins=4, edgecolor='black', label=list(categories.keys()))
# Ajouter des labels et un titre
plt.xlabel('Fréquence des acouphènes (en Hz)')
plt.ylabel('Nombre de sujets avec une surdité moyenne')
plt.xticks([350, 1000, 2250, 5500],
           labels=["Bas médium", "Médium", "Haut médium", "Aigu"])
plt.legend()
plt.title('Histogramme des fréquences des acouphènes par catégories de surdité')
plt.show()