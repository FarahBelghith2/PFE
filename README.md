# 📈 PFE — Développement d’un système intelligent de gestion de promotions et de réservations de terrains sportifs

## 📌 Description du projet

Ce projet de fin d’études a pour objectif de développer un système intelligent capable de prédire la demande de réservation de terrains sportifs selon plusieurs facteurs : la date, l’heure, le type de terrain, la météo, les jours fériés, les vacances scolaires et les événements.

Le système permet d’aider l’administrateur à :

- anticiper les périodes de forte et de faible demande ;
- mieux gérer les créneaux horaires ;
- améliorer l’occupation des terrains ;
- identifier les moments où une promotion peut être utile ;
- proposer une décision marketing adaptée au niveau de demande prévu ;
- publier des codes promo ciblés selon le contexte détecté.

Ce projet s’inscrit dans le cadre d’un module intelligent intégré à une plateforme de gestion de réservations de terrains sportifs.

---

## 🎯 Objectifs du projet

Les objectifs principaux du projet sont :

- analyser les données historiques de réservation ;
- identifier les jours et les créneaux à forte ou faible demande ;
- exploiter des facteurs externes comme la météo, les jours fériés, les vacances et les événements ;
- construire et comparer plusieurs modèles de régression ;
- sélectionner le modèle le plus performant ;
- intégrer le modèle final dans une API Flask ;
- générer une recommandation promotionnelle ;
- permettre à l’administrateur de modifier, supprimer, publier ou exporter les prévisions ;
- générer automatiquement des codes promo après validation.

---

## 🤖 Modèles étudiés

Durant la phase de modélisation, plusieurs modèles de régression ont été étudiés :

- Random Forest Regressor ;
- Gradient Boosting Regressor ;
- XGBoost Regressor.

Le modèle **XGBoost** a été retenu comme modèle principal, car il a fourni les meilleures performances lors de l’évaluation.

---

## 🛠️ Technologies utilisées

### Langage principal

- Python

### Data Science

- Pandas
- NumPy
- Matplotlib
- Seaborn

### Machine Learning

- Scikit-learn
- XGBoost

### Back-end / API

- Flask
- Flask-CORS
- Requests

### Tests API

- Postman

### Environnement de développement

- Google Colab
- Jupyter Notebook
- Visual Studio Code

---

## 📂 Structure du projet

```text
PFE/
│
├── README.md
│   Documentation du projet
│
├── app.py
│   API Flask principale
│
├── features_utils.py
│   Fonctions utilitaires utilisées pour le feature engineering
│
├── xgboost_model.pkl
│   Modèle XGBoost final sauvegardé
│
├── Nettoyage_données.ipynb
│   Nettoyage initial des données
│
├── Préparation_des_données.ipynb
│   Préparation et enrichissement des données
│
├── Business_Understanding_Data_Understanding.ipynb
│   Compréhension métier et compréhension des données
│
├── XGBOOST2.ipynb
│   Entraînement, évaluation et sauvegarde du modèle XGBoost
│
└── requirements.txt
    Liste des dépendances Python nécessaires au projet
```

---

## ⚙️ Étapes principales du projet

Le projet suit les étapes suivantes :

1. Compréhension métier.
2. Collecte et préparation des données.
3. Nettoyage des données.
4. Analyse exploratoire.
5. Feature engineering temporel et contextuel.
6. Intégration des facteurs externes.
7. Entraînement de plusieurs modèles de régression.
8. Évaluation des modèles avec MAE, RMSE et R².
9. Sélection du modèle XGBoost.
10. Sauvegarde du modèle final.
11. Intégration du modèle dans une API Flask.
12. Génération de décisions marketing.
13. Publication de codes promo.

---

## 🚀 Installation et lancement du projet

### 1. Cloner le dépôt

```bash
git clone https://github.com/VOTRE_USERNAME/PFE.git
cd PFE
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Lancer l’API Flask

```bash
python app.py
```

L’API démarre par défaut sur :

```text
http://127.0.0.1:5000
```

---

## 📡 Routes API disponibles

### 1. Route principale de prédiction

```http
POST /predict
```

Cette route prédit la demande de réservation et retourne une décision marketing.

Exemple de body JSON :

```json
{
  "date": "2026-07-04",
  "heure_num": 18,
  "type_terrain": "outdoor",
  "adresse_site": "247 Rue du chêne-Brulé, 54700 Lesménils",
  "zone_scolaire": "B"
}
```

---

### 2. Route prédiction avec événement

```http
POST /predict/evenement
```

Cette route permet de tester principalement l’impact des événements sur la prévision et sur la recommandation promotionnelle.

Exemple :

```json
{
  "date": "2026-07-04",
  "heure_num": 18,
  "type_terrain": "outdoor"
}
```

---

### 3. Route prédiction métier

```http
POST /predict/metier
```

Cette route applique des règles métier sans utiliser directement le modèle IA.

Exemple :

```json
{
  "date": "2026-07-04",
  "heure_num": 18,
  "type_terrain": "outdoor",
  "adresse_site": "247 Rue du chêne-Brulé, 54700 Lesménils",
  "zone_scolaire": "B",
  "active_rules": ["1", "2", "3", "4", "5"]
}
```

Les règles métier disponibles sont :

```text
1 : Weekend
2 : Jour férié
3 : Météo outdoor
4 : Météo indoor
5 : Vacances / événements
```

---

### 4. Route météo

```http
POST /meteo
```

Retourne les informations météo associées à une date et une heure.

Exemple :

```json
{
  "date": "2026-07-04",
  "heure_num": 18
}
```

---

### 5. Route jours fériés

```http
POST /jour_ferie
```

Indique si une date correspond à un jour férié.

Exemple :

```json
{
  "date": "2026-07-14"
}
```

---

### 6. Route vacances scolaires

```http
POST /vacances
```

Indique si une date tombe pendant une période de vacances scolaires.

Exemple :

```json
{
  "date": "2026-07-04"
}
```

---

### 7. Route événement

```http
POST /evenement
```

Retourne les événements locaux ou sportifs détectés pour une date donnée.

Exemple :

```json
{
  "date": "2026-07-04"
}
```

---

## 🧠 Fonctionnement général du système

Le système fonctionne selon les étapes suivantes :

1. L’administrateur saisit une date, une heure et un type de terrain.
2. L’API récupère les informations externes :
   - météo ;
   - jours fériés ;
   - vacances scolaires ;
   - événements locaux ;
   - événements sportifs.
3. Les variables nécessaires au modèle sont construites automatiquement.
4. Le modèle XGBoost prédit la demande ou le taux de remplissage.
5. Le système interprète le résultat sous forme de niveau de demande.
6. Une décision marketing est générée.
7. Une promotion peut être proposée.
8. Si un événement ciblé est détecté, le système peut appliquer un ciblage marketing spécifique.
9. L’administrateur peut publier un code promo.

---

## 🧩 Modes de fonctionnement

### Mode Réel

Le mode réel utilise le modèle IA XGBoost et les facteurs externes pour générer une prévision.

Ce mode permet d’analyser l’impact de plusieurs filtres :

- Sans variables ;
- Vacances ;
- Météo ;
- Jour férié ;
- Événement.

### Mode Métier

Le mode métier applique des règles de décision prédéfinies.

Il est utile lorsque l’historique réel des réservations n’est pas encore suffisamment disponible ou lorsqu’on souhaite tester l’impact de règles spécifiques.

---

## 🎯 Ciblage promotionnel

Le système peut proposer des promotions ciblées selon le contexte détecté.

Exemples de ciblage :

- événement lié aux enfants ;
- fête des mères ;
- fête des pères ;
- journée des femmes ;
- vacances scolaires.

Exemple de réponse API :

```json
{
  "message_ciblage": "Promotion spéciale enfants",
  "promo_bonus_evenement": 15,
  "target_gender": null,
  "min_age": 2,
  "max_age": 10
}
```

Le ciblage peut ensuite être utilisé pour générer un code promo adapté, par exemple :

```text
ENFANTS-15P-1234
```

---

## 📊 Résultats attendus

Lors d’un appel à l’API, le système peut retourner :

- le nombre de réservations prévu ;
- le taux de remplissage prévu ;
- le niveau de demande ;
- la météo détectée ;
- les vacances scolaires ;
- les jours fériés ;
- les événements détectés ;
- la décision marketing ;
- la promotion recommandée ;
- le ciblage éventuel.

Exemple de résultat possible :

```json
{
  "date_demande": "2026-07-04",
  "heure_demande": "18.0h00",
  "type_terrain": "outdoor",
  "taux_remplissage_prevu": "52%",
  "niveau_demande": "Moyenne",
  "evenement_detecte": "Cité des enfants 2-6 ans",
  "decision_marketing": "15%",
  "message_ciblage": "Promotion spéciale enfants",
  "promo_bonus_evenement": 15,
  "min_age": 2,
  "max_age": 6,
  "target_gender": null
}
```

---

## 🔧 Configuration du projet

Le projet est configuré par défaut pour fonctionner avec un contexte français.

Les paramètres principaux se trouvent dans :

```text
app.py
```

---

## 🌍 Modifier la ville ou la localisation

Si vous souhaitez adapter le projet à une autre ville, modifiez les constantes suivantes dans `app.py` :

```python
LATITUDE = 48.8566
LONGITUDE = 2.3522
TIMEZONE = "Europe/Paris"
```

Exemple pour Lyon :

```python
LATITUDE = 45.7640
LONGITUDE = 4.8357
TIMEZONE = "Europe/Paris"
```

---

## 🗓️ Modifier la zone scolaire

Le projet utilise les zones scolaires françaises.

Dans `app.py`, modifiez :

```python
ZONE_SCOLAIRE = "B"
```

Valeurs possibles :

```text
A
B
C
```

---

## 🎉 Modifier les jours fériés

Le projet utilise les données françaises des jours fériés.

Dans `app.py`, modifiez :

```python
ZONE_FERIES = "metropole"
```

Exemples possibles :

```text
metropole
alsace-moselle
guadeloupe
martinique
guyane
reunion
mayotte
```

Si le projet est utilisé hors de France, il faut remplacer l’API des jours fériés par une source Open Data adaptée au pays concerné.

---

## 📍 Modifier le département ou la ville des événements

Pour adapter la détection des événements locaux, modifiez :

```python
DEPARTEMENT = "Paris"
VILLE = None
```

Exemple :

```python
DEPARTEMENT = "Gironde"
VILLE = "Bordeaux"
```

---

## 🔑 Modifier les clés API

Si vous utilisez une API externe nécessitant une clé, modifiez la variable correspondante.

Exemple :

```python
FOOTBALL_DATA_API_KEY = "VOTRE_CLE_API"
```

Il est recommandé de placer les clés API dans un fichier `.env` au lieu de les écrire directement dans le code.

---

## 🧩 Modifier les mots-clés des événements importants

Les événements sont analysés à partir de mots-clés.

Dans `app.py`, vous pouvez modifier :

```python
BIG_EVENT_KEYWORDS = [
    "concert",
    "festival",
    "spectacle",
    "match",
    "champions league",
    "coupe du monde",
    "enfant",
    "enfants",
    "enfance"
]
```

Ajoutez vos propres mots-clés selon le contexte local.

Exemple :

```python
BIG_EVENT_KEYWORDS.extend([
    "super bowl",
    "carnaval",
    "marathon",
    "foire",
    "salon"
])
```

---

## 🤖 Modifier ou remplacer le modèle IA

Le modèle utilisé par défaut est :

```text
xgboost_model.pkl
```

Si vous entraînez un nouveau modèle, remplacez ce fichier par votre nouveau modèle sauvegardé.

Le chargement du modèle se fait dans `app.py` :

```python
with open("xgboost_model.pkl", "rb") as f:
    model = pickle.load(f)
```

Si vous changez le nom du fichier, modifiez également cette ligne.

---

## 📁 Modifier les données d’entraînement

Les données sont préparées à travers les notebooks :

```text
Nettoyage_données.ipynb
Préparation_des_données.ipynb
XGBOOST2.ipynb
```

Si vous utilisez un nouveau dataset :

1. Vérifiez les noms des colonnes.
2. Nettoyez les valeurs manquantes.
3. Recréez les variables temporelles.
4. Ajoutez les facteurs externes.
5. Réentraînez le modèle.
6. Sauvegardez le nouveau modèle au format `.pkl`.
7. Remplacez `xgboost_model.pkl`.

---

## 🧪 Tester l’API avec Postman

### Exemple de test `/predict/evenement`

URL :

```http
POST http://127.0.0.1:5000/predict/evenement
```

Body :

```json
{
  "date": "2026-07-04",
  "heure_num": 18,
  "type_terrain": "outdoor"
}
```

Headers :

```text
Content-Type: application/json
```

---

## ✅ Avantages du système

Le système permet de :

- mieux anticiper la demande ;
- mieux gérer les créneaux horaires ;
- améliorer l’occupation des terrains ;
- automatiser partiellement la logique marketing ;
- aider l’administrateur à prendre des décisions plus rapides ;
- publier des promotions ciblées ;
- suivre les codes promo publiés.

---

## ⚠️ Limites actuelles

Le projet présente encore certaines limites :

- le modèle dépend fortement de la qualité des données historiques ;
- la météo future peut être indisponible pour des dates trop éloignées ;
- la détection des événements dépend des APIs externes ;
- certaines règles métier restent manuelles ;
- l’adaptation à une autre ville nécessite une configuration ;
- la généralisation à un autre pays nécessite de nouvelles sources de données ;
- les décisions marketing restent une aide à la décision et non une décision totalement automatique.

---

## 🔮 Perspectives d’amélioration

Plusieurs améliorations peuvent être envisagées :

- améliorer l’interface administrateur avec plus d’indicateurs visuels ;
- intégrer une base de données dynamique avec un historique réel plus volumineux ;
- ajouter de nouveaux facteurs d’analyse comme les prix concurrents ou les tendances saisonnières ;
- suivre l’efficacité réelle des codes promo publiés ;
- automatiser la mise à jour des données réelles ;
- optimiser davantage les hyperparamètres du modèle XGBoost ;
- tester d’autres modèles de machine learning.

---

## 👩‍💻 Auteur

Projet réalisé par :

```text
Farah Belghith
```

Dans le cadre d’un projet de fin d’études en Licence Informatique.

---

## 📜 Licence

Ce projet est réalisé dans un cadre académique.

