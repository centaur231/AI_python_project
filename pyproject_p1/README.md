# Projet MNIST - Reconnaissance de Chiffres Manuscrits

**École Nationale des Sciences Appliquées - Berrechid**  
**Année Universitaire: 2025-2026**

## 📋 Description

Ce projet implémente un système de reconnaissance de chiffres manuscrits (0-9) en utilisant un réseau de neurones convolutif (CNN) entraîné sur le dataset MNIST. Le projet comprend un pipeline complet d'entraînement, d'évaluation, d'analyse et une interface graphique interactive.

## ✨ Fonctionnalités

- **Entraînement de modèle CNN** : Architecture optimisée pour la classification de chiffres
- **Évaluation complète** : Matrice de confusion, métriques de performance, analyse d'erreurs
- **Interface graphique interactive** : Dessinez des chiffres et obtenez des prédictions en temps réel
- **Génération de rapports** : Rapports d'analyse détaillés avec visualisations
- **Visualisations** : Graphiques de performance, distribution de confiance, exemples de prédictions

## 📁 Structure du Projet

```
pyproject_p1/
├── entrainement_mnist.py          # Script principal d'entraînement
├── interface_dessin.py             # Interface graphique interactive
├── generer_rapport_analyse.py     # Générateur de rapports d'analyse
├── modele_mnist_cnn.h5            # Modèle entraîné sauvegardé
└── rapport_analyse_modele/        # Dossier contenant les rapports
    ├── info_modele.json
    ├── info_modele.txt
    ├── matrice_confusion.png
    ├── rapport_classification.txt
    ├── rapport_classification.json
    ├── performance_par_classe.png
    ├── exemples_predictions.png
    ├── analyse_erreurs.png
    └── distribution_confiance.png
```

## 🔧 Dépendances

Les bibliothèques Python suivantes sont requises :

- `tensorflow` (ou `tensorflow-gpu`)
- `numpy`
- `matplotlib`
- `scikit-learn`
- `seaborn`
- `PIL` (Pillow)
- `scipy`
- `tkinter` (généralement inclus avec Python)

### Installation des dépendances

```bash
pip install tensorflow numpy matplotlib scikit-learn seaborn pillow scipy
```

## 🚀 Utilisation

### 1. Entraînement du modèle

Pour entraîner le modèle CNN sur le dataset MNIST :

```bash
python entrainement_mnist.py
```

Ce script effectue :
- Chargement et préparation des données MNIST
- Construction de l'architecture CNN
- Entraînement du modèle avec callbacks (EarlyStopping, ReduceLROnPlateau)
- Évaluation sur l'ensemble de test
- Visualisation des résultats
- Sauvegarde du modèle dans `modele_mnist_cnn.h5`

### 2. Interface graphique interactive

Pour utiliser l'interface de dessin :

```bash
python interface_dessin.py
```

**Instructions d'utilisation :**
- Cliquez et glissez pour dessiner un chiffre sur le canvas
- Cliquez sur **"Guess"** pour obtenir la prédiction
- Cliquez sur **"Clear"** pour effacer le canvas
- La prédiction et le score de confiance s'affichent en temps réel

### 3. Génération de rapports d'analyse

Pour générer un rapport complet d'analyse du modèle :

```bash
python generer_rapport_analyse.py
```

Ce script génère :
- Informations détaillées sur l'architecture du modèle
- Matrice de confusion (nombres et pourcentages)
- Rapport de classification avec métriques par classe
- Graphiques de performance (précision, rappel, F1-score)
- Visualisation d'exemples de prédictions
- Analyse des erreurs de classification
- Distribution de confiance des prédictions

Tous les fichiers sont sauvegardés dans le dossier `rapport_analyse_modele/`.

## 🏗️ Architecture du Modèle

Le modèle CNN est composé de :

- **3 couches convolutives** :
  - Conv2D(32 filtres, 3×3) + MaxPooling
  - Conv2D(64 filtres, 3×3) + MaxPooling
  - Conv2D(128 filtres, 3×3)

- **Couches fully connected** :
  - Dense(128) + Dropout(0.5)
  - Dense(64) + Dropout(0.3)
  - Dense(10) avec activation softmax (sortie)

**Paramètres totaux :** ~249,162 paramètres

**Optimiseur :** Adam  
**Fonction de perte :** Categorical Crossentropy  
**Métrique :** Accuracy

## 📊 Résultats

Le modèle atteint les performances suivantes sur l'ensemble de test :

- **Accuracy globale :** 99.33%
- **Précision moyenne (macro) :** 99.33%
- **Rappel moyen (macro) :** 99.32%
- **F1-score moyen (macro) :** 99.32%

### Performance par classe

| Chiffre | Précision | Rappel | F1-score |
|---------|-----------|--------|----------|
| 0       | 99%       | 100%   | 100%     |
| 1       | 100%      | 100%   | 100%     |
| 2       | 100%      | 100%   | 100%     |
| 3       | 99%       | 100%   | 99%      |
| 4       | 99%       | 99%    | 99%      |
| 5       | 99%       | 99%    | 99%      |
| 6       | 99%       | 99%    | 99%      |
| 7       | 99%       | 100%   | 99%      |
| 8       | 100%      | 99%    | 99%      |
| 9       | 99%       | 99%    | 99%      |

## 📝 Notes

- Le modèle est sauvegardé au format `.h5` (HDF5)
- Les visualisations sont générées en haute résolution (150 DPI)
- L'interface graphique utilise Tkinter pour la compatibilité multiplateforme
- Le prétraitement des images dessinées inclut la détection de bounding box, le centrage et la normalisation

## 👤 Auteur

Projet réalisé dans le cadre du cursus à l'École Nationale des Sciences Appliquées - Berrechid

## 📄 Licence

Ce projet est à des fins éducatives.
