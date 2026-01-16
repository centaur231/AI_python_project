"""
Projet: Réseau de Neurones Convolutif pour la Classification MNIST
École Nationale des Sciences Appliquées - Berrechid
Année Universitaire: 2025-2026
"""

import numpy as np
import os
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report

# ============================================================================
# 1. CHARGEMENT ET PRÉPARATION DES DONNÉES
# ============================================================================

def charger_et_preparer_donnees():
    """
    Charge le dataset MNIST et prépare les données pour l'entraînement
    """
    print("=" * 60)
    print("CHARGEMENT DES DONNÉES MNIST")
    print("=" * 60)
    
    # Charger les données MNIST
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    print(f"Forme des données d'entraînement: {x_train.shape}")
    print(f"Forme des étiquettes d'entraînement: {y_train.shape}")
    print(f"Forme des données de test: {x_test.shape}")
    print(f"Forme des étiquettes de test: {y_test.shape}")
    
    # Normalisation des images (valeurs entre 0 et 1)
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Redimensionner pour le CNN (ajouter dimension du canal)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    
    print(f"\nNouvelle forme après préparation: {x_train.shape}")
    
    # Conversion des étiquettes en format catégoriel (one-hot encoding)
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)
    
    print(f"Forme des étiquettes après encodage: {y_train_cat.shape}")
    
    return x_train, y_train, x_test, y_test, y_train_cat, y_test_cat


# ============================================================================
# 2. CONSTRUCTION DU RÉSEAU DE NEURONES CONVOLUTIF
# ============================================================================

def construire_modele_cnn():
    """
    Construction d'un réseau de neurones convolutif pour MNIST
    Architecture:
    - 2 couches de convolution + pooling
    - Couches fully connected
    - Dropout pour la régularisation
    """
    print("\n" + "=" * 60)
    print("CONSTRUCTION DU RÉSEAU DE NEURONES CONVOLUTIF")
    print("=" * 60)
    
    modele = keras.Sequential([
        # Première couche de convolution
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', 
                     input_shape=(28, 28, 1), name='conv1'),
        layers.MaxPooling2D(pool_size=(2, 2), name='pool1'),
        
        # Deuxième couche de convolution
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu', name='conv2'),
        layers.MaxPooling2D(pool_size=(2, 2), name='pool2'),
        
        # Troisième couche de convolution
        layers.Conv2D(128, kernel_size=(3, 3), activation='relu', name='conv3'),
        
        # Aplatissement
        layers.Flatten(name='flatten'),
        
        # Couches fully connected
        layers.Dense(128, activation='relu', name='dense1'),
        layers.Dropout(0.5, name='dropout1'),
        
        layers.Dense(64, activation='relu', name='dense2'),
        layers.Dropout(0.3, name='dropout2'),
        
        # Couche de sortie (10 classes)
        layers.Dense(10, activation='softmax', name='sortie')
    ])
    
    # Compilation du modèle
    modele.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nArchitecture du modèle:")
    modele.summary()
    
    return modele


# ============================================================================
# 3. ENTRAÎNEMENT DU MODÈLE
# ============================================================================

def entrainer_modele(modele, x_train, y_train_cat, x_test, y_test_cat):
    """
    Entraîne le modèle CNN sur les données MNIST
    """
    print("\n" + "=" * 60)
    print("ENTRAÎNEMENT DU MODÈLE")
    print("=" * 60)
    
    # Callbacks pour améliorer l'entraînement
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            verbose=1
        )
    ]
    
    # Entraînement
    historique = modele.fit(
        x_train, y_train_cat,
        batch_size=128,
        epochs=20,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\nEntraînement terminé!")
    
    return historique




# ============================================================================
# 4. ÉVALUATION DU MODÈLE
# ============================================================================

def evaluer_modele(modele, x_test, y_test, y_test_cat):
    """
    Évalue les performances du modèle sur l'ensemble de test
    """
    print("\n" + "=" * 60)
    print("ÉVALUATION DU MODÈLE SUR L'ENSEMBLE DE TEST")
    print("=" * 60)
    
    # Évaluation globale
    score = modele.evaluate(x_test, y_test_cat, verbose=0)
    print(f"\nPerte sur le test: {score[0]:.4f}")
    print(f"Précision sur le test: {score[1]*100:.2f}%")
    
    # Prédictions
    predictions = modele.predict(x_test, verbose=0)
    predictions_classes = np.argmax(predictions, axis=1)
    
    # Rapport de classification
    print("\n" + "-" * 60)
    print("RAPPORT DE CLASSIFICATION DÉTAILLÉ")
    print("-" * 60)
    print(classification_report(y_test, predictions_classes, 
                               target_names=[f'Chiffre {i}' for i in range(10)]))
    
    return predictions_classes






# ============================================================================
# 6. SAUVEGARDE DU MODÈLE
# ============================================================================

def sauvegarder_modele(modele, nom_fichier='modele_mnist_cnn.h5'):
    """
    Sauvegarde le modèle entraîné dans le même répertoire que le script
    """
    print("\n" + "=" * 60)
    print("SAUVEGARDE DU MODÈLE")
    print("=" * 60)
    
    # Obtenir le répertoire du script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, nom_fichier)
    
    modele.save(model_path)
    print(f"Modèle sauvegardé: {model_path}")


# ============================================================================
# 7. FONCTION PRINCIPALE
# ============================================================================

def main():
    """
    Fonction principale qui exécute tout le pipeline
    """
    print("\n" + "=" * 60)
    print("PROJET: CLASSIFICATION MNIST AVEC CNN")
    print("École Nationale des Sciences Appliquées - Berrechid")
    print("=" * 60)
    
    # 1. Charger et préparer les données
    x_train, y_train, x_test, y_test, y_train_cat, y_test_cat = charger_et_preparer_donnees()
    
    # 2. Construire le modèle
    modele = construire_modele_cnn()
    
    # 3. Entraîner le modèle
    historique = entrainer_modele(modele, x_train, y_train_cat, x_test, y_test_cat)
    
    # 4. Évaluer le modèle
    predictions = evaluer_modele(modele, x_test, y_test, y_test_cat)
    
    # 5. Sauvegarder le modèle
    sauvegarder_modele(modele)
    
    print("\n" + "=" * 60)
    print("ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
    print("=" * 60)
    print("\nModèle sauvegardé: modele_mnist_cnn.h5")
    print("\nPour générer les graphiques et rapports d'analyse, exécutez:")
    print("  python generer_rapport_analyse.py")


# ============================================================================
# EXÉCUTION DU PROGRAMME
# ============================================================================

if __name__ == "__main__":
    main()