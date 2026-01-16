"""
Projet: Réseau de Neurones Convolutif pour la Classification MNIST
École Nationale des Sciences Appliquées - Berrechid
Année Universitaire: 2025-2026
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from scipy import ndimage

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


def visualiser_exemples(x_train, y_train, nombre=10):
    """
    Visualise quelques exemples du dataset
    """
    print("\n" + "=" * 60)
    print("VISUALISATION D'EXEMPLES DU DATASET")
    print("=" * 60)
    
    plt.figure(figsize=(15, 3))
    for i in range(nombre):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x_train[i].reshape(28, 28), cmap='gray')
        plt.title(f'Chiffre: {y_train[i]}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('exemples_mnist.png', dpi=150, bbox_inches='tight')
    print("Figure sauvegardée: exemples_mnist.png")
    plt.show()


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


def visualiser_historique(historique):
    """
    Visualise l'évolution de l'entraînement
    """
    print("\n" + "=" * 60)
    print("VISUALISATION DE L'HISTORIQUE D'ENTRAÎNEMENT")
    print("=" * 60)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Courbe de précision
    axes[0].plot(historique.history['accuracy'], label='Précision entraînement')
    axes[0].plot(historique.history['val_accuracy'], label='Précision validation')
    axes[0].set_title('Évolution de la Précision', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Époque')
    axes[0].set_ylabel('Précision')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Courbe de perte
    axes[1].plot(historique.history['loss'], label='Perte entraînement')
    axes[1].plot(historique.history['val_loss'], label='Perte validation')
    axes[1].set_title('Évolution de la Perte', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Époque')
    axes[1].set_ylabel('Perte')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('historique_entrainement.png', dpi=150, bbox_inches='tight')
    print("Figure sauvegardée: historique_entrainement.png")
    plt.show()


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
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, predictions_classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.title('Matrice de Confusion', fontsize=16, fontweight='bold')
    plt.xlabel('Prédiction')
    plt.ylabel('Vérité')
    plt.tight_layout()
    plt.savefig('matrice_confusion.png', dpi=150, bbox_inches='tight')
    print("\nMatrice de confusion sauvegardée: matrice_confusion.png")
    plt.show()
    
    return predictions_classes


def visualiser_predictions(x_test, y_test, predictions, nombre=20):
    """
    Visualise des exemples de prédictions
    """
    print("\n" + "=" * 60)
    print("VISUALISATION DES PRÉDICTIONS")
    print("=" * 60)
    
    plt.figure(figsize=(20, 8))
    
    for i in range(nombre):
        plt.subplot(4, 5, i + 1)
        plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
        
        # Couleur selon si la prédiction est correcte ou non
        couleur = 'green' if predictions[i] == y_test[i] else 'red'
        plt.title(f'Vrai: {y_test[i]}\nPrédit: {predictions[i]}', 
                 color=couleur, fontweight='bold')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('exemples_predictions.png', dpi=150, bbox_inches='tight')
    print("Figure sauvegardée: exemples_predictions.png")
    plt.show()


def analyser_erreurs(x_test, y_test, predictions, nombre=10):
    """
    Analyse les erreurs de classification
    """
    print("\n" + "=" * 60)
    print("ANALYSE DES ERREURS DE CLASSIFICATION")
    print("=" * 60)
    
    # Trouver les indices des erreurs
    erreurs = np.where(predictions != y_test)[0]
    print(f"\nNombre total d'erreurs: {len(erreurs)}")
    print(f"Taux d'erreur: {len(erreurs)/len(y_test)*100:.2f}%")
    
    if len(erreurs) > 0:
        plt.figure(figsize=(20, 4))
        
        for i, idx in enumerate(erreurs[:nombre]):
            plt.subplot(2, 5, i + 1)
            plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')
            plt.title(f'Vrai: {y_test[idx]}\nPrédit: {predictions[idx]}', 
                     color='red', fontweight='bold')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('analyse_erreurs.png', dpi=150, bbox_inches='tight')
        print("Figure sauvegardée: analyse_erreurs.png")
        plt.show()


# ============================================================================
# 5. TEST SUR DE NOUVELLES IMAGES
# ============================================================================

def predire_nouvelle_image(modele, x_test, index):
    """
    Prédit le chiffre pour une nouvelle image
    """
    print("\n" + "=" * 60)
    print("PRÉDICTION SUR UNE NOUVELLE IMAGE")
    print("=" * 60)
    
    # Sélectionner une image
    image = x_test[index:index+1]
    
    # Prédiction
    prediction = modele.predict(image, verbose=0)
    classe_predite = np.argmax(prediction[0])
    confiance = prediction[0][classe_predite] * 100
    
    # Affichage
    plt.figure(figsize=(12, 4))
    
    # Image
    plt.subplot(1, 3, 1)
    plt.imshow(image.reshape(28, 28), cmap='gray')
    plt.title(f'Image à classifier', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    # Probabilités
    plt.subplot(1, 3, 2)
    plt.bar(range(10), prediction[0])
    plt.xlabel('Chiffre')
    plt.ylabel('Probabilité')
    plt.title('Probabilités pour chaque classe', fontsize=12, fontweight='bold')
    plt.xticks(range(10))
    plt.grid(True, alpha=0.3, axis='y')
    
    # Résultat
    plt.subplot(1, 3, 3)
    plt.text(0.5, 0.5, f'Prédiction:\n\n{classe_predite}\n\nConfiance:\n{confiance:.2f}%',
             ha='center', va='center', fontsize=24, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('prediction_nouvelle_image.png', dpi=150, bbox_inches='tight')
    print(f"\nChiffre prédit: {classe_predite}")
    print(f"Confiance: {confiance:.2f}%")
    print("Figure sauvegardée: prediction_nouvelle_image.png")
    plt.show()
    
    return classe_predite, confiance


# ============================================================================
# 5.5 DESSINER ET RECONNAÎTRE UN CHIFFRE
# ============================================================================

def dessiner_et_predire(modele):
    """
    Interface interactive pour dessiner un chiffre et obtenir une prédiction
    """
    print("\n" + "=" * 60)
    print("MODE INTERACTIF: DESSINEZ VOTRE CHIFFRE")
    print("=" * 60)
    print("Instructions:")
    print("- Cliquez et glissez pour dessiner")
    print("- Appuyez sur 'C' pour effacer")
    print("- Appuyez sur 'P' pour prédire")
    print("- Fermez la fenêtre pour quitter")
    print("=" * 60)
    
    # Créer un canvas vierge
    canvas = np.zeros((280, 280))
    derniere_position = [None, None]
    
    def dessiner(event):
        """Fonction pour dessiner sur le canvas"""
        if event.xdata is not None and event.ydata is not None:
            x, y = int(event.xdata), int(event.ydata)
            # Dessiner un cercle épais
            for i in range(-8, 8):
                for j in range(-8, 8):
                    if 0 <= x+i < 280 and 0 <= y+j < 280:
                        if i*i + j*j <= 64:  # Rayon de 8 pixels
                            canvas[y+j, x+i] = 255
            
            # Mettre à jour l'affichage
            ax.clear()
            ax.imshow(canvas, cmap='gray', vmin=0, vmax=255)
            ax.set_title('Dessinez un chiffre (C=Effacer, P=Prédire)', 
                        fontsize=14, fontweight='bold')
            ax.axis('off')
            fig.canvas.draw()
    
    def on_key(event):
        """Gestion des touches clavier"""
        nonlocal canvas
        
        if event.key == 'c':  # Effacer
            canvas = np.zeros((280, 280))
            ax.clear()
            ax.imshow(canvas, cmap='gray', vmin=0, vmax=255)
            ax.set_title('Dessinez un chiffre (C=Effacer, P=Prédire)', 
                        fontsize=14, fontweight='bold')
            ax.axis('off')
            fig.canvas.draw()
            print("Canvas effacé!")
            
        elif event.key == 'p':  # Prédire
            print("\nTraitement de l'image...")
            
            # Redimensionner à 28x28
            from scipy import ndimage
            image_redim = ndimage.zoom(canvas, (28/280, 28/280))
            
            # Normaliser
            image_norm = image_redim / 255.0
            
            # Préparer pour le modèle
            image_input = image_norm.reshape(1, 28, 28, 1)
            
            # Prédiction
            prediction = modele.predict(image_input, verbose=0)
            classe_predite = np.argmax(prediction[0])
            confiance = prediction[0][classe_predite] * 100
            
            # Afficher le résultat
            print(f"\n{'='*50}")
            print(f"RÉSULTAT DE LA PRÉDICTION")
            print(f"{'='*50}")
            print(f"Chiffre reconnu: {classe_predite}")
            print(f"Confiance: {confiance:.2f}%")
            print(f"{'='*50}\n")
            
            # Créer une figure avec le résultat
            fig_result = plt.figure(figsize=(15, 5))
            
            # Dessin original
            ax1 = fig_result.add_subplot(1, 3, 1)
            ax1.imshow(canvas, cmap='gray')
            ax1.set_title('Votre dessin', fontsize=14, fontweight='bold')
            ax1.axis('off')
            
            # Image prétraitée
            ax2 = fig_result.add_subplot(1, 3, 2)
            ax2.imshow(image_redim, cmap='gray')
            ax2.set_title('Image prétraitée (28x28)', fontsize=14, fontweight='bold')
            ax2.axis('off')
            
            # Probabilités
            ax3 = fig_result.add_subplot(1, 3, 3)
            bars = ax3.bar(range(10), prediction[0], color=['green' if i == classe_predite else 'lightblue' for i in range(10)])
            ax3.set_xlabel('Chiffre', fontsize=12)
            ax3.set_ylabel('Probabilité', fontsize=12)
            ax3.set_title(f'Prédiction: {classe_predite} ({confiance:.1f}%)', 
                         fontsize=14, fontweight='bold')
            ax3.set_xticks(range(10))
            ax3.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(f'prediction_dessin_{classe_predite}.png', dpi=150, bbox_inches='tight')
            print(f"Résultat sauvegardé: prediction_dessin_{classe_predite}.png")
            plt.show()
    
    # Créer la figure
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(canvas, cmap='gray', vmin=0, vmax=255)
    ax.set_title('Dessinez un chiffre (C=Effacer, P=Prédire)', 
                fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Connecter les événements
    fig.canvas.mpl_connect('button_press_event', dessiner)
    fig.canvas.mpl_connect('motion_notify_event', 
                          lambda event: dessiner(event) if event.button else None)
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# 6. SAUVEGARDE DU MODÈLE
# ============================================================================

def sauvegarder_modele(modele, nom_fichier='modele_mnist_cnn.h5'):
    """
    Sauvegarde le modèle entraîné
    """
    print("\n" + "=" * 60)
    print("SAUVEGARDE DU MODÈLE")
    print("=" * 60)
    
    modele.save(nom_fichier)
    print(f"Modèle sauvegardé: {nom_fichier}")


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
    
    # 2. Visualiser des exemples
    visualiser_exemples(x_train, y_train)
    
    # 3. Construire le modèle
    modele = construire_modele_cnn()
    
    # 4. Entraîner le modèle
    historique = entrainer_modele(modele, x_train, y_train_cat, x_test, y_test_cat)
    
    # 5. Visualiser l'historique
    visualiser_historique(historique)
    
    # 6. Évaluer le modèle
    predictions = evaluer_modele(modele, x_test, y_test, y_test_cat)
    
    # 7. Visualiser les prédictions
    visualiser_predictions(x_test, y_test, predictions)
    
    # 8. Analyser les erreurs
    analyser_erreurs(x_test, y_test, predictions)
    
    # 9. Tester sur quelques nouvelles images
    for idx in [0, 100, 500]:
        predire_nouvelle_image(modele, x_test, idx)
    
    # 10. Sauvegarder le modèle
    sauvegarder_modele(modele)
    
    # 11. MODE INTERACTIF - Dessiner et prédire
    print("\n" + "=" * 60)
    print("LANCEMENT DU MODE INTERACTIF")
    print("=" * 60)
    reponse = input("\nVoulez-vous dessiner un chiffre et tester le modèle? (o/n): ")
    
    if reponse.lower() in ['o', 'oui', 'y', 'yes']:
        dessiner_et_predire(modele)
        
        # Possibilité de dessiner plusieurs fois
        while True:
            reponse = input("\nVoulez-vous dessiner un autre chiffre? (o/n): ")
            if reponse.lower() in ['o', 'oui', 'y', 'yes']:
                dessiner_et_predire(modele)
            else:
                break
    
    print("\n" + "=" * 60)
    print("PROJET TERMINÉ AVEC SUCCÈS!")
    print("=" * 60)
    print("\nFichiers générés:")
    print("- exemples_mnist.png")
    print("- historique_entrainement.png")
    print("- matrice_confusion.png")
    print("- exemples_predictions.png")
    print("- analyse_erreurs.png")
    print("- prediction_nouvelle_image.png")
    print("- prediction_dessin_X.png (vos dessins)")
    print("- modele_mnist_cnn.h5")


# ============================================================================
# EXÉCUTION DU PROGRAMME
# ============================================================================

if __name__ == "__main__":
    main()