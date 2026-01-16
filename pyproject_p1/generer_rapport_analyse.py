"""
Générateur de Rapport d'Analyse du Modèle MNIST
École Nationale des Sciences Appliquées - Berrechid
Année Universitaire: 2025-2026

Ce script génère des graphiques et statistiques complètes pour présenter le modèle
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from datetime import datetime
import json

# Créer le répertoire pour sauvegarder les résultats (dans le même répertoire que le script)
script_dir = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.join(script_dir, "rapport_analyse_modele")
os.makedirs(REPORT_DIR, exist_ok=True)

# Configuration matplotlib pour de meilleurs graphiques
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('dark_background')
sns.set_palette("husl")


def charger_modele_et_donnees():
    """Charge le modèle entraîné et les données de test"""
    print("\n" + "=" * 70)
    print("CHARGEMENT DU MODÈLE ET DES DONNÉES")
    print("=" * 70)
    
    # Obtenir le répertoire du script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'modele_mnist_cnn.h5')
    
    # Charger le modèle
    try:
        if not os.path.exists(model_path):
            print(f"[ERREUR] Fichier modele introuvable: {model_path}")
            return None, None, None
        modele = keras.models.load_model(model_path)
        print(f"[OK] Modele charge avec succes depuis: {model_path}")
    except Exception as e:
        print(f"[ERREUR] Erreur lors du chargement du modele: {e}")
        return None, None, None
    
    # Charger les données de test
    (_, _), (x_test, y_test) = mnist.load_data()
    
    # Normaliser les données
    x_test = x_test.astype('float32') / 255.0
    x_test = np.expand_dims(x_test, -1)
    
    print(f"[OK] Donnees de test chargees: {x_test.shape[0]} images")
    
    return modele, x_test, y_test


def generer_info_modele(modele, save_path):
    """Génère un fichier texte avec les informations du modèle"""
    print("\n📊 Génération des informations du modèle...")
    
    info = {
        "Architecture": [],
        "Paramètres_totaux": modele.count_params(),
        "Couches": len(modele.layers),
        "Date_analyse": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Compter les paramètres par type de couche
    params_par_type = {}
    for layer in modele.layers:
        layer_type = type(layer).__name__
        params = layer.count_params()
        
        if layer_type not in params_par_type:
            params_par_type[layer_type] = 0
        params_par_type[layer_type] += params
        
        info["Architecture"].append({
            "Nom": layer.name,
            "Type": layer_type,
            "Paramètres": params,
            "Forme_sortie": str(layer.output_shape) if hasattr(layer, 'output_shape') else "N/A"
        })
    
    info["Paramètres_par_type"] = params_par_type
    
    # Sauvegarder en JSON
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    
    # Générer un résumé texte
    txt_path = save_path.replace('.json', '.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("INFORMATIONS SUR LE MODÈLE CNN\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Date d'analyse: {info['Date_analyse']}\n")
        f.write(f"Nombre total de paramètres: {info['Paramètres_totaux']:,}\n")
        f.write(f"Nombre de couches: {info['Couches']}\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("ARCHITECTURE DU MODÈLE\n")
        f.write("-" * 70 + "\n\n")
        
        for i, layer_info in enumerate(info["Architecture"], 1):
            f.write(f"{i}. {layer_info['Nom']}\n")
            f.write(f"   Type: {layer_info['Type']}\n")
            f.write(f"   Paramètres: {layer_info['Paramètres']:,}\n")
            f.write(f"   Forme de sortie: {layer_info['Forme_sortie']}\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("PARAMÈTRES PAR TYPE DE COUCHE\n")
        f.write("-" * 70 + "\n\n")
        for layer_type, params in info["Paramètres_par_type"].items():
            f.write(f"{layer_type}: {params:,} paramètres\n")
    
    print(f"✅ Informations sauvegardées: {txt_path}")
    return info


def generer_matrice_confusion(modele, x_test, y_test, save_path):
    """Génère la matrice de confusion"""
    print("\n📊 Génération de la matrice de confusion...")
    
    # Prédictions
    predictions = modele.predict(x_test, verbose=0)
    predictions_classes = np.argmax(predictions, axis=1)
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, predictions_classes)
    
    # Calculer les pourcentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Créer la figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Matrice de confusion (nombres)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10),
                ax=axes[0], cbar_kws={'label': 'Nombre'})
    axes[0].set_title('Matrice de Confusion (Nombres)', fontsize=16, fontweight='bold')
    axes[0].set_xlabel('Prédiction', fontsize=12)
    axes[0].set_ylabel('Vérité', fontsize=12)
    
    # Matrice de confusion (pourcentages)
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Greens',
                xticklabels=range(10), yticklabels=range(10),
                ax=axes[1], cbar_kws={'label': 'Pourcentage (%)'})
    axes[1].set_title('Matrice de Confusion (Pourcentages)', fontsize=16, fontweight='bold')
    axes[1].set_xlabel('Prédiction', fontsize=12)
    axes[1].set_ylabel('Vérité', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Matrice de confusion sauvegardée: {save_path}")
    
    # Calculer les statistiques par classe
    stats = {}
    for i in range(10):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        stats[i] = {
            "Vrais_positifs": int(tp),
            "Faux_positifs": int(fp),
            "Faux_négatifs": int(fn),
            "Précision": float(precision * 100),
            "Rappel": float(recall * 100),
            "F1_score": float(f1 * 100)
        }
    
    return stats, cm


def generer_rapport_classification(modele, x_test, y_test, save_path):
    """Génère le rapport de classification détaillé"""
    print("\n📊 Génération du rapport de classification...")
    
    # Prédictions
    predictions = modele.predict(x_test, verbose=0)
    predictions_classes = np.argmax(predictions, axis=1)
    
    # Rapport de classification
    report = classification_report(y_test, predictions_classes,
                                  target_names=[f'Chiffre {i}' for i in range(10)],
                                  output_dict=True)
    
    # Sauvegarder en JSON
    with open(save_path.replace('.txt', '.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Sauvegarder en texte
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("RAPPORT DE CLASSIFICATION DÉTAILLÉ\n")
        f.write("=" * 70 + "\n\n")
        f.write(classification_report(y_test, predictions_classes,
                                     target_names=[f'Chiffre {i}' for i in range(10)]))
        f.write("\n" + "=" * 70 + "\n")
        f.write(f"Précision globale (macro avg): {report['macro avg']['precision']*100:.2f}%\n")
        f.write(f"Rappel global (macro avg): {report['macro avg']['recall']*100:.2f}%\n")
        f.write(f"F1-score global (macro avg): {report['macro avg']['f1-score']*100:.2f}%\n")
        f.write(f"Précision globale (weighted avg): {report['weighted avg']['precision']*100:.2f}%\n")
        f.write(f"Accuracy: {report['accuracy']*100:.2f}%\n")
    
    print(f"✅ Rapport de classification sauvegardé: {save_path}")
    return report


def generer_graphiques_performances_par_classe(stats, save_path):
    """Génère des graphiques de performance par classe"""
    print("\n📊 Génération des graphiques de performance par classe...")
    
    classes = list(stats.keys())
    precision = [stats[c]['Précision'] for c in classes]
    recall = [stats[c]['Rappel'] for c in classes]
    f1 = [stats[c]['F1_score'] for c in classes]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Précision
    bars1 = axes[0].bar(classes, precision, color='steelblue', alpha=0.8)
    axes[0].set_title('Précision par Classe', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Chiffre', fontsize=12)
    axes[0].set_ylabel('Précision (%)', fontsize=12)
    axes[0].set_ylim([0, 105])
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].set_xticks(classes)
    for bar, val in zip(bars1, precision):
        axes[0].text(bar.get_x() + bar.get_width()/2, val + 1,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Rappel
    bars2 = axes[1].bar(classes, recall, color='coral', alpha=0.8)
    axes[1].set_title('Rappel par Classe', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Chiffre', fontsize=12)
    axes[1].set_ylabel('Rappel (%)', fontsize=12)
    axes[1].set_ylim([0, 105])
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_xticks(classes)
    for bar, val in zip(bars2, recall):
        axes[1].text(bar.get_x() + bar.get_width()/2, val + 1,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # F1-Score
    bars3 = axes[2].bar(classes, f1, color='mediumseagreen', alpha=0.8)
    axes[2].set_title('F1-Score par Classe', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Chiffre', fontsize=12)
    axes[2].set_ylabel('F1-Score (%)', fontsize=12)
    axes[2].set_ylim([0, 105])
    axes[2].grid(True, alpha=0.3, axis='y')
    axes[2].set_xticks(classes)
    for bar, val in zip(bars3, f1):
        axes[2].text(bar.get_x() + bar.get_width()/2, val + 1,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Graphiques de performance sauvegardés: {save_path}")


def generer_exemples_predictions(modele, x_test, y_test, save_path, nombre=30):
    """Génère une visualisation d'exemples de prédictions"""
    print("\n📊 Génération d'exemples de prédictions...")
    
    # Sélectionner des indices variés
    indices = np.random.choice(len(x_test), min(nombre, len(x_test)), replace=False)
    
    # Prédictions
    predictions = modele.predict(x_test[indices], verbose=0)
    predictions_classes = np.argmax(predictions, axis=1)
    confidences = np.max(predictions, axis=1) * 100
    
    # Créer la figure
    cols = 10
    rows = (nombre + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, 2*rows))
    axes = axes.flatten() if nombre > 1 else [axes]
    
    for i, idx in enumerate(indices):
        ax = axes[i] if nombre > 1 else axes[0]
        
        # Afficher l'image
        ax.imshow(x_test[idx].reshape(28, 28), cmap='gray')
        
        # Couleur selon si la prédiction est correcte
        couleur = 'green' if predictions_classes[i] == y_test[indices][i] else 'red'
        symbole = '[OK]' if predictions_classes[i] == y_test[indices][i] else '[X]'
        
        ax.set_title(f'{symbole}\nVrai: {y_test[indices][i]}\nPredit: {predictions_classes[i]}\n({confidences[i]:.1f}%)',
                    color=couleur, fontsize=8, fontweight='bold')
        ax.axis('off')
    
    # Masquer les axes supplémentaires
    for i in range(len(indices), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Exemples de prédictions sauvegardés: {save_path}")


def generer_analyse_erreurs(modele, x_test, y_test, save_path, nombre=20):
    """Génère une analyse des erreurs de classification"""
    print("\n📊 Génération de l'analyse des erreurs...")
    
    # Prédictions
    predictions = modele.predict(x_test, verbose=0)
    predictions_classes = np.argmax(predictions, axis=1)
    
    # Trouver les erreurs
    erreurs = np.where(predictions_classes != y_test)[0]
    
    if len(erreurs) == 0:
        print("✅ Aucune erreur trouvée!")
        return
    
    # Limiter le nombre d'erreurs à afficher
    indices_erreurs = erreurs[:min(nombre, len(erreurs))]
    
    # Créer la figure
    cols = 5
    rows = (len(indices_erreurs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, 4*rows))
    axes = axes.flatten() if len(indices_erreurs) > 1 else [axes]
    
    for i, idx in enumerate(indices_erreurs):
        ax = axes[i] if len(indices_erreurs) > 1 else axes[0]
        
        # Afficher l'image
        ax.imshow(x_test[idx].reshape(28, 28), cmap='gray')
        
        conf = np.max(predictions[idx]) * 100
        
        ax.set_title(f'Vrai: {y_test[idx]}\nPrédit: {predictions_classes[idx]}\nConf: {conf:.1f}%',
                    color='red', fontsize=10, fontweight='bold')
        ax.axis('off')
    
    # Masquer les axes supplémentaires
    for i in range(len(indices_erreurs), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'Analyse des Erreurs ({len(erreurs)} erreurs sur {len(y_test)} images, soit {len(erreurs)/len(y_test)*100:.2f}%)',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Analyse des erreurs sauvegardée: {save_path}")


def generer_distribution_confiance(modele, x_test, y_test, save_path):
    """Génère un graphique de la distribution de confiance"""
    print("\n📊 Génération de la distribution de confiance...")
    
    # Prédictions
    predictions = modele.predict(x_test, verbose=0)
    confidences = np.max(predictions, axis=1) * 100
    predictions_classes = np.argmax(predictions, axis=1)
    
    # Séparer les prédictions correctes et incorrectes
    correctes = confidences[predictions_classes == y_test]
    incorrectes = confidences[predictions_classes != y_test]
    
    # Créer la figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Histogramme
    axes[0].hist(correctes, bins=50, alpha=0.7, label='Correctes', color='green', edgecolor='black')
    if len(incorrectes) > 0:
        axes[0].hist(incorrectes, bins=50, alpha=0.7, label='Incorrectes', color='red', edgecolor='black')
    axes[0].set_xlabel('Confiance (%)', fontsize=12)
    axes[0].set_ylabel('Fréquence', fontsize=12)
    axes[0].set_title('Distribution de la Confiance', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    data_to_plot = [correctes]
    labels = ['Correctes']
    if len(incorrectes) > 0:
        data_to_plot.append(incorrectes)
        labels.append('Incorrectes')
    
    bp = axes[1].boxplot(data_to_plot, labels=labels, patch_artist=True)
    bp['boxes'][0].set_facecolor('green')
    bp['boxes'][0].set_alpha(0.7)
    if len(bp['boxes']) > 1:
        bp['boxes'][1].set_facecolor('red')
        bp['boxes'][1].set_alpha(0.7)
    
    axes[1].set_ylabel('Confiance (%)', fontsize=12)
    axes[1].set_title('Distribution de la Confiance (Box Plot)', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Distribution de confiance sauvegardée: {save_path}")


def generer_resume_global(modele, x_test, y_test, save_path):
    """Génère un résumé global des performances"""
    print("\n📊 Génération du résumé global...")
    
    # Convertir y_test en format one-hot pour l'évaluation
    from tensorflow.keras.utils import to_categorical
    y_test_cat = to_categorical(y_test, 10)
    
    # Évaluation
    score = modele.evaluate(x_test, y_test_cat, verbose=0)
    loss, accuracy = score[0], score[1]
    
    # Prédictions
    predictions = modele.predict(x_test, verbose=0)
    predictions_classes = np.argmax(predictions, axis=1)
    
    # Statistiques
    confidences = np.max(predictions, axis=1) * 100
    erreurs = np.sum(predictions_classes != y_test)
    taux_erreur = erreurs / len(y_test) * 100
    
    # Créer la figure
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Métriques principales
    ax1 = fig.add_subplot(gs[0, :])
    ax1.axis('off')
    
    metrics_text = f"""
    ====================================================================
                            RÉSUMÉ GLOBAL DU MODÈLE
    ====================================================================
    
    📊 MÉTRIQUES PRINCIPALES:
    
    • Accuracy (Précision globale): {accuracy*100:.2f}%
    • Loss (Perte): {loss:.4f}
    • Taux d'erreur: {taux_erreur:.2f}%
    • Nombre d'erreurs: {erreurs} / {len(y_test)}
    
    📈 STATISTIQUES DE CONFIANCE:
    
    • Confiance moyenne: {np.mean(confidences):.2f}%
    • Confiance médiane: {np.median(confidences):.2f}%
    • Confiance minimale: {np.min(confidences):.2f}%
    • Confiance maximale: {np.max(confidences):.2f}%
    
    🔢 INFORMATIONS DU MODÈLE:
    
    • Paramètres totaux: {modele.count_params():,}
    • Nombre de couches: {len(modele.layers)}
    • Date d'analyse: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    
    ====================================================================
    """
    
    ax1.text(0.5, 0.5, metrics_text, transform=ax1.transAxes,
            fontsize=11, verticalalignment='center', horizontalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Graphique de précision par classe
    ax2 = fig.add_subplot(gs[1, 0])
    cm = confusion_matrix(y_test, predictions_classes)
    precision_par_classe = [cm[i, i] / cm[:, i].sum() if cm[:, i].sum() > 0 else 0 
                           for i in range(10)]
    
    bars = ax2.bar(range(10), [p*100 for p in precision_par_classe], 
                   color='steelblue', alpha=0.8)
    ax2.set_xlabel('Chiffre', fontsize=11)
    ax2.set_ylabel('Précision (%)', fontsize=11)
    ax2.set_title('Précision par Classe', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(10))
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0, 105])
    for bar, val in zip(bars, precision_par_classe):
        ax2.text(bar.get_x() + bar.get_width()/2, val*100 + 1,
                f'{val*100:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # Graphique de rappel par classe
    ax3 = fig.add_subplot(gs[1, 1])
    recall_par_classe = [cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0 
                        for i in range(10)]
    
    bars = ax3.bar(range(10), [r*100 for r in recall_par_classe], 
                   color='coral', alpha=0.8)
    ax3.set_xlabel('Chiffre', fontsize=11)
    ax3.set_ylabel('Rappel (%)', fontsize=11)
    ax3.set_title('Rappel par Classe', fontsize=12, fontweight='bold')
    ax3.set_xticks(range(10))
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim([0, 105])
    for bar, val in zip(bars, recall_par_classe):
        ax3.text(bar.get_x() + bar.get_width()/2, val*100 + 1,
                f'{val*100:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Résumé global sauvegardé: {save_path}")


def main():
    """Fonction principale"""
    print("\n" + "=" * 70)
    print("GÉNÉRATEUR DE RAPPORT D'ANALYSE DU MODÈLE MNIST")
    print("École Nationale des Sciences Appliquées - Berrechid")
    print("=" * 70)
    
    # Charger le modèle et les données
    modele, x_test, y_test = charger_modele_et_donnees()
    
    if modele is None:
        print("\n[ERREUR] Impossible de charger le modele. Arret du programme.")
        return
    
    print(f"\n📁 Tous les fichiers seront sauvegardés dans: {REPORT_DIR}/")
    
    # Générer tous les rapports et graphiques
    try:
        # 1. Informations du modèle
        info = generer_info_modele(modele, 
                                   os.path.join(REPORT_DIR, "info_modele.json"))
        
        # 2. Matrice de confusion
        stats, cm = generer_matrice_confusion(modele, x_test, y_test,
                                             os.path.join(REPORT_DIR, "matrice_confusion.png"))
        
        # 3. Rapport de classification
        report = generer_rapport_classification(modele, x_test, y_test,
                                               os.path.join(REPORT_DIR, "rapport_classification.txt"))
        
        # 4. Graphiques de performance par classe
        generer_graphiques_performances_par_classe(stats,
                                                  os.path.join(REPORT_DIR, "performance_par_classe.png"))
        
        # 5. Exemples de prédictions
        generer_exemples_predictions(modele, x_test, y_test,
                                    os.path.join(REPORT_DIR, "exemples_predictions.png"))
        
        # 6. Analyse des erreurs
        generer_analyse_erreurs(modele, x_test, y_test,
                              os.path.join(REPORT_DIR, "analyse_erreurs.png"))
        
        # 7. Distribution de confiance
        generer_distribution_confiance(modele, x_test, y_test,
                                      os.path.join(REPORT_DIR, "distribution_confiance.png"))
        
        # 8. Résumé global
        generer_resume_global(modele, x_test, y_test,
                            os.path.join(REPORT_DIR, "resume_global.png"))
        
        print("\n" + "=" * 70)
        print("✅ RAPPORT D'ANALYSE GÉNÉRÉ AVEC SUCCÈS!")
        print("=" * 70)
        print(f"\n📁 Tous les fichiers ont été sauvegardés dans: {REPORT_DIR}/")
        print("\n📊 Fichiers générés:")
        print("   • info_modele.json / info_modele.txt - Informations du modèle")
        print("   • matrice_confusion.png - Matrice de confusion")
        print("   • rapport_classification.txt / .json - Rapport détaillé")
        print("   • performance_par_classe.png - Graphiques de performance")
        print("   • exemples_predictions.png - Exemples de prédictions")
        print("   • analyse_erreurs.png - Analyse des erreurs")
        print("   • distribution_confiance.png - Distribution de confiance")
        print("   • resume_global.png - Résumé global")
        print("=" * 70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Erreur lors de la génération du rapport: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
