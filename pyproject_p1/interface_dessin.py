"""
Interface Graphique pour Dessiner et Reconnaître des Chiffres MNIST
École Nationale des Sciences Appliquées - Berrechid
Année Universitaire: 2025-2026
"""

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageDraw, ImageTk
from tensorflow import keras
from scipy import ndimage


class DessinInterface:
    def __init__(self, root):
        self.root = root
        self.root.title("Reconnaissance de Chiffres MNIST")
        self.root.geometry("600x500")
        self.root.resizable(False, False)
        
        # Charger le modèle
        self.modele = self.charger_modele()
        if self.modele is None:
            messagebox.showerror("Erreur", "Le modèle 'modele_mnist_cnn.h5' n'a pas été trouvé!")
            self.root.destroy()
            return
        
        # Variables pour le dessin
        self.dessin_actif = False
        self.dernier_point = None
        
        # Taille du canvas de dessin (plus grand pour meilleure qualité)
        self.canvas_width = 280
        self.canvas_height = 280
        
        # Créer l'interface
        self.creer_interface()
        
    def charger_modele(self):
        """Charge le modèle CNN entraîné"""
        try:
            modele = keras.models.load_model('modele_mnist_cnn.h5')
            print("✅ Modèle chargé avec succès")
            return modele
        except Exception as e:
            print(f"❌ Erreur lors du chargement du modèle: {e}")
            return None
    
    def creer_interface(self):
        """Crée l'interface utilisateur"""
        # Titre
        titre = tk.Label(self.root, text="Dessinez un chiffre (0-9)", 
                        font=("Arial", 16, "bold"))
        titre.pack(pady=10)
        
        # Frame pour le canvas
        frame_canvas = tk.Frame(self.root)
        frame_canvas.pack(pady=10)
        
        # Canvas de dessin
        self.canvas = tk.Canvas(frame_canvas, 
                                  width=self.canvas_width, 
                                  height=self.canvas_height,
                                  bg="white", 
                                  cursor="pencil")
        self.canvas.pack(side=tk.LEFT, padx=10)
        
        # Image PIL pour le dessin
        self.image = Image.new("RGB", (self.canvas_width, self.canvas_height), "white")
        self.draw = ImageDraw.Draw(self.image)
        
        # Bind des événements de souris
        self.canvas.bind("<Button-1>", self.commencer_dessin)
        self.canvas.bind("<B1-Motion>", self.dessiner)
        self.canvas.bind("<ButtonRelease-1>", self.arreter_dessin)
        
        # Label pour afficher le résultat
        frame_resultat = tk.Frame(self.root)
        frame_resultat.pack(pady=10)
        
        self.label_resultat = tk.Label(frame_resultat, 
                                       text="Prédiction: -", 
                                       font=("Arial", 24, "bold"),
                                       fg="blue")
        self.label_resultat.pack()
        
        self.label_confiance = tk.Label(frame_resultat, 
                                       text="Confiance: -", 
                                       font=("Arial", 14),
                                       fg="gray")
        self.label_confiance.pack()
        
        # Frame pour les boutons
        frame_boutons = tk.Frame(self.root)
        frame_boutons.pack(pady=20)
        
        # Bouton Guess
        self.btn_guess = tk.Button(frame_boutons, 
                                   text="🔍 Guess", 
                                   command=self.predire,
                                   font=("Arial", 14, "bold"),
                                   bg="#4CAF50",
                                   fg="white",
                                   width=12,
                                   height=2)
        self.btn_guess.pack(side=tk.LEFT, padx=10)
        
        # Bouton Clear
        self.btn_clear = tk.Button(frame_boutons, 
                                   text="🗑️ Clear", 
                                   command=self.effacer,
                                   font=("Arial", 14, "bold"),
                                   bg="#f44336",
                                   fg="white",
                                   width=12,
                                   height=2)
        self.btn_clear.pack(side=tk.LEFT, padx=10)
        
        # Instructions
        instructions = tk.Label(self.root, 
                               text="💡 Cliquez et glissez pour dessiner un chiffre",
                               font=("Arial", 10),
                               fg="gray")
        instructions.pack(pady=5)
        
    def commencer_dessin(self, event):
        """Démarre le dessin"""
        self.dessin_actif = True
        self.dernier_point = (event.x, event.y)
    
    def dessiner(self, event):
        """Dessine sur le canvas"""
        if self.dessin_actif and self.dernier_point:
            # Dessiner sur le canvas tkinter
            x, y = event.x, event.y
            self.canvas.create_line(self.dernier_point[0], self.dernier_point[1],
                                   x, y,
                                   width=12,
                                   fill="black",
                                   capstyle=tk.ROUND,
                                   smooth=tk.TRUE)
            
            # Dessiner sur l'image PIL
            self.draw.line([self.dernier_point, (x, y)],
                          fill="black",
                          width=12)
            
            self.dernier_point = (x, y)
    
    def arreter_dessin(self, event):
        """Arrête le dessin"""
        self.dessin_actif = False
        self.dernier_point = None
    
    def effacer(self):
        """Efface le canvas"""
        self.canvas.delete("all")
        self.image = Image.new("RGB", (self.canvas_width, self.canvas_height), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.label_resultat.config(text="Prédiction: -", fg="blue")
        self.label_confiance.config(text="Confiance: -")
    
    def predire(self):
        """Prédit le chiffre dessiné avec un prétraitement optimisé"""
        # Convertir l'image PIL en tableau numpy (niveaux de gris)
        img_gray = self.image.convert("L")
        img_array = np.array(img_gray, dtype=np.uint8)
        
        # Vérifier si le canvas n'est pas vide
        if np.all(img_array >= 250):  # Seuil pour blanc (tolérance pour anti-aliasing)
            messagebox.showwarning("Attention", "Veuillez dessiner un chiffre avant de prédire!")
            return
        
        # Inverser les couleurs: fond blanc (255) -> noir (0), dessin noir (0) -> blanc (255)
        # MNIST utilise: fond noir (0), dessin blanc (255)
        img_array = 255 - img_array
        
        # Trouver la bounding box du chiffre avec un seuil pour ignorer le bruit
        threshold = 30  # Seuil plus élevé pour ignorer l'anti-aliasing
        rows_with_digit = np.any(img_array > threshold, axis=1)
        cols_with_digit = np.any(img_array > threshold, axis=0)
        
        if not np.any(rows_with_digit) or not np.any(cols_with_digit):
            messagebox.showwarning("Attention", "Aucun chiffre détecté!")
            return
        
        # Obtenir les coordonnées de la bounding box
        y_min, y_max = np.where(rows_with_digit)[0][[0, -1]]
        x_min, x_max = np.where(cols_with_digit)[0][[0, -1]]
        
        # Ajouter du padding proportionnel
        h, w = y_max - y_min + 1, x_max - x_min + 1
        padding_h = max(4, h // 8)
        padding_w = max(4, w // 8)
        
        y_min = max(0, y_min - padding_h)
        y_max = min(img_array.shape[0], y_max + padding_h + 1)
        x_min = max(0, x_min - padding_w)
        x_max = min(img_array.shape[1], x_max + padding_w + 1)
        
        # Extraire la région du chiffre
        digit_region = img_array[y_min:y_max, x_min:x_max].copy()
        
        # Calculer le centre de masse du chiffre pour un meilleur centrage
        # Utiliser seulement les pixels significatifs (> seuil)
        mask = digit_region > threshold
        if np.any(mask):
            y_coords, x_coords = np.where(mask)
            weights = digit_region[y_coords, x_coords].astype(np.float32)
            center_y = np.average(y_coords, weights=weights)
            center_x = np.average(x_coords, weights=weights)
        else:
            h, w = digit_region.shape
            center_y = h / 2.0
            center_x = w / 2.0
        
        # Calculer la taille pour centrer dans un carré (avec padding)
        h, w = digit_region.shape
        # S'assurer que h et w sont positifs
        if h <= 0 or w <= 0:
            messagebox.showwarning("Attention", "Erreur lors du traitement de l'image!")
            return
        
        size = max(h, w)
        # Ajouter du padding pour permettre le centrage basé sur le centre de masse
        padding = 4
        size = max(size + padding * 2, 20)  # Minimum 20 pour le redimensionnement
        
        # Créer une image carrée avec fond noir (0)
        square_img = np.zeros((size, size), dtype=np.uint8)
        
        # Calculer l'offset pour centrer selon le centre de masse
        # Le centre de masse du chiffre doit être au centre de l'image carrée
        y_offset = int(size / 2.0 - center_y)
        x_offset = int(size / 2.0 - center_x)
        
        # Placer la région du chiffre dans l'image carrée
        # Gérer les cas où l'offset est négatif (chiffre dépasse)
        y_start_dst = max(0, y_offset)
        x_start_dst = max(0, x_offset)
        y_end_dst = min(size, y_offset + h)
        x_end_dst = min(size, x_offset + w)
        
        y_start_src = max(0, -y_offset)
        x_start_src = max(0, -x_offset)
        y_end_src = y_start_src + (y_end_dst - y_start_dst)
        x_end_src = x_start_src + (x_end_dst - x_start_dst)
        
        if y_end_src > y_start_src and x_end_src > x_start_src:
            square_img[y_start_dst:y_end_dst, x_start_dst:x_end_dst] = \
                digit_region[y_start_src:y_end_src, x_start_src:x_end_src]
        
        # Redimensionner à 20x20 en utilisant PIL pour une meilleure qualité
        # Convertir en PIL Image
        if size > 0:
            pil_img = Image.fromarray(square_img, mode='L')
            # Utiliser LANCZOS (résampling haute qualité) - compatible avec les versions récentes de PIL
            try:
                pil_img_resized = pil_img.resize((20, 20), Image.Resampling.LANCZOS)
            except AttributeError:
                # Fallback pour les anciennes versions de PIL
                pil_img_resized = pil_img.resize((20, 20), Image.LANCZOS)
            image_20x20 = np.array(pil_img_resized, dtype=np.uint8)
        else:
            # Fallback si taille invalide
            image_20x20 = ndimage.zoom(square_img, (20/max(size, 1), 20/max(size, 1))).astype(np.uint8)
        
        # Créer l'image finale 28x28 avec fond noir
        image_28x28 = np.zeros((28, 28), dtype=np.uint8)
        
        # Centrer l'image 20x20 dans l'image 28x28
        y_offset = (28 - 20) // 2
        x_offset = (28 - 20) // 2
        image_28x28[y_offset:y_offset+20, x_offset:x_offset+20] = image_20x20
        
        # Normaliser entre 0 et 1 (comme les données MNIST)
        image_norm = image_28x28.astype(np.float32) / 255.0
        
        # Préparer pour le modèle: (batch_size, height, width, channels)
        image_input = image_norm.reshape(1, 28, 28, 1)
        
        # Prédiction
        try:
            prediction = self.modele.predict(image_input, verbose=0)
            classe_predite = np.argmax(prediction[0])
            confiance = prediction[0][classe_predite] * 100
            
            # Afficher le résultat
            self.label_resultat.config(text=f"Prédiction: {classe_predite}", 
                                     fg="#4CAF50")
            self.label_confiance.config(text=f"Confiance: {confiance:.2f}%")
            
            # Afficher toutes les probabilités dans la console
            print("\n" + "="*50)
            print("RÉSULTAT DE LA PRÉDICTION")
            print("="*50)
            print(f"Chiffre reconnu: {classe_predite}")
            print(f"Confiance: {confiance:.2f}%")
            print("\nProbabilités détaillées:")
            for i in range(10):
                prob = prediction[0][i] * 100
                barre = '█' * int(prob / 2)
                print(f"   {i}: {barre} {prob:.2f}%")
            print("="*50)
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de la prédiction: {e}")


def main():
    """Fonction principale"""
    root = tk.Tk()
    app = DessinInterface(root)
    root.mainloop()


if __name__ == "__main__":
    main()
