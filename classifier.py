import os
import numpy as np
import librosa
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# ÉTAPE 1 — Extraction des features (MFCC + delta + delta2)
# ============================================================

def extract_features(file_path, n_mfcc=13, duration=5):
    """
    Extrait les MFCC moyens d'un fichier audio.
    
    Pourquoi la moyenne ?
    → Un fichier = des centaines de frames
    → On résume tout en UN vecteur fixe pour le classifieur
    → Vecteur final : 39 valeurs (13 MFCC + 13 delta + 13 delta2)
    """
    try:
        # Charger seulement les 5 premières secondes → plus rapide
        signal, sr = librosa.load(file_path, sr=22050, duration=duration)
        
        # Paramètres identiques à ton main.py
        n_fft      = int(0.02 * sr)   # fenêtre 20ms
        hop_length = int(0.01 * sr)   # pas 10ms
        
        # MFCC bruts (13 x N_frames)
        mfcc = librosa.feature.mfcc(
            y=signal,
            sr=sr,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length
        )
        
        # Delta et Delta-Delta
        delta  = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Concatener → shape (39, N_frames)
        combined = np.vstack([mfcc, delta, delta2])
        
        # RÉSUMER en vecteur fixe → moyenne sur le temps
        # shape finale : (39,)  ← c'est ce que le classifieur reçoit
        features = np.mean(combined, axis=1)
        
        return features
    
    except Exception as e:
        print(f"Erreur avec {file_path}: {e}")
        return None


# ============================================================
# ÉTAPE 2 — Charger tous les fichiers et créer le dataset
# ============================================================

def build_dataset(speech_dir, music_dir):
    """
    Parcourt les dossiers et construit X (features) et y (labels)
    
    Label 0 = parole (speech)
    Label 1 = musique (music)
    """
    X = []  # features
    y = []  # labels
    
    print("🔄 Chargement des fichiers PAROLE...")
    for filename in os.listdir(speech_dir):
        if filename.endswith(('.wav', '.mp3', '.flac', '.ogg')):
            path = os.path.join(speech_dir, filename)
            features = extract_features(path)
            if features is not None:
                X.append(features)
                y.append(0)  # 0 = parole
                print(f"  ✅ {filename} → vecteur shape {features.shape}")
    
    print("\n🔄 Chargement des fichiers MUSIQUE...")
    for filename in os.listdir(music_dir):
        if filename.endswith(('.wav', '.mp3', '.flac', '.ogg')):
            path = os.path.join(music_dir, filename)
            features = extract_features(path)
            if features is not None:
                X.append(features)
                y.append(1)  # 1 = musique
                print(f"  ✅ {filename} → vecteur shape {features.shape}")
    
    return np.array(X), np.array(y)


# ============================================================
# ÉTAPE 3 — Entraîner le classifieur SVM
# ============================================================

def train_classifier(X, y):
    """
    Pipeline complet :
    1. Split train/test (80/20)
    2. Normalisation (StandardScaler)
    3. Entraînement SVM
    4. Évaluation
    """
    print(f"\n📊 Dataset : {X.shape[0]} fichiers, {X.shape[1]} features chacun")
    print(f"   - Fichiers parole : {np.sum(y == 0)}")
    print(f"   - Fichiers musique : {np.sum(y == 1)}")
    
    # Split 80% entraînement / 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y          # garde les proportions dans chaque split
    )
    
    # Normalisation IMPORTANTE pour SVM
    # → met toutes les features sur la même échelle
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)   # apprend la moyenne/std sur train
    X_test  = scaler.transform(X_test)         # applique sans recalculer
    
    # SVM avec kernel RBF (meilleur pour audio)
    clf = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
    clf.fit(X_train, y_train)
    
    print("\n✅ Entraînement terminé !")
    
    # Évaluation
    y_pred = clf.predict(X_test)
    
    print("\n📈 RÉSULTATS :")
    print(classification_report(
        y_test, y_pred,
        target_names=['Parole', 'Musique']
    ))
    
    return clf, scaler, X_test, y_test, y_pred


# ============================================================
# ÉTAPE 4 — Visualisations
# ============================================================

def plot_results(y_test, y_pred, X, y):
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # --- Graphique 1 : Matrice de confusion ---
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Parole', 'Musique'],
        yticklabels=['Parole', 'Musique'],
        ax=axes[0]
    )
    axes[0].set_title('Matrice de Confusion', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Prédiction')
    axes[0].set_ylabel('Réalité')
    
    # --- Graphique 2 : Distribution des features (MFCC1 moyen) ---
    speech_mfcc1 = X[y == 0, 0]   # 1er MFCC des fichiers parole
    music_mfcc1  = X[y == 1, 0]   # 1er MFCC des fichiers musique
    
    axes[1].hist(speech_mfcc1, bins=10, alpha=0.7, 
                 label='Parole', color='steelblue', edgecolor='white')
    axes[1].hist(music_mfcc1,  bins=10, alpha=0.7, 
                 label='Musique', color='tomato',    edgecolor='white')
    axes[1].set_title('Distribution MFCC1 moyen\n(parole vs musique)', 
                      fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Valeur MFCC1')
    axes[1].set_ylabel('Nombre de fichiers')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('resultats_classification.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n💾 Figure sauvegardée : resultats_classification.png")


# ============================================================
# ÉTAPE 5 — Tester sur UN nouveau fichier (démonstration)
# ============================================================

def predict_file(file_path, clf, scaler):
    """
    Prédit si un fichier est de la parole ou de la musique.
    C'est ça qui impressionne le prof en démo live !
    """
    print(f"\n🎯 Analyse de : {file_path}")
    
    features = extract_features(file_path)
    if features is None:
        print("Erreur de chargement")
        return
    
    # Reshape pour sklearn (attend 2D)
    features_scaled = scaler.transform(features.reshape(1, -1))
    
    prediction = clf.predict(features_scaled)[0]
    
    # Score de décision (confiance du SVM)
    score = clf.decision_function(features_scaled)[0]
    
    label = "🗣️  PAROLE" if prediction == 0 else "🎵 MUSIQUE"
    print(f"Résultat : {label}")
    print(f"Score de décision : {score:.3f} (+ = musique, - = parole)")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    
    # ⚠️ MODIFIE CES CHEMINS selon ta structure
    SPEECH_DIR = "data/speech"
    MUSIC_DIR  = "data/music"
    
    # Vérifier que les dossiers existent
    if not os.path.exists(SPEECH_DIR) or not os.path.exists(MUSIC_DIR):
        print("❌ Erreur : crée les dossiers data/speech/ et data/music/")
        print("   et mets tes fichiers audio dedans")
        exit()
    
    # Pipeline complet
    print("=" * 50)
    print("  CLASSIFICATION PAROLE / MUSIQUE — MFCC + SVM")
    print("=" * 50)
    
    # 1. Construire le dataset
    X, y = build_dataset(SPEECH_DIR, MUSIC_DIR)
    
    if len(X) < 6:
        print("❌ Pas assez de fichiers ! Mets au moins 3 speech + 3 music")
        exit()
    
    # 2. Entraîner
    clf, scaler, X_test, y_test, y_pred = train_classifier(X, y)
    
    # 3. Visualiser
    plot_results(y_test, y_pred, X, y)
    
    # 4. Démo live sur tes fichiers existants
    predict_file("data/audio.wav",              clf, scaler)
    predict_file("data/card_mixing_audio.mp3",  clf, scaler)
    
    print("\n🎉 Projet terminé !")