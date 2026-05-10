import librosa
import librosa.display
import matplotlib.pyplot as plt

# Charger audio
# signal, sr = librosa.load("data/audio.wav", sr=None)
signal, sr = librosa.load("data/card_mixing_audio.mp3", sr=None)

# Extraire MFCC
mfcc = librosa.feature.mfcc(
    y=signal,
    sr=sr,
    n_mfcc=13,
    n_fft=int(0.02 * sr),
    hop_length=int(0.01 * sr)
)

# Afficher MFCC
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfcc, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()