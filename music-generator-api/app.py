from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import io
import os
import librosa
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Permet les requêtes cross-origin depuis votre application Next.js

# Définition de l'architecture du modèle PerformanceNet
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class PerformanceNetModel(nn.Module):
    def __init__(self, input_channels=1, output_channels=1):
        super(PerformanceNetModel, self).__init__()

        # Encodeur (spectrogramme → features)
        self.encoder = nn.Sequential(
            ConvBlock(input_channels, 32),
            nn.MaxPool2d(2, 2),  # 1/2
            ConvBlock(32, 64),
            nn.MaxPool2d(2, 2),  # 1/4
            ConvBlock(64, 128),
            nn.MaxPool2d(2, 2),  # 1/8
            ConvBlock(128, 256)
        )

        # Goulot d'étranglement
        self.bottleneck = ConvBlock(256, 256)

        # Décodeur (features → pianoroll)
        self.decoder = nn.Sequential(
            ConvBlock(256, 128),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 2x
            ConvBlock(128, 64),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 4x
            ConvBlock(64, 32),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 8x
            nn.Conv2d(32, output_channels, kernel_size=1)
        )

    def forward(self, x):
        # Redimensionner pour le format CNN (batch, channels, hauteur, largeur)
        # Entrée: (batch, seq_len, n_mels) -> (batch, 1, seq_len, n_mels)
        x = x.unsqueeze(1)

        # Encoder
        features = self.encoder(x)

        # Bottleneck
        features = self.bottleneck(features)

        # Decoder
        output = self.decoder(features)

        # Sortie: (batch, 1, seq_len, 88) -> (batch, seq_len, 88)
        output = output.squeeze(1)

        # Activation sigmoïde pour obtenir des valeurs entre 0 et 1
        return torch.sigmoid(output)

# Charger le modèle
def load_model():
    # Créer une instance du modèle
    model = PerformanceNetModel(input_channels=1, output_channels=1)
    
    # Charger les poids pré-entraînés
    model_path = 'bestmodel.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()  # Mettre le modèle en mode évaluation
        print("Modèle chargé avec succès!")
    else:
        print(f"Erreur: Le fichier {model_path} n'existe pas!")
    
    return model

# Charger le modèle au démarrage de l'application
model = load_model()

# Paramètres pour le prétraitement audio
SR = 16000  # Fréquence d'échantillonnage
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 88  # Correspondant aux 88 touches du piano
MIN_DB = -80

def audio_to_melspectrogram(audio_bytes):
    """
    Convertit un fichier audio en mel-spectrogramme avec paramètres cohérents.
    """
    try:
        # Charger l'audio à partir des bytes
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=SR, mono=True)
        
        # Prétraitement audio
        # Normalisation du volume
        if y.max() > 0:
            y = y / y.max()
        
        # Mel spectrogramme
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, 
            n_mels=N_MELS, fmin=librosa.note_to_hz('A0'), fmax=librosa.note_to_hz('C8')
        )
        
        # Conversion en dB avec limite inférieure
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max, top_db=-MIN_DB)
        
        # Normalisation entre 0 et 1
        mel_spec_norm = (mel_spec_db - MIN_DB) / (-MIN_DB)
        
        return mel_spec_norm.T  # Transpose pour obtenir (T, n_mels)
    except Exception as e:
        raise Exception(f"Erreur lors du prétraitement audio: {str(e)}")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Vérifier si la requête contient un fichier
        if 'file' not in request.files:
            return jsonify({'error': 'Aucun fichier audio trouvé'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'Aucun fichier sélectionné'}), 400
        
        try:
            # Lire le fichier audio
            audio_bytes = file.read()
            
            # Convertir en mel-spectrogramme
            # Le résultat est déjà dans la forme (T, n_mels) attendue par le modèle
            mel_spec = audio_to_melspectrogram(audio_bytes)
            
            # Convertir en tensor et ajouter la dimension de batch
            mel_tensor = torch.FloatTensor(mel_spec).unsqueeze(0)
            
            # Faire la prédiction
            with torch.no_grad():
                output = model(mel_tensor)
                
                # Convertir le résultat en numpy pour le traitement
                pianoroll = output.squeeze(0).cpu().numpy()
                
                # Appliquer un seuil pour obtenir des notes binaires (activées/désactivées)
                threshold = 0.5
                binary_roll = (pianoroll > threshold).astype(np.int8)
                
                # Convertir en liste pour JSON
                result = {
                    'pianoroll': binary_roll.tolist(),
                    'probabilities': pianoroll.tolist(),
                    'shape': pianoroll.shape,
                    'timesteps': pianoroll.shape[0],
                    'notes': pianoroll.shape[1] if len(pianoroll.shape) > 1 else 88
                }
                
                return jsonify(result)
                
        except Exception as e:
            return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'OK', 'message': 'Le serveur API est opérationnel'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)