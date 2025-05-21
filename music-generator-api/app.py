from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
import io
import os
import librosa
from flask_cors import CORS
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Important pour l'utilisation non-interactive (serveur)
import base64
import pretty_midi
from datetime import datetime
import soundfile as sf

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
    model_path = 'PerformanceNet_model.pth'
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

def pianoroll_to_midi(pianoroll, threshold=0.5, fs=16000, hop_length=256):
    """
    Convertit un pianoroll en fichier MIDI.
    
    Args:
        pianoroll: Tableau 2D (time, pitch) contenant les probabilités des notes
        threshold: Seuil pour considérer une note comme active
        fs: Fréquence d'échantillonnage de l'audio original
        hop_length: Taille du hop utilisé pour le spectrogramme
    
    Returns:
        Bytes du fichier MIDI
    """
    # Créer un objet PrettyMIDI
    midi = pretty_midi.PrettyMIDI()
    
    # Ajouter un instrument (piano)
    piano_program = 0  # 0 = Piano
    piano = pretty_midi.Instrument(program=piano_program)
    
    # Créer une version binaire du pianoroll
    binary_roll = pianoroll > threshold
    
    # Durée d'un frame en secondes
    frame_duration = hop_length / fs
    
    # Notes MIDI commencent à 21 (A0) pour un piano standard
    midi_offset = 21
    
    # Parcourir chaque note (pitch)
    for pitch in range(binary_roll.shape[1]):
        # Trouver les débuts et fins de notes
        diff = np.diff(binary_roll[:, pitch].astype(int), prepend=0, append=0)
        note_starts = np.where(diff > 0)[0]
        note_ends = np.where(diff < 0)[0] - 1  # -1 pour compenser le décalage dû à diff
        
        # Créer des notes MIDI pour chaque segment trouvé
        for start, end in zip(note_starts, note_ends):
            # Vérifier que la note a une durée positive et n'est pas trop courte
            if end > start and end - start >= 2:  # Au moins 2 frames de durée
                note = pretty_midi.Note(
                    velocity=100,  # Vélocité constante pour simplicité
                    pitch=pitch + midi_offset,
                    start=start * frame_duration,
                    end=end * frame_duration
                )
                piano.notes.append(note)
    
    # Ajouter l'instrument au MIDI
    midi.instruments.append(piano)
    
    # Convertir en bytes
    midi_bytes = io.BytesIO()
    midi.write(midi_bytes)
    midi_bytes.seek(0)
    
    return midi_bytes.getvalue()

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Vérifier si la requête contient un fichier
        if 'file' not in request.files:
            return jsonify({'error': 'Aucun fichier audio trouvé'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'Aucun fichier sélectionné'}), 400
        
        # Récupérer le seuil de la requête (s'il est fourni) ou utiliser une valeur par défaut plus basse
        threshold = float(request.form.get('threshold', 0.2))  # Valeur par défaut plus basse: 0.2 au lieu de 0.5
        
        # Récupérer les paramètres d'affichage (optionnels)
        max_display_length = int(request.form.get('max_display_length', 500))  # Nombre maximum de frames à afficher
        
        try:
            # Lire le fichier audio
            audio_bytes = file.read()
            
            # Convertir en mel-spectrogramme
            mel_spec = audio_to_melspectrogram(audio_bytes)
            
            # Convertir en tensor et ajouter la dimension de batch
            mel_tensor = torch.FloatTensor(mel_spec).unsqueeze(0)
            
            # Faire la prédiction
            with torch.no_grad():
                output = model(mel_tensor)
                
                # Convertir le résultat en numpy pour le traitement
                pianoroll = output.squeeze(0).cpu().numpy()
                
                # Informations de débogage
                print(f"Piano roll shape: {pianoroll.shape}")
                print(f"Piano roll min et max: {pianoroll.min()} {pianoroll.max()}")
                
                # Afficher un échantillon des valeurs maximales
                top_values = np.sort(pianoroll.flatten())[-10:]  # Les 10 plus grandes valeurs
                print(f"Top 10 valeurs: {top_values}")
                
                # Utiliser le seuil adaptatif si aucun seuil n'est fourni
                if 'threshold' not in request.form:
                    # Seuil adaptatif: calculer le seuil en fonction des données
                    if pianoroll.max() < 0.5:
                        # Si toutes les valeurs sont < 0.5, prendre un percentile élevé comme seuil
                        adaptive_threshold = np.percentile(pianoroll, 99)  # 99ème percentile
                        print(f"Utilisation d'un seuil adaptatif (99ème percentile): {adaptive_threshold}")
                        threshold = adaptive_threshold
                
                print(f"Seuil utilisé: {threshold}")
                
                # Appliquer le seuil pour obtenir des notes binaires
                binary_roll = (pianoroll > threshold).astype(np.int8)
                
                # Vérifier combien de notes sont détectées
                nb_notes = np.sum(binary_roll)
                print(f"Nombre de notes détectées avec seuil {threshold}: {nb_notes}")
                percentage = 100 * nb_notes / binary_roll.size
                print(f"Pourcentage de cellules activées: {percentage:.4f}%")
                
                # Si trop peu de notes sont détectées, générer un avertissement
                warning_message = None
                if nb_notes < 10:
                    warning_message = f"Seulement {nb_notes} notes détectées avec un seuil de {threshold}. " \
                                     f"Essayez de réduire le seuil ou d'utiliser un autre fichier audio."
                
                # OPTIMISATION DE L'AFFICHAGE: Trouver les segments pertinents
                if nb_notes > 0:
                    # Déterminer les limites pertinentes pour le graphique
                    active_frames = np.where(np.sum(binary_roll, axis=1) > 0)[0]
                    
                    if len(active_frames) > 0:
                        # Trouver les limites des régions actives
                        start_frame = max(0, active_frames[0] - 10)  # 10 frames avant la première note
                        end_frame = min(pianoroll.shape[0], active_frames[-1] + 10)  # 10 frames après la dernière note
                        
                        # Si la région est trop grande, identifier les segments les plus denses
                        display_length = end_frame - start_frame
                        if display_length > max_display_length:
                            # Diviser en fenêtres et trouver les fenêtres les plus actives
                            window_size = 50  # taille de la fenêtre d'analyse
                            activity = []
                            
                            for i in range(0, pianoroll.shape[0] - window_size, window_size // 2):  # Chevauchement de 50%
                                window_activity = np.sum(binary_roll[i:i+window_size])
                                activity.append((i, window_activity))
                            
                            # Trier les fenêtres par activité décroissante
                            activity.sort(key=lambda x: x[1], reverse=True)
                            
                            # Utiliser les N premières fenêtres les plus actives pour rester sous max_display_length
                            top_windows = []
                            cumulative_length = 0
                            for win_start, _ in activity:
                                if cumulative_length < max_display_length:
                                    win_end = min(win_start + window_size, pianoroll.shape[0])
                                    top_windows.append((win_start, win_end))
                                    cumulative_length += win_end - win_start
                                else:
                                    break
                            
                            # Fusionner les fenêtres qui se chevauchent
                            if top_windows:
                                top_windows.sort()  # Trier par temps croissant
                                merged_windows = [top_windows[0]]
                                
                                for current_start, current_end in top_windows[1:]:
                                    prev_start, prev_end = merged_windows[-1]
                                    if current_start <= prev_end:
                                        # Les fenêtres se chevauchent
                                        merged_windows[-1] = (prev_start, max(prev_end, current_end))
                                    else:
                                        # Nouvelle fenêtre
                                        merged_windows.append((current_start, current_end))
                                
                                # Utiliser ces segments pour l'affichage
                                display_segments = merged_windows
                            else:
                                # Fallback: afficher le début jusqu'à max_display_length
                                display_segments = [(0, min(max_display_length, pianoroll.shape[0]))]
                        else:
                            # La région active est assez petite pour être entièrement affichée
                            display_segments = [(start_frame, end_frame)]
                    else:
                        # Aucune note active, afficher juste le début
                        display_segments = [(0, min(max_display_length, pianoroll.shape[0]))]
                else:
                    # Aucune note détectée, montrer juste le début du pianoroll
                    display_segments = [(0, min(max_display_length, pianoroll.shape[0]))]
                
                # Créer une figure pour chaque segment et les concaténer
                plt.figure(figsize=(12, 8))
                
                # S'il y a plusieurs segments, utiliser une mise en page subplots
                n_segments = len(display_segments)
                total_display_frames = sum(end - start for start, end in display_segments)
                
                # Si on a trop de segments, on en limite le nombre
                if n_segments > 3:  # Limiter à 3 segments au maximum
                    display_segments = display_segments[:3]
                    n_segments = 3
                
                # Créer un affichage pour les probabilités brutes
                plt.subplot(2, 1, 1)
                
                if n_segments == 1:
                    # Un seul segment: afficher normalement
                    start, end = display_segments[0]
                    plt.imshow(pianoroll[start:end].T, aspect='auto', origin='lower', cmap='hot', interpolation='nearest')
                    plt.title(f'Pianoroll - Probabilités brutes (frames {start}-{end}, max={pianoroll.max():.4f})')
                else:
                    # Plusieurs segments: créer une image composite
                    segment_images = []
                    segment_labels = []
                    
                    for i, (start, end) in enumerate(display_segments):
                        segment_images.append(pianoroll[start:end].T)
                        segment_labels.append(f"{start}-{end}")
                    
                    # Ajouter des séparateurs verticaux entre segments
                    composite_image = np.hstack([np.ones((pianoroll.shape[1], 3)) * -1] + 
                                              [np.hstack([img, np.ones((pianoroll.shape[1], 3)) * -1]) 
                                               for img in segment_images])
                    
                    plt.imshow(composite_image, aspect='auto', origin='lower', cmap='hot', interpolation='nearest')
                    plt.title(f'Pianoroll - Probabilités brutes (segments: {", ".join(segment_labels)}, max={pianoroll.max():.4f})')
                
                plt.colorbar(label='Probabilités de notes (valeurs brutes)')
                plt.xlabel('Temps (frames)')
                plt.ylabel('Notes (MIDI)')
                plt.grid(True, linestyle='--', alpha=0.7)
                
                # Créer un affichage pour le pianoroll binaire
                plt.subplot(2, 1, 2)
                
                if n_segments == 1:
                    # Un seul segment: afficher normalement
                    start, end = display_segments[0]
                    plt.imshow(binary_roll[start:end].T, aspect='auto', origin='lower', cmap='Blues', interpolation='nearest')
                    plt.title(f'Pianoroll binaire - {nb_notes} notes (frames {start}-{end})')
                else:
                    # Plusieurs segments: créer une image composite
                    segment_images = []
                    
                    for start, end in display_segments:
                        segment_images.append(binary_roll[start:end].T)
                    
                    # Ajouter des séparateurs verticaux entre segments
                    composite_image = np.hstack([np.ones((binary_roll.shape[1], 3)) * -1] + 
                                              [np.hstack([img, np.ones((binary_roll.shape[1], 3)) * -1]) 
                                               for img in segment_images])
                    
                    plt.imshow(composite_image, aspect='auto', origin='lower', cmap='Blues', interpolation='nearest')
                    plt.title(f'Pianoroll binaire - {nb_notes} notes détectées')
                
                plt.colorbar(label=f'Notes binaires (seuil={threshold:.4f})')
                plt.xlabel('Temps (frames)')
                plt.ylabel('Notes (MIDI)')
                plt.grid(True, linestyle='--', alpha=0.7)
                
                plt.tight_layout()
                
                # Enregistrer l'image
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                plt.close()
                buf.seek(0)
                img_base64 = base64.b64encode(buf.read()).decode('utf-8')
                
                # Convertir le pianoroll en fichier MIDI
                try:
                    midi_bytes = pianoroll_to_midi(pianoroll, threshold, SR, HOP_LENGTH)
                    midi_base64 = base64.b64encode(midi_bytes).decode('utf-8')
                    
                    # Générer un nom de fichier avec timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    midi_filename = f"transcription_{timestamp}.mid"
                    
                    midi_success = True
                    midi_message = "Fichier MIDI généré avec succès"
                except Exception as midi_error:
                    print(f"Erreur lors de la génération du MIDI: {str(midi_error)}")
                    import traceback
                    traceback.print_exc()
                    midi_base64 = None
                    midi_filename = None
                    midi_success = False
                    midi_message = f"Erreur lors de la génération du MIDI: {str(midi_error)}"
                
                # Préparer la réponse avec les données
                result = {
                    'pianoroll': binary_roll.tolist(),
                    'probabilities': pianoroll.tolist(),
                    'shape': pianoroll.shape,
                    'timesteps': pianoroll.shape[0],
                    'notes': pianoroll.shape[1],
                    'image_base64': img_base64,
                    'threshold_used': float(threshold),
                    'max_probability': float(pianoroll.max()),
                    'notes_detected': int(nb_notes),
                    'percentage_active': float(percentage),
                    'display_segments': [{'start': int(start), 'end': int(end)} for start, end in display_segments],
                    'midi_base64': midi_base64,
                    'midi_filename': midi_filename,
                    'midi_success': midi_success,
                    'midi_message': midi_message
                }
                
                if warning_message:
                    result['warning'] = warning_message
                
                status = 'warning' if warning_message else 'success'
                return jsonify({'status': status, 'music_data': result})
                
        except Exception as e:
            print(f"ERREUR: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({'status': 'error', 'message': str(e)}), 500

class PianoToAudioModel(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=512, output_dim=128, num_layers=3, dropout=0.6):
        super(PianoToAudioModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, dropout=dropout)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_dim // 2, hidden_dim // 4, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_dim // 4, output_dim)
        self.softplus = nn.Softplus()

        self._initialize_weights()

    def _initialize_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if param.dim() >= 2:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x, lengths):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            lengths = [lengths] if not isinstance(lengths, (list, torch.Tensor)) else lengths
        x_packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        x_packed, _ = self.lstm(x_packed)
        x, _ = nn.utils.rnn.pad_packed_sequence(x_packed, batch_first=True)
        x = x.permute(1, 0, 2)
        attn_output, _ = self.attention(x, x, x)
        x = x + attn_output
        x = x.permute(1, 2, 0)
        x = self.bn(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        x = self.fc(x)
        return self.softplus(x)

# Charger le modèle génération partition -> audio
model2 = PianoToAudioModel()
model2.load_state_dict(torch.load("best_piano_to_audio_model17.pth", map_location="cpu"))
model2.eval()

def midi_to_pianoroll(midi_data, fs=100):
    pm = pretty_midi.PrettyMIDI(io.BytesIO(midi_data))
    # piano_roll : shape (128, T)
    pr = pm.get_piano_roll(fs=fs)
    # transpose ou normalisation éventuelle
    return pr.astype(np.float32)

def predict_spectrogram(pianoroll):
    # On transpose : (notes, T) → (T, notes)
    if pianoroll.shape[0] == 128:
        pianoroll = pianoroll.T  # (T, 128)
    
    input_tensor = torch.tensor(pianoroll, dtype=torch.float32).unsqueeze(0)  # shape (1, T, 128)
    lengths = [pianoroll.shape[0]]

    with torch.no_grad():
        output = model(input_tensor, lengths)  # output shape: (1, T, output_dim)
    
    return output.squeeze(0).T.numpy()  # shape: (output_dim, T)

def spectrogram_to_audio(spec, n_fft=1024, hop_length=256, n_iter=60):
    # spec doit être magnitude spectrogram
    audio = librosa.griffinlim(spec,
                               n_fft=n_fft,
                               hop_length=hop_length,
                               win_length=n_fft,
                               n_iter=n_iter)
    return audio

@app.route("/midi-to-audio", methods=["POST"])
def midi_to_audio_endpoint():
    file = request.files.get("file")
    if not file or not file.filename.endswith(".mid"):
        return jsonify({"error": "No MIDI file"}), 400

    midi_bytes = file.read()
    pr = midi_to_pianoroll(midi_bytes)
    spec = predict_spectrogram(pr)
    audio = spectrogram_to_audio(spec)

    # Sauvegarder WAV en mémoire
    wav_io = io.BytesIO()
    sf.write(wav_io, audio, samplerate=22050, format="WAV")
    wav_io.seek(0)

    # Générer spectrogramme en image (base64)
    plt.figure(figsize=(6, 4))
    plt.imshow(20 * np.log10(spec + 1e-6), origin="lower", aspect="auto")
    plt.axis("off")
    img_io = io.BytesIO()
    plt.savefig(img_io, bbox_inches="tight", pad_inches=0)
    plt.close()
    img_io.seek(0)

    # Lire et encoder l'audio
    wav_io.seek(0)
    audio_base64 = base64.b64encode(wav_io.read()).decode('utf-8')

    # Lire et encoder l'image
    img_io.seek(0)
    spectrogram_base64 = base64.b64encode(img_io.read()).decode('utf-8')

    # Retourner les deux dans un JSON
    return jsonify({
        "audio": "data:audio/wav;base64," + audio_base64,
        "spectrogram": "data:image/png;base64," + spectrogram_base64
    }), 200

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'OK', 'message': 'Le serveur API est opérationnel'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)