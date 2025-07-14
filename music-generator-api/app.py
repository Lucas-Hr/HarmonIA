from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset, DataLoader
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
import gc
import re
from music21 import converter, stream, meter, key, pitch, duration, note, chord,environment
import os
import tempfile
import subprocess
import scipy.ndimage

# Configuration de music21
us = environment.UserSettings()
us['musicxmlPath'] = 'C:\\Program Files\\MuseScore 3\\bin\\MuseScore3.exe'
us['musescoreDirectPNGPath'] = 'C:\\Program Files\\MuseScore 3\\bin\\MuseScore3.exe'
print("MuseScore path for PNG:", us['musescoreDirectPNGPath'])
print("MuseScore path for XML:", us['musicxmlPath'])

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}},
     allow_headers=["Content-Type"], methods=["GET", "POST", "OPTIONS"])  # Permet les requêtes cross-origin depuis votre application Next.js

class MidiToSheetConverter:
    def __init__(self, midi_file_path):
        self.midi_file_path = midi_file_path
        self.score = None
        self.us = us

    def load_midi(self):
        """Load MIDI file and convert to music21 stream"""
        try:
            self.score = converter.parse(self.midi_file_path)
            print(f"Successfully loaded MIDI file: {self.midi_file_path}")
            return True
        except Exception as e:
            print(f"Error loading MIDI file: {e}")
            return False

    def analyze_score(self):
        """Analyze the musical content of the score"""
        if not self.score:
            print("No score loaded. Please load a MIDI file first.")
            return

        # Get key signature
        key_sig = self.score.analyze('key')
        print(f"Key signature: {key_sig}")

        # Get time signature
        time_sigs = self.score.flat.getElementsByClass(meter.TimeSignature)
        if time_sigs:
            print(f"Time signature: {time_sigs[0]}")

        # Get tempo
        tempos = self.score.flat.getElementsByClass('TempoIndication')
        if tempos:
            print(f"Tempo: {tempos[0]}")

        # Count parts/instruments
        parts = self.score.parts
        print(f"Number of parts: {len(parts)}")

        return {
            'key': str(key_sig),
            'time_signature': str(time_sigs[0]) if time_sigs else 'Not found',
            'parts_count': len(parts)
        }

    def generate_musicxml(self, output_path=None):
        """Generate MusicXML from the score"""
        if not self.score:
            print("No score loaded. Please load a MIDI file first.")
            return None

        if output_path is None:
            output_path = os.path.splitext(self.midi_file_path)[0] + '.musicxml'

        try:
            self.score.write('musicxml', fp=output_path)
            print(f"MusicXML saved to: {output_path}")
            return output_path
        except Exception as e:
            print(f"Error generating MusicXML: {e}")
            return None

    # def generate_musicxml_string(self):
    #     """Generate MusicXML string from the score without saving to file."""
    #     if not self.score:
    #         print("No score loaded.")
    #         return None
    #     try:
    #         musicxml_string = self.score.musicxml
    #         return musicxml_string
    #     except Exception as e:
    #         print(f"Error generating MusicXML string: {e}")
    #         return None

    def generate_png(self, output_path=None):
        """Generate PNG image of the music sheet"""
        if not self.score:
            print("No score loaded. Please load a MIDI file first.")
            return None
    
        musicxml_path = os.path.splitext(self.midi_file_path)[0] + '.musicxml'
        # generated_xml_path = self.generate_musicxml(musicxml_path)
        if not musicxml_path or not os.path.exists(musicxml_path):
            print("Failed to generate MusicXML file required for PNG conversion.")
            return None
        if output_path is None:
            output_path = os.path.splitext(self.midi_file_path)[0] + '.png'
    
        try:
            command = [
                    str(self.us['musescoreDirectPNGPath']),
                    '-o', str(output_path),
                    str(musicxml_path)
                ]

            print(f"Executing command: {' '.join(command)}")

            result = subprocess.run(command, capture_output=True, text=True, check=True)
            # Print MuseScore's output regardless, for debugging
            if result.stdout:
                print(f"MuseScore stdout:\n{result.stdout}")
            if result.stderr:
                print(f"MuseScore stderr:\n{result.stderr}")

            if result.returncode != 0:
                print(f"MuseScore command failed with exit code {result.returncode}")
                print("Please ensure MuseScore 3 is correctly installed and accessible at the specified path.")
                print("Also, check if the MusicXML file generated by music21 is valid and can be opened by MuseScore manually.")
                return None

            # Now check if the file exists after MuseScore has finished and (hopefully) succeeded
            if os.path.exists(output_path):
                print(f"PNG saved to: {output_path}")
                return output_path
            else:
                print("MuseScore command completed without error, but PNG file was not found at expected path.")
                print(f"Expected path: {output_path}")
                # Essayer de chercher des fichiers PNG générés dans le répertoire courant
                base_name = os.path.splitext(os.path.basename(self.midi_file_path))[0]
                possible_paths = [
                    f"{base_name}-1.png",  # MuseScore ajoute souvent -1 pour la première page
                    f"{base_name}.png",
                    os.path.join(os.path.dirname(output_path), f"{base_name}-1.png")
                ]

                for possible_path in possible_paths:
                    if os.path.exists(possible_path):
                        print(f"Found PNG at alternative path: {possible_path}")
                        # Optionnel: renommer vers le chemin souhaité
                        if possible_path != output_path:
                            os.rename(possible_path, output_path)
                            print(f"Renamed to: {output_path}")
                        return output_path

                return None

        except Exception as e: # Catch any other unexpected errors
            print(f"An unexpected error occurred during PNG generation: {e}")
            return None

    def simplify_score(self, max_parts=4):
        """Simplify the score for better display"""
        if not self.score:
            return None

        # If there are too many parts, combine or select the most important ones
        if len(self.score.parts) > max_parts:
            # Keep the first few parts (usually melody and bass)
            simplified = stream.Score()
            for i, part in enumerate(self.score.parts[:max_parts]):
                simplified.append(part)
            self.score = simplified

        # Remove very short notes that might clutter the display
        for part in self.score.parts:
            for element in part.flat:
                if hasattr(element, 'duration') and element.duration.quarterLength < 0.125:
                    part.remove(element)

        return self.score

    def get_json_representation(self):
        """Get a JSON representation of the score for web display"""
        if not self.score:
            return None

        score_data = {
            'parts': [],
            'metadata': {}
        }

        # Add metadata
        key_sig = self.score.analyze('key')
        score_data['metadata']['key'] = str(key_sig)

        time_sigs = self.score.flat.getElementsByClass(meter.TimeSignature)
        if time_sigs:
            score_data['metadata']['time_signature'] = str(time_sigs[0])

        # Process each part
        for i, part in enumerate(self.score.parts):
            part_data = {
                'part_number': i,
                'instrument': str(part.getInstrument()) if part.getInstrument() else 'Unknown',
                'notes': []
            }

            for element in part.flat:
                if isinstance(element, note.Note):
                    note_data = {
                        'type': 'note',
                        'pitch': str(element.pitch),
                        'duration': float(element.duration.quarterLength),
                        'offset': float(element.offset)
                    }
                    part_data['notes'].append(note_data)
                elif isinstance(element, chord.Chord):
                    chord_data = {
                        'type': 'chord',
                        'pitches': [str(p) for p in element.pitches],
                        'duration': float(element.duration.quarterLength),
                        'offset': float(element.offset)
                    }
                    part_data['notes'].append(chord_data)

            score_data['parts'].append(part_data)

        return score_data


# Définition du modèle OptimizedPerformanceNetModel
class OptimizedPerformanceNetModel(nn.Module):
    def __init__(self, input_channels=1, output_channels=1):
        super(OptimizedPerformanceNetModel, self).__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.pool3 = nn.MaxPool2d(2, 2)
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.pool4 = nn.MaxPool2d(2, 2)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.final_conv_pianoroll = nn.Conv2d(64, output_channels, kernel_size=1)
        self.final_conv_onset = nn.Conv2d(64, output_channels, kernel_size=1)
        self.final_conv_offset = nn.Conv2d(64, output_channels, kernel_size=1)

    def forward(self, x):
        x = x.unsqueeze(1)
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))
        x = self.bottleneck(self.pool4(enc4))
        x = self.upconv4(x)
        x = torch.cat([x, enc4], dim=1)
        x = self.dec4(x)
        x = self.upconv3(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.dec3(x)
        x = self.upconv2(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec2(x)
        x = self.upconv1(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec1(x)
        pianoroll = self.final_conv_pianoroll(x)
        onset = self.final_conv_onset(x)
        offset = self.final_conv_offset(x)
        return pianoroll.squeeze(1), onset.squeeze(1), offset.squeeze(1)

# Post-traitement
def advanced_post_processing(predictions, confidence_thresholds=[0.33, 0.51, 0.54, 0.54, 0.51, 0.45, 0.48, 0.29]):
    processed = predictions.copy()
    octave_ranges = [(i * 11, (i + 1) * 11) for i in range(8)]
    
    for i, (start, end) in enumerate(octave_ranges):
        threshold = confidence_thresholds[i]
        low_confidence_mask = (processed[:, start:end] > 0.2) & (processed[:, start:end] < threshold)
        processed[:, start:end][low_confidence_mask] = 0
        for pitch in range(start, end):
            pitch_roll = processed[:, pitch]
            binary_roll = pitch_roll > threshold
            labeled, num_labels = scipy.ndimage.label(binary_roll)
            for label in range(1, num_labels + 1):
                segment_mask = labeled == label
                segment_indices = np.where(segment_mask)[0]
                if len(segment_indices) < 2:
                    processed[segment_mask, pitch] = 0
                    continue
                start_idx = max(0, segment_indices[0] - 2)
                end_idx = min(len(pitch_roll), segment_indices[-1] + 3)
                context_before = pitch_roll[start_idx:segment_indices[0]]
                context_after = pitch_roll[segment_indices[-1]+1:end_idx]
                if len(context_before) > 0 and np.mean(context_before) > 0.3:
                    extension_mask = context_before > 0.3
                    processed[start_idx:segment_indices[0]][extension_mask, pitch] = threshold + 0.05
                if len(context_after) > 0 and np.mean(context_after) > 0.3:
                    extension_mask = context_after > 0.3
                    processed[segment_indices[-1]+1:end_idx][extension_mask, pitch] = threshold + 0.05
    for pitch in range(processed.shape[1]):
        pitch_roll = processed[:, pitch]
        binary_roll = pitch_roll > confidence_thresholds[pitch // 11]
        kernel = np.ones(4)
        opened = scipy.ndimage.binary_opening(binary_roll, structure=kernel)
        processed[~opened, pitch] = 0
    for t in range(processed.shape[0]):
        frame = processed[t, :]
        strong_notes = np.where(frame > confidence_thresholds[pitch // 11] + 0.2)[0]
        for note in strong_notes:
            if note + 12 < len(frame) and frame[note + 12] > 0.25:
                processed[t, note + 12] = min(frame[note + 12] + 0.05, 1.0)
            if note + 7 < len(frame) and frame[note + 7] > 0.25:
                processed[t, note + 7] = min(frame[note + 7] + 0.03, 1.0)
    return processed

# Classe pour segmenter les données
class SpectrogramDataset(Dataset):
    def __init__(self, spectrogram, segment_length=400, overlap=0.5):
        self.spectrogram = spectrogram
        self.segment_length = segment_length
        self.hop_length = int(segment_length * (1 - overlap))
        self.num_segments = max(1, (spectrogram.shape[0] - segment_length) // self.hop_length + 1)
        
    def __len__(self):
        return self.num_segments
    
    def __getitem__(self, idx):
        start = idx * self.hop_length
        end = start + self.segment_length
        spec_segment = self.spectrogram[start:end, :]
        if spec_segment.shape[0] < self.segment_length:
            pad_length = self.segment_length - spec_segment.shape[0]
            spec_segment = np.pad(spec_segment, ((0, pad_length), (0, 0)), mode='constant')
        return torch.tensor(spec_segment, dtype=torch.float32)

# Reconstruction des prédictions
def reconstruct_pianoroll(predictions, segment_length, hop_length, total_length):
    num_segments, _, num_pitches = predictions.shape
    pianoroll = np.zeros((total_length, num_pitches))
    counts = np.zeros((total_length, num_pitches))
    for i in range(num_segments):
        start = i * hop_length
        end = start + segment_length
        pianoroll[start:end, :] += predictions[i, :min(segment_length, total_length - start), :]
        counts[start:end, :] += 1
    counts[counts == 0] = 1
    pianoroll = pianoroll / counts
    return pianoroll

# Charger le modèle
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Créer une instance du modèle
    model = OptimizedPerformanceNetModel(input_channels=1, output_channels=1).to(device)
    
    # Charger les poids pré-entraînés
    model_path = 'GenerPart-9-22.pth'
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            model.eval()  # Mettre le modèle en mode évaluation
            print("Modèle chargé avec succès!")
        except RuntimeError as e:
            print(f"Erreur lors du chargement du modèle {model_path}: {e}")
            raise
    else:
        print(f"Erreur: Le fichier {model_path} n'existe pas!")
        raise FileNotFoundError(f"Le fichier {model_path} n'existe pas!")
    return model, device

# Charger le modèle au démarrage de l'application
try:
    model, device = load_model()
except Exception as e:
    print(f"Échec du chargement du modèle: {e}")
    exit(1)

# Paramètres pour le prétraitement audio
SR = 16000  # Fréquence d'échantillonnage
N_FFT = 2048
HOP_LENGTH = 160
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
        mel_spec_norm = (mel_spec_db - np.mean(mel_spec_db)) / (np.std(mel_spec_db) + 1e-6)
        
        return mel_spec_norm.T  # Transpose pour obtenir (T, n_mels)
    except Exception as e:
        raise Exception(f"Erreur lors du prétraitement audio: {str(e)}")

def pianoroll_to_midi(pianoroll, onsets, offsets, pianoroll_thresholds=[0.33, 0.51, 0.54, 0.54, 0.51, 0.45, 0.48, 0.29], 
                      onset_threshold=0.4, offset_threshold=0.4, fs=16000, hop_length=160):
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
    piano = pretty_midi.Instrument(program=0) # 0 = Piano
    
    # Créer une version binaire du pianoroll
    pianoroll_binary = np.zeros_like(pianoroll)
    octave_ranges = [(i * 11, (i + 1) * 11) for i in range(8)]
    for i, (start, end) in enumerate(octave_ranges):
        pianoroll_binary[:, start:end] = (pianoroll[:, start:end] >= pianoroll_thresholds[i]).astype(np.int32)
    onsets_binary = (onsets >= onset_threshold).astype(np.int32)
    offsets_binary = (offsets >= offset_threshold).astype(np.int32)
    
    # Durée d'un frame en secondes
    frame_duration = hop_length / fs
    
    # Notes MIDI commencent à 21 (A0) pour un piano standard
    midi_offset = 21
    
    # Parcourir chaque note (pitch)
    for pitch in range(pianoroll_binary.shape[1]):
        onset_times = np.where(onsets_binary[:, pitch] > 0)[0]
        for start_idx in onset_times:
            offset_candidates = np.where(offsets_binary[start_idx:, pitch] > 0)[0]
            if len(offset_candidates) > 0:
                end_idx = start_idx + offset_candidates[0]
                if end_idx > start_idx and end_idx - start_idx >= 2:
                    note = pretty_midi.Note(
                        velocity=100,
                        pitch=pitch + midi_offset,
                        start=start_idx * frame_duration,
                        end=end_idx * frame_duration
                    )
                    piano.notes.append(note)
    
    # Ajouter l'instrument au MIDI
    midi.instruments.append(piano)
    
    # Convertir en bytes
    midi_bytes = io.BytesIO()
    midi.write(midi_bytes)
    midi_bytes.seek(0)
    
    return midi_bytes.getvalue()

def midi_to_musicxml(midi_bytes):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mid') as tmp_midi:
            tmp_midi.write(midi_bytes)
            tmp_midi_path = tmp_midi.name
        with tempfile.NamedTemporaryFile(delete=False, suffix='.musicxml') as tmp_xml:
            tmp_xml_path = tmp_xml.name
        midi_stream = converter.parse(tmp_midi_path)
        midi_stream.write('musicxml', fp=tmp_xml_path)
        with open(tmp_xml_path, 'r', encoding='utf-8') as f:
            musicxml_content = f.read()
        os.unlink(tmp_midi_path)
        os.unlink(tmp_xml_path)
        return musicxml_content
    except Exception as e:
        raise Exception(f"Erreur lors de la conversion MIDI vers MusicXML: {str(e)}")



# --- 2. Functions for Piano Roll to ABC (from your input) ---

def extract_notes_from_piano_roll(piano_roll, time_resolution=0.1, velocity_threshold=0.5):
    notes = []
    # Iterate over each MIDI key (0-87 for 21-108)
    for key_idx in range(piano_roll.shape[1]): # Use piano_roll.shape[1] for actual number of keys
        midi_note = key_idx + 21 # MIDI notes from 21 (A0) to 108 (C8)
        key_data = piano_roll[:, key_idx] # All frames for this single key

        in_note = False
        note_start_frame = 0

        for frame_idx in range(len(key_data)):
            if key_data[frame_idx] > velocity_threshold and not in_note:
                # Note starts
                note_start_frame = frame_idx
                in_note = True
            elif key_data[frame_idx] <= velocity_threshold and in_note:
                # Note ends
                note_end_frame = frame_idx
                duration_frames = note_end_frame - note_start_frame
                
                # Convert duration to time units
                duration_time = duration_frames * time_resolution
                
                # Only add if duration is meaningful
                if duration_time > 0.05: # Minimum duration threshold
                    notes.append({
                        'midi': midi_note,
                        'start_time': note_start_frame * time_resolution,
                        'duration': duration_time,
                        'velocity': np.mean(key_data[note_start_frame:note_end_frame]) # Average velocity over the note
                    })
                in_note = False
        
        # Handle notes that extend to the very end of the piano roll
        if in_note:
            note_end_frame = len(key_data)
            duration_frames = note_end_frame - note_start_frame
            duration_time = duration_frames * time_resolution
            if duration_time > 0.05:
                notes.append({
                    'midi': midi_note,
                    'start_time': note_start_frame * time_resolution,
                    'duration': duration_time,
                    'velocity': np.mean(key_data[note_start_frame:note_end_frame])
                })
    return notes

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier audio trouvé'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Aucun fichier sélectionné'}), 400
    try:
        audio_bytes = file.read()
        mel_spec = audio_to_melspectrogram(audio_bytes)
        if mel_spec.shape[1] != 88:
            return jsonify({'error': f"Le spectrogramme doit avoir 88 bins fréquentiels, mais a {mel_spec.shape[1]}"}), 400
        segment_length = 400
        overlap = 0.5
        hop_length = int(segment_length * (1 - overlap))
        batch_size = 64
        confidence_thresholds = [0.33, 0.51, 0.54, 0.54, 0.51, 0.45, 0.48, 0.29]
        onset_threshold = 0.4
        offset_threshold = 0.4
        dataset = SpectrogramDataset(mel_spec, segment_length, overlap)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                                pin_memory=torch.cuda.is_available())
        pred_pianorolls = []
        pred_onsets = []
        pred_offsets = []
        with torch.no_grad():
            for spec_batch in dataloader:
                spec_batch = spec_batch.to(device, non_blocking=True)
                if device.type == 'cuda':
                    with torch.amp.autocast(device_type='cuda'):
                        pred_pianoroll, pred_onset, pred_offset = model(spec_batch)
                else:
                    pred_pianoroll, pred_onset, pred_offset = model(spec_batch)
                pred_pianoroll = torch.sigmoid(pred_pianoroll).cpu().numpy()
                pred_onset = torch.sigmoid(pred_onset).cpu().numpy()
                pred_offset = torch.sigmoid(pred_offset).cpu().numpy()
                for i in range(pred_pianoroll.shape[0]):
                    pred_pianoroll[i] = advanced_post_processing(pred_pianoroll[i], confidence_thresholds)
                pred_pianorolls.append(pred_pianoroll)
                pred_onsets.append(pred_onset)
                pred_offsets.append(pred_offset)
        pred_pianorolls = np.concatenate(pred_pianorolls, axis=0)
        pred_onsets = np.concatenate(pred_onsets, axis=0)
        pred_offsets = np.concatenate(pred_offsets, axis=0)
        pianoroll_pred = reconstruct_pianoroll(pred_pianorolls, segment_length, hop_length, mel_spec.shape[0])
        onset_pred = reconstruct_pianoroll(pred_onsets, segment_length, hop_length, mel_spec.shape[0])
        offset_pred = reconstruct_pianoroll(pred_offsets, segment_length, hop_length, mel_spec.shape[0])
        pianoroll_binary = np.zeros_like(pianoroll_pred)
        octave_ranges = [(i * 11, (i + 1) * 11) for i in range(8)]
        for i, (start, end) in enumerate(octave_ranges):
            pianoroll_binary[:, start:end] = (pianoroll_pred[:, start:end] >= confidence_thresholds[i]).astype(np.int32)
        nb_notes = np.sum(pianoroll_binary)
        percentage = 100 * nb_notes / pianoroll_binary.size
        warning_message = None
        if nb_notes < 10:
            warning_message = f"Seulement {nb_notes} notes détectées. Essayez un fichier audio différent."
        midi_bytes = pianoroll_to_midi(pianoroll_pred, onset_pred, offset_pred, 
                                       confidence_thresholds, onset_threshold, offset_threshold, SR, HOP_LENGTH)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        midi_filename = f"transcription_{timestamp}.mid"
        with open(midi_filename, 'wb') as f:
            f.write(midi_bytes)
        converter = MidiToSheetConverter(midi_filename)
        if converter.load_midi():
            analysis = converter.analyze_score()
            converter.simplify_score()
            musicxml_path = converter.generate_musicxml()
            musicxml_base64 = None
            if musicxml_path and os.path.exists(musicxml_path):
                with open(musicxml_path, 'r', encoding='utf-8') as f:
                    musicxml_content = f.read()
                    musicxml_base64 = base64.b64encode(musicxml_content.encode('utf-8')).decode('utf-8')
            png_path = converter.generate_png()
            png_base64 = None
            if png_path and os.path.exists(png_path):
                with open(png_path, 'rb') as f:
                    png_content = f.read()
                    png_base64 = base64.b64encode(png_content).decode('utf-8')
            json_data = converter.get_json_representation()
            print("Conversion complete!")
            print(f"MusicXML: {musicxml_path}")
            print(f"PNG: {png_path}")
        else:
            raise Exception("Échec du chargement du fichier MIDI pour conversion.")
        result = {
            'pianoroll': pianoroll_binary.tolist(),
            'probabilities': pianoroll_pred.tolist(),
            'onsets': onset_pred.tolist(),
            'offsets': offset_pred.tolist(),
            'shape': pianoroll_pred.shape,
            'timesteps': pianoroll_pred.shape[0],
            'notes': pianoroll_pred.shape[1],
            'threshold_used': confidence_thresholds,
            'onset_threshold': float(onset_threshold),
            'offset_threshold': float(offset_threshold),
            'max_probability': float(pianoroll_pred.max()),
            'notes_detected': int(nb_notes),
            'percentage_active': float(percentage),
            'midi_base64': base64.b64encode(midi_bytes).decode('utf-8'),
            'midi_filename': midi_filename,
            'midi_success': True,
            'midi_message': "Fichier MIDI généré avec succès",
            'xml_base64': musicxml_base64,
            'xml_filename': musicxml_path.split('/')[-1] if musicxml_path else None,
            'png_path': png_path,
            'png_base64': png_base64,
            'json_data': json_data
        }
        if warning_message:
            result['warning'] = warning_message
        status = 'warning' if warning_message else 'success'
        if device.type == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()
        return jsonify({'status': status, 'music_data': result})
    except Exception as e:
        print(f"ERREUR: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

# Generation partition to audio      
# Paramètres
DURATION = 30
SR = 22050
FS = 86
N_FFT = 254
HOP_LENGTH = 256
WIN_LENGTH = 254
N_ITER = 100
MAX_FRAMES = 500
class PianoToAudioModel(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=128, num_layers=2, dropout=0.6):
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
        self.sigmoid = nn.Sigmoid()

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
        return self.sigmoid(x)

# Charger le modèle génération partition -> audio
model2 = PianoToAudioModel()
model2.load_state_dict(torch.load("best_piano_to_audio_model17.pth", map_location="cpu"))
model2.eval()

def midi_to_pianoroll(midi_data, fs=FS):
    try:
        # Vérifier si midi_data est vide
        if not midi_data:
            raise ValueError("Fichier MIDI vide ou non fourni")
        
        pm = pretty_midi.PrettyMIDI(io.BytesIO(midi_data))
        end_time = min(pm.get_end_time(), DURATION)
        if end_time <= 0:
            raise ValueError("Le fichier MIDI n'a pas de durée valide")
        
        times = np.linspace(0, end_time, int(end_time * fs))
        pianoroll = pm.get_piano_roll(fs=fs, times=times)
        pianoroll = pianoroll.T  # (T, 128)
        pianoroll = (pianoroll > 0).astype(np.float32)
        
        # Padding ou troncature à MAX_FRAMES
        current_length = pianoroll.shape[0]
        if current_length < MAX_FRAMES:
            padding = np.zeros((MAX_FRAMES - current_length, pianoroll.shape[1]), dtype=np.float32)
            pianoroll = np.concatenate([pianoroll, padding], axis=0)
        elif current_length > MAX_FRAMES:
            pianoroll = pianoroll[:MAX_FRAMES, :]
        
        print(f"Pianoroll shape: {pianoroll.shape}")  # Débogage
        return pianoroll, min(current_length, MAX_FRAMES)
    except ValueError as e:
        raise ValueError(f"Erreur de validation MIDI : {e}")
    except Exception as e:
        raise RuntimeError(f"Erreur lors de la conversion MIDI en pianoroll : {e}")

def predict_spectrogram(pianoroll, length, chunk_size=1000):
    pianoroll_tensor = torch.tensor(pianoroll, dtype=torch.float32)
    outputs = []
    
    # Traiter par segments
    for i in range(0, length, chunk_size):
        chunk = pianoroll_tensor[i:i + chunk_size]  # Segment de taille chunk_size
        input_tensor = chunk.unsqueeze(0)  # (1, chunk_T, 128)
        chunk_length = [chunk.shape[0]]

        with torch.no_grad():
            output = model2(input_tensor, chunk_length)  # output shape: (1, chunk_T, 128)
            print(f"Output shape for chunk {i}: {output.shape}")  # Débogage
        outputs.append(output.squeeze(0))  # (chunk_T, 128)

    # Recombinaison des segments
    full_output = torch.cat(outputs, dim=0)  # (T, 128)
    print(f"Full output shape: {full_output.shape}")  # Débogage
    spectrogram = full_output.numpy()
    # Normalisation comme dans generate_spectrogram
    spec_min, spec_max = spectrogram.min(), spectrogram.max()
    spectrogram = np.clip(spectrogram, 0, 1)
    del pianoroll_tensor, full_output
    gc.collect()
    return spectrogram.T, spec_min, spec_max  # (128, T)

def spectrogram_to_audio(spectrogram, spec_min, spec_max):
    print(f"Spectrogram shape: {spectrogram.shape}")  # Débogage
    # spec doit être magnitude spectrogram
    if spectrogram.shape[0] != N_FFT // 2 + 1:
        raise ValueError(f"Spectrogram frequency dimension {spectrogram.shape[0]} does not match expected {N_FFT // 2 + 1}")
    # Inverser la normalisation
    spectrogram = spectrogram * (spec_max - spec_min) + spec_min
    spectrogram = np.expm1(spectrogram)
    spectrogram = np.clip(spectrogram, 0, spectrogram.max() * 1.1)
    try:
        audio = librosa.griffinlim(spectrogram, n_iter=N_ITER, hop_length=HOP_LENGTH,
                                  win_length=WIN_LENGTH, n_fft=N_FFT)
    except Exception as e:
        raise RuntimeError(f"Erreur dans griffinlim : {e}")
    audio = audio / max(abs(audio)) if np.max(np.abs(audio)) > 0 else audio
    print(f"Audio shape: {audio.shape}")  # Débogag
    return audio

@app.route("/midi-to-audio", methods=["POST"])
def midi_to_audio_endpoint():
    try:
        file = request.files.get("file")
        if not file or not file.filename.endswith(".midi"):
            return jsonify({"error": "No MIDI file"}), 400
        midi_bytes = file.read()
        if not midi_bytes:
            return jsonify({"error": "Fichier MIDI vide"}), 400
        
        pianoroll, length = midi_to_pianoroll(midi_bytes)
        spec, spec_min, spec_max = predict_spectrogram(pianoroll, length)
        audio = spectrogram_to_audio(spec, spec_min, spec_max)

        # Sauvegarder WAV en mémoire
        wav_io = io.BytesIO()
        print(f"Audio length: {len(audio)}, wav_io type: {type(wav_io)}")  # Débogage
        sf.write(wav_io, audio, SR, format="WAV")
        wav_io.seek(0)

        # Générer spectrogramme en image (base64)
        plt.figure(figsize=(6, 4))
        plt.imshow(20 * np.log10(spec + 1e-6), origin="lower", aspect="auto")
        plt.axis("off")
        img_io = io.BytesIO()
        plt.savefig(img_io, bbox_inches="tight", pad_inches=0, format="png")
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
            'status': 'success',
            "audio": "data:audio/wav;base64," + audio_base64,
            "spectrogram": "data:image/png;base64," + spectrogram_base64
        }), 200
    except ValueError as e:
        app.logger.error(f"Erreur de validation: {str(e)}")
        return jsonify({"error": str(e)}), 400
    except RuntimeError as e:
        app.logger.error(f"Erreur de mémoire ou de traitement: {str(e)}")
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        app.logger.error(f"Erreur inattendue: {str(e)}")
        return jsonify({"error": f"Erreur interne: {str(e)}"}), 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'OK', 'message': 'Le serveur API est opérationnel'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)