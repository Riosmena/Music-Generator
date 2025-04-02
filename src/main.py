import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import glob
import soundfile as sf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
    
    def call(self, inputs):
        input_data, labels = inputs
        z_mean, z_log_var, z = self.encoder([input_data, labels])
        reconstruction = self.decoder([z, labels])
        return reconstruction

def generate_audio(genre):
    genre_encoded = onehot_encoder.transform([[label_encoder.transform([genre])[0]]])
    random_latent = np.random.normal(size=(1, latent_dim))
    generated_spec = vae.decoder.predict([random_latent, genre_encoded])[0, :, :, 0]
    
    # Normalizar la salida entre 0 y 1
    generated_spec = np.clip(generated_spec, 0, 1)

    # Convertir a dB (evitando valores negativos extremos)
    generated_spec = librosa.power_to_db(generated_spec, ref=np.max)

    return generated_spec


def spectrogram_to_audio(mel_spec, sr=22050):
    mel_spec = np.maximum(mel_spec, -80)  # Evita valores extremos en dB
    mel_spec = librosa.db_to_power(mel_spec)  # Convierte a escala de potencia
    audio = librosa.feature.inverse.mel_to_audio(mel_spec, sr=sr)
    return audio


def load_audio(file_path, sr=22050, n_mels=128):
    """
    Load an audio file and return the audio time series and sample rate.
    """
    y, sr = librosa.load(file_path, sr=sr)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec

def sampling(args):
    """
    Reparameterization trick by sampling from an isotropic unit Gaussian.
    """
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

data_dir = "data"
labels = []
audio_files = []

for genre in os.listdir(data_dir):
    genre_dir = os.path.join(data_dir, genre)
    if os.path.isdir(genre_dir):
        for file in glob.glob(os.path.join(genre_dir, "*.wav")):
            mel_spec = load_audio(file)
            if mel_spec.shape[1] >= 100:
                mel_spec = mel_spec[:, :100]
                audio_files.append(mel_spec)
                labels.append(genre)

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(labels)
onehot_encoder = OneHotEncoder(sparse=False)
labels_onehot = onehot_encoder.fit_transform(integer_encoded.reshape(-1, 1))

x_train = np.array(audio_files).reshape(-1, 128, 100, 1)
y_train = np.array(labels_onehot)

latent_dim = 16
num_classes = len(label_encoder.classes_)

input_shape = (128, 100, 1)
inputs = keras.Input(shape=input_shape)
x = layers.Conv2D(32, (3, 3), activation='relu', strides=2, padding='same')(inputs)
x = layers.Conv2D(64, (3, 3), activation='relu', strides=2, padding='same')(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)

labels_input = keras.Input(shape=(num_classes,))
x = layers.Concatenate()([x, labels_input])

z_mean = layers.Dense(latent_dim)(x)
z_log_var = layers.Dense(latent_dim)(x)

z = layers.Lambda(sampling)([z_mean, z_log_var])

encoder = keras.Model([inputs, labels_input], [z_mean, z_log_var, z], name="encoder")

decoder_inputs = keras.Input(shape=(latent_dim,))
decoder_lables = keras.Input(shape=(num_classes,))
x = layers.Concatenate()([decoder_inputs, decoder_lables])
x = layers.Dense(128, activation='relu')(x)
x = layers.Dense(32 * 25 * 64, activation='relu')(x)
x = layers.Reshape((32, 25, 64))(x)
x = layers.Conv2DTranspose(64, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
x = layers.Conv2DTranspose(32, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
outputs = layers.Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(x)

decoder = keras.Model([decoder_inputs, decoder_lables], outputs, name="decoder")

vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam(), loss='mse')

vae.fit([x_train, y_train], x_train, epochs=10, batch_size=16)

genre = "rock"
generated_spec = generate_audio(genre)
print(np.isnan(generated_spec).any(), np.isinf(generated_spec).any())
audio = spectrogram_to_audio(generated_spec)

sf.write(f"generated_{genre}.wav", audio, 22050)