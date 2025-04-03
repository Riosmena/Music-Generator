import os
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

DATA_PATH = "data"
SAMPLE_RATE = 22050
DURATION = 10
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
N_MELS = 128
LATENT_DIM = 128
BATCH_SIZE = 32
EPOCHS = 10

def get_genre_encodings(genres):
    le = LabelEncoder()
    int_labels = le.fit_transform(genres)
    ohe = OneHotEncoder(sparse_output=False)
    onehots = ohe.fit_transform(int_labels.reshape(-1, 1))
    return le, {genre: onehots[i] for i, genre in enumerate(genres)}

def preprocess_audio(data_path):
    genres = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
    label_encoder, genre_dict = get_genre_encodings(genres)
    X, y = [], []

    for genre in genres:
        genre_path = os.path.join(data_path, genre)
        print(f"Processing genre: {genre}")
        for file in os.listdir(genre_path):
            if file.endswith(".wav"):
                try:
                    filepath = os.path.join(genre_path, file)
                    signal, sr = librosa.load(filepath, sr=SAMPLE_RATE)
                    if len(signal) >= SAMPLES_PER_TRACK:
                        signal = signal[:SAMPLES_PER_TRACK]
                        mel = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=N_MELS)
                        mel_db = librosa.power_to_db(mel, ref=np.max)
                        mel_db = mel_db / 80.0 + 1.0  # Normaliza a [0,1]
                        X.append(mel_db)
                        y.append(genre_dict[genre])
                except:
                    print(f"Error with {file}")
    X = np.array(X)[..., np.newaxis]
    y = np.array(y)
    return X, y, label_encoder

class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def build_encoder(input_shape, genre_dim):
    x_in = Input(shape=input_shape)
    g_in = Input(shape=(genre_dim,))
    
    x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(x_in)
    x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Flatten()(x)
    g = layers.Dense(x.shape[1], activation="relu")(g_in)
    x = layers.Concatenate()([x, g])
    x = layers.Dense(256, activation="relu")(x)

    z_mean = layers.Dense(LATENT_DIM)(x)
    z_log_var = layers.Dense(LATENT_DIM)(x)
    z = Sampling()([z_mean, z_log_var])
    
    return Model([x_in, g_in], [z_mean, z_log_var, z])

def build_decoder(output_shape, genre_dim):
    z_in = Input(shape=(LATENT_DIM,))
    g_in = Input(shape=(genre_dim,))
    x = layers.Concatenate()([z_in, g_in])

    x = layers.Dense(128 * 431, activation="relu")(x)
    x = layers.Reshape((128, 431, 1))(x)

    x = layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(1, 3, padding="same", activation="sigmoid")(x)

    return Model([z_in, g_in], x)

class VAE(Model):
    def __init__(self, encoder, decoder, kl_weight=0.1):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.kl_weight = kl_weight
        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.recon_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.recon_loss_tracker, self.kl_loss_tracker]

    def compile(self, optimizer):
        super().compile()
        self.optimizer = optimizer

    def train_step(self, data):
        (x, g) = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder([x, g])
            reconstruction = self.decoder([z, g])
            recon_loss = tf.reduce_mean(tf.square(x - reconstruction))
            kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            total_loss = recon_loss + self.kl_weight * kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.recon_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
    def test_step(self, data):
        (x, g) = data[0]
        z_mean, z_log_var, z = self.encoder([x, g])
        reconstruction = self.decoder([z, g])
        recon_loss = tf.reduce_mean(tf.square(x - reconstruction))
        kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        total_loss = recon_loss + self.kl_weight * kl_loss
        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.recon_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

def spectrogram_to_audio(mel, output_file="output.wav"):
    mel = np.clip(mel, 0, 1) 
    mel_db = mel * 80.0
    mel_power = librosa.db_to_power(mel_db)
    stft = librosa.feature.inverse.mel_to_stft(mel_power, sr=SAMPLE_RATE)
    audio = librosa.griffinlim(stft)
    sf.write(output_file, audio, SAMPLE_RATE)
    print(f"Saved generated audio in: {output_file}")

if __name__ == "__main__":
    print("Loading & preprocessing data...")
    X, y, label_encoder = preprocess_audio(DATA_PATH)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    input_shape = X.shape[1:]
    genre_dim = y.shape[1]

    encoder = build_encoder(input_shape, genre_dim)
    decoder = build_decoder(input_shape, genre_dim)

    vae = VAE(encoder, decoder)
    vae.compile(optimizer=tf.keras.optimizers.Adam())

    history = vae.fit(
        x=[X_train, y_train], 
        validation_data=([X_val, y_val], None),
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE
        )

    genre_name = label_encoder.classes_[6]
    genre_index = list(label_encoder.classes_).index(genre_name)
    genre_onehot = np.zeros((1, genre_dim))
    genre_onehot[0, genre_index] = 1

    z_random = tf.random.normal(shape=(1, LATENT_DIM))
    z_random = tf.convert_to_tensor(z_random, dtype=tf.float32)
    genre_onehot = tf.convert_to_tensor(genre_onehot, dtype=tf.float32)
    generated = decoder([z_random, genre_onehot]).numpy()
    spectrogram = np.squeeze(generated)

    plt.imshow(spectrogram, origin='lower', aspect='auto', cmap='magma')
    plt.title(f"Generated genre: {genre_name}")
    plt.colorbar()
    plt.show()

    spectrogram_to_audio(spectrogram, f"{genre_name}_generated.wav")
