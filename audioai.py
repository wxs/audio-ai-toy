import ddsp
import ddsp.training
import numpy as np
import tensorflow as tf
import pickle
import gin
import os
import time
from scipy.io import wavfile
import sys
import numpy as np
import matplotlib.pyplot as plt


def process_audio(file):
    rate, audio = load_wav(file)
    ddsp.spectral_ops.reset_crepe()
    audio_features = ddsp.training.metrics.compute_audio_features(audio)
    audio_features['n_samples'] = audio.shape[0]
    return audio_features

def load_wav(file):
    rate, audio = wavfile.read(file)
    return rate, audio.astype("float") / 32768.0

def save_wav(fname, rate, audio):
    wavfile.write(fname, rate, (np.array(audio)*32768).astype('int16'))


def to_violin(audio_features):
    # plt.plot(audio_features['f0_hz'])
    # plt.show()
    #audio = audio_features["audio"]

    model = "Violin"
    model_dir = f"data/solo_{model.lower()}_ckpt"

    gin_file = os.path.join(model_dir, "operative_config-0.gin")
    DATASET_STATS = None
    dataset_stats_file = os.path.join(model_dir, "dataset_statistics.pkl")
    print(f"Loading dataset statistics from {dataset_stats_file}")
    try:
        if tf.io.gfile.exists(dataset_stats_file):
            with tf.io.gfile.GFile(dataset_stats_file, "rb") as f:
                DATASET_STATS = pickle.load(f)
    except Exception as err:
        print("Loading dataset statistics from pickle failed: {}.".format(err))
    with gin.unlock_config():
        gin.parse_config_file(gin_file, skip_unknown=True)
    ckpt_files = [f for f in tf.io.gfile.listdir(model_dir) if "ckpt" in f]
    ckpt_name = ckpt_files[0].split(".")[0]
    ckpt = os.path.join(model_dir, ckpt_name)
    time_steps_train = gin.query_parameter("F0LoudnessPreprocessor.time_steps")
    n_samples_train = gin.query_parameter("Harmonic.n_samples")
    hop_size = int(n_samples_train / time_steps_train)

    time_steps = int(audio_features['n_samples'] / hop_size)
    n_samples = time_steps * hop_size

    gin_params = [
        "Harmonic.n_samples = {}".format(n_samples),
        "FilteredNoise.n_samples = {}".format(n_samples),
        "F0LoudnessPreprocessor.time_steps = {}".format(time_steps),
        "oscillator_bank.use_angular_cumsum = True",  # Avoids cumsum accumulation errors.
    ]

    with gin.unlock_config():
        gin.parse_config(gin_params)

    # Trim all input vectors to correct lengths
    for key in ["f0_hz", "f0_confidence", "loudness_db"]:
        audio_features[key] = audio_features[key][:time_steps]
    #audio_features["audio"] = audio_features["audio"][:n_samples]
    model = ddsp.training.models.Autoencoder()
    model.restore(ckpt)

    # Build model by running a batch through it.
    start_time = time.time()
    _ = model(audio_features, training=False)

    outputs = model(audio_features, training=False)
    audio_gen = model.get_audio_from_outputs(outputs)
    print(audio_gen.shape)
    return audio_gen[0]

def main():
    rate, audio = load_wav("test.wav")
    print(audio.shape)
    print(np.max(audio), np.min(audio))
    ddsp.spectral_ops.reset_crepe()
    audio_features = ddsp.training.metrics.compute_audio_features(audio)
    print(audio_features)

    start_time = time.time()
    # plt.plot(audio_features['f0_hz'])
    # plt.show()
    model = "Violin"
    model_dir = f"data/solo_{model.lower()}_ckpt"
    gin_file = os.path.join(model_dir, "operative_config-0.gin")
    DATASET_STATS = None
    dataset_stats_file = os.path.join(model_dir, "dataset_statistics.pkl")
    print(f"Loading dataset statistics from {dataset_stats_file}")
    try:
        if tf.io.gfile.exists(dataset_stats_file):
            with tf.io.gfile.GFile(dataset_stats_file, "rb") as f:
                DATASET_STATS = pickle.load(f)
    except Exception as err:
        print("Loading dataset statistics from pickle failed: {}.".format(err))
    with gin.unlock_config():
        gin.parse_config_file(gin_file, skip_unknown=True)
    ckpt_files = [f for f in tf.io.gfile.listdir(model_dir) if "ckpt" in f]
    ckpt_name = ckpt_files[0].split(".")[0]
    ckpt = os.path.join(model_dir, ckpt_name)
    time_steps_train = gin.query_parameter("F0LoudnessPreprocessor.time_steps")
    n_samples_train = gin.query_parameter("Harmonic.n_samples")
    hop_size = int(n_samples_train / time_steps_train)

    time_steps = int(audio.shape[0] / hop_size)
    n_samples = time_steps * hop_size

    gin_params = [
        "Harmonic.n_samples = {}".format(n_samples),
        "FilteredNoise.n_samples = {}".format(n_samples),
        "F0LoudnessPreprocessor.time_steps = {}".format(time_steps),
        "oscillator_bank.use_angular_cumsum = True",  # Avoids cumsum accumulation errors.
    ]

    with gin.unlock_config():
        gin.parse_config(gin_params)

    # Trim all input vectors to correct lengths
    for key in ["f0_hz", "f0_confidence", "loudness_db"]:
        audio_features[key] = audio_features[key][:time_steps]
    #audio_features["audio"] = audio_features["audio"][:n_samples]
    model = ddsp.training.models.Autoencoder()
    model.restore(ckpt)

    # Build model by running a batch through it.
    start_time = time.time()
    _ = model(audio_features, training=False)

    outputs = model(audio_features, training=False)
    audio_gen = model.get_audio_from_outputs(outputs)
    print("Generated audio")
    print(audio_gen.shape)
    save_wav("output.wav", rate, audio_gen[0])


if __name__ == "__main__":
    sys.exit(main())
