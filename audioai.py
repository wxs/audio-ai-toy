import copy
import ddsp
import ddsp.training
from ddsp.training.postprocessing import detect_notes, fit_quantile_transform
import numpy as np
import tensorflow as tf
import pickle
import gin
import librosa
import os
import time
from scipy.io import wavfile
import sys
import numpy as np
import matplotlib.pyplot as plt

# Refer to https://colab.research.google.com/github/magenta/ddsp/blob/master/ddsp/colab/demos/timbre_transfer.ipynb,
# which is Copyright 2021 Google LLC licensed under the Apache license and from which much of this code is derived.



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


def shift_ld(audio_features, ld_shift=0.0):
    """Shift loudness by a number of ocatves."""
    audio_features['loudness_db'] += ld_shift
    return audio_features


def shift_f0(audio_features, pitch_shift=0.0):
    """Shift f0 by a number of ocatves."""
    audio_features['f0_hz'] *= 2.0 ** (pitch_shift)
    audio_features['f0_hz'] = np.clip(audio_features['f0_hz'],
                                            0.0,
                                            librosa.midi_to_hz(110.0))
    return audio_features


def modify_features(audio_features, model='violin'):
    # Modified from Google's Colab here: https://colab.research.google.com/github/magenta/ddsp/blob/master/ddsp/colab/demos/timbre_transfer.ipynb
    # so architected a bit strangely.
    threshold = 1.0
    quiet = 20
    pitch_shift = 1
    loudness_shift = 0


    model_dir = f"data/solo_{model.lower()}_ckpt"
    DATASET_STATS = None
    dataset_stats_file = os.path.join(model_dir, "dataset_statistics.pkl")
    print(f"Loading dataset statistics from {dataset_stats_file}")
    try:
        if tf.io.gfile.exists(dataset_stats_file):
            with tf.io.gfile.GFile(dataset_stats_file, "rb") as f:
                DATASET_STATS = pickle.load(f)
    except Exception as err:
        print("Loading dataset statistics from pickle failed: {}.".format(err))

    audio_features_mod = copy.deepcopy(audio_features)

    mask_on, note_on_value = detect_notes(audio_features['loudness_db'],
                                        audio_features['f0_confidence'],
                                        threshold)

    if np.any(mask_on):
        # Shift the pitch register.
        target_mean_pitch = DATASET_STATS['mean_pitch']
        pitch = ddsp.core.hz_to_midi(audio_features['f0_hz'])
        mean_pitch = np.mean(pitch[mask_on])
        p_diff = target_mean_pitch - mean_pitch
        p_diff_octave = p_diff / 12.0
        round_fn = np.floor if p_diff_octave > 1.5 else np.ceil
        p_diff_octave = round_fn(p_diff_octave)
        audio_features_mod = shift_f0(audio_features_mod, p_diff_octave)


        # Quantile shift the note_on parts.
        _, loudness_norm = fit_quantile_transform(
            audio_features['loudness_db'],
            mask_on,
            inv_quantile=DATASET_STATS['quantile_transform'])

        # Turn down the note_off parts.
        mask_off = np.logical_not(mask_on)
        loudness_norm[mask_off] -=  quiet * (1.0 - note_on_value[mask_off][:, np.newaxis])
        loudness_norm = np.reshape(loudness_norm, audio_features['loudness_db'].shape)

        audio_features_mod['loudness_db'] = loudness_norm

        # Auto-tune.
        #if autotune:
          #f0_midi = np.array(ddsp.core.hz_to_midi(audio_features_mod['f0_hz']))
          #tuning_factor = get_tuning_factor(f0_midi, audio_features_mod['f0_confidence'], mask_on)
          #f0_midi_at = auto_tune(f0_midi, tuning_factor, mask_on, amount=autotune)
          #audio_features_mod['f0_hz'] = ddsp.core.midi_to_hz(f0_midi_at)

    else:
        print('\nSkipping auto-adjust (no notes detected or ADJUST box empty).')
    # Manual Shifts.
    audio_features_mod = shift_ld(audio_features_mod, loudness_shift)
    audio_features_mod = shift_f0(audio_features_mod, pitch_shift)
    return audio_features_mod


def to_violin(audio_features):
    # plt.plot(audio_features['f0_hz'])
    # plt.show()
    #audio = audio_features["audio"]

    model = "Violin"
    model_dir = f"data/solo_{model.lower()}_ckpt"

    gin_file = os.path.join(model_dir, "operative_config-0.gin")

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
    #_ = model(audio_features, training=False)
    audio_features = modify_features(audio_features)

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
