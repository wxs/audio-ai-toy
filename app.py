from flask import Flask, render_template, request, jsonify, make_response
import audioai
from io import BytesIO
import numpy as np
import os
import subprocess

if not os.path.exists("data"):
    print("Data files not found! Downloading from Google Drive...")
    os.mkdir("data")
    cmd = [
        "gsutil",
        "cp",
        "-r",
        "gs://ddsp/models/timbre_transfer_colab/2021-07-08/*",
        "data"
        ]
    subprocess.run(cmd)
    print("Downloaded!")

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/crepe", methods=['POST'])
def extract_features():
    f = request.files['audio']
    #f.save('uploaded.wav')
    features = audioai.process_audio(f)
    result = {
        'n_samples': features['n_samples'],
        'f0_hz': features['f0_hz'].tolist(),
        'f0_confidence': features['f0_confidence'].tolist(),
        'loudness_db': features['loudness_db'].tolist()}

    return jsonify(result)

@app.route("/violin", methods=['POST'])
def generate_violin():
    J = request.json

    features = {
    'n_samples': J['n_samples'],
    'f0_hz': np.asarray(J['f0_hz']),
    'f0_confidence': np.asarray(J['f0_confidence']),
    'loudness_db': np.asarray(J['loudness_db']),
    }
    violin_sound = audioai.to_violin(features)
    print(violin_sound)

    buffer = BytesIO()
    audioai.save_wav(buffer, 44100, violin_sound)

    response = make_response(buffer.getvalue())
    response.mimetype = 'audio/wav'
    #with open('testout.wav', 'wb') as f:
        #f.write(buffer.getvalue())
    return response
