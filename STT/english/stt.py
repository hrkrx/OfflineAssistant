import io
import os
import time
from pathlib import Path

import subprocess
import shlex
import numpy as np
import wavTranscriber
import logging

from flask import Flask, Response, render_template, request
from flask_cors import CORS
from werkzeug.utils import secure_filename

def prepare():
    # Point to a path containing the pre-trained models & resolve ~ if used
    dirName = os.path.expanduser("/app/model")

    # Resolve all the paths of model files
    output_graph, scorer = wavTranscriber.resolve_models(dirName)

    # Load output_graph, alpahbet and scorergit config --list
    return wavTranscriber.load_model(output_graph, scorer)

def stt(audio, aggressive, model):
    if audio is not None:
        title_names = ['Filename', 'Duration(s)', 'Inference Time(s)', 'Model Load Time(s)', 'Scorer Load Time(s)']
        print("\n%-30s %-20s %-20s %-20s %s" % (title_names[0], title_names[1], title_names[2], title_names[3], title_names[4]))

        inference_time = 0.0

        # Run VAD on the input file
        waveFile = audio
        segments, sample_rate, audio_length = wavTranscriber.vad_segment_generator(waveFile, aggressive)
        
        logging.debug("Saving Transcript @: %s" % waveFile.rstrip(".wav") + ".txt")
        transcript = ""
        for i, segment in enumerate(segments):
            # Run deepspeech on the chunk that just completed VAD
            logging.debug("Processing chunk %002d" % (i,))
            audio = np.frombuffer(segment, dtype=np.int16)
            output = wavTranscriber.stt(model[0], audio, sample_rate)
            inference_time += output[1]
            logging.debug("Transcript: %s" % output[0])
            transcript = transcript +output[0] + " "

        # Extract filename from the full file path
        filename, ext = os.path.split(os.path.basename(waveFile))
        logging.debug("************************************************************************************************************")
        logging.debug("%-30s %-20s %-20s %-20s %s" % (title_names[0], title_names[1], title_names[2], title_names[3], title_names[4]))
        logging.debug("%-30s %-20.3f %-20.3f %-20.3f %-0.3f" % (filename + ext, audio_length, inference_time, model[1], model[2]))
        logging.debug("************************************************************************************************************")
        print("%-30s %-20.3f %-20.3f %-20.3f %-0.3f" % (filename + ext, audio_length, inference_time, model[1], model[2]))

        return transcript

    else:
        return "No audio input detected"

# -----------------------------------------------------------------------------

UPLOAD_FOLDER = "/app/uploads"
ALLOWED_EXTENSIONS = {"wav", "ogg", "mp3", "wma", "aac"}
MODEL = prepare()

# -----------------------------------------------------------------------------

app = Flask("mozillastt")
app.config['UPLOAD_PATH'] = UPLOAD_FOLDER
CORS(app)

# -----------------------------------------------------------------------------

@app.route("/api/stt", methods=["GET", "POST"])
def api_tts():
    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(f.filename)
        f.save(filename)
        # prepare file for usage
        os.system("ffmpeg -i " + filename + " -hide_banner -nostats -f wav -acodec pcm_s16le -ar 16000 -ac 1 " + "usable_" + filename )

        # get transcription here
        text = stt("usable_" + filename, 1, MODEL)

        # remove garbage
        os.unlink(filename)
        os.unlink("usable_" + filename)

        return Response(text, mimetype="text/plain")
    return Response("Only POST is accepted", mimetype="text/plain")

@app.route("/")
def index():
    return render_template("index.html")

# -----------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003)
