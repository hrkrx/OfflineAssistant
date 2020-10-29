import os
import json
import time
import logging
import threading

import docker
import pymongo
import requests
from flask_cors import CORS
from werkzeug.utils import secure_filename
from flask import Flask, Response, render_template, request

SHUTTING_DOWN = False
APP = Flask("OfflineAssistant")

def main():
    global SHUTTING_DOWN
    global APP
    logging.info("Starting Offline Assistant")

    # load settings
    with open("settings.json", "r") as json_file:
        settings = json.load(json_file)
        UploadFolder = settings["UploadFolder"]
        if settings["UseContainerWatchdog"] == True:
            thread = threading.Thread(target=container_watchdog)
            thread.start()

        # prepare flask to receive audio samples

        if not os.path.exists(UploadFolder):
            os.mkdir(UploadFolder)

        APP.config['UPLOAD_PATH'] = UploadFolder
        CORS(APP)
        APP.run(host="0.0.0.0", port=5001)
    SHUTTING_DOWN = True

def container_watchdog():
    client = docker.from_env()
    while not SHUTTING_DOWN:
        logging.info(client.containers.list())
        time.sleep(5)

def get_transcript(lang, filename):
    with open("settings.json", "r") as json_file:
        settings = json.load(json_file)
        languages = settings["languages"]
        if lang in languages:
            stt_port = languages[lang]["sttPort"]
            # build POST request
            with open(filename, "rb") as audiofile:
                files = {"file": audiofile}
                return requests.post("http://localhost:" + str(stt_port) + "/api/stt", files=files).text

def get_audio(lang, text):
    with open("settings.json", "r") as json_file:
        settings = json.load(json_file)
        languages = settings["languages"]
        if lang in languages:
            tts_port = languages[lang]["ttsPort"]
            # build GET request
            data = {"text": text}
            response = requests.get("http://localhost:" + str(tts_port) + "/api/tts", data).content
            return response

@APP.route("/api/recognition", methods=["GET", "POST"])
def api_tts():
    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(f.filename)
        f.save(filename)
        
        text = get_transcript("de", filename)
        audio = get_audio("de", text)
        # remove garbage
        os.unlink(filename)

        return Response(audio, mimetype="audio/wav")
    return Response("Only POST is accepted", mimetype="text/plain")

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    main()