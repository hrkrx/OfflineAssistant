import os
import json
import time
import logging
import threading

import docker
import requests
from flask_cors import CORS
from pymongo import MongoClient
from Levenshtein import distance
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

def advanced_compare(list1, list2, min_confidence_level = 0.8):
    confidence_level = 0
    segment_mismatch = []
    for segment in list2:
        if segment in list1:
            confidence_level = confidence_level + 1 / len(list1)
        else:
            similarty_found = False
            for target in list1:
                levenshtein_distance = distance(segment, target)
                if levenshtein_distance <= 3:
                    confidence_level = confidence_level + 1 / len(list1) / 2
                    similarty_found = True
                    break
            if not similarty_found:
                segment_mismatch.append(segment)
    return confidence_level >= min_confidence_level

def get_command(lang, text, client):
    with open("settings.json", "r") as json_file:
        settings = json.load(json_file)
        db = client.actions
        actions = db.actions.find()
        action_request = text.split(" ")
        if any(item in  settings["AttentionName"] for item in action_request):
            for action in actions:
                if advanced_compare(action["keywords"], action_request):
                    return action
    return None

def execute_command(command, text):
    result = {"hasAudioResponse":False, "audio":None}
    exec(str(command["code"]).replace("{text}", text))
    return result

@APP.route("/api/recognition", methods=["GET", "POST"])
def api_recognition():
    if request.method == 'POST':
        with open("settings.json", "r") as json_file:
            settings = json.load(json_file)
            client = MongoClient(settings["ActionDB"])
            f = request.files['file']
            filename = secure_filename(f.filename)
            f.save(filename)
            
            text = get_transcript("de", filename)
            command = get_command("de", text, client)
            
            # remove garbage
            os.unlink(filename)

            if command != None:
                result = execute_command(command, text)
                if result["hasAudioResponse"]:
                    return Response(result["audio"], mimetype="audio/wav")
                else:
                    return Response("command executed", mimetype="text/plain")
            else:
                return Response("nothing executed", mimetype="text/plain")
    return Response("Only POST is accepted", mimetype="text/plain")

@APP.route("/api/commands",  methods=["GET", "POST"])
def api_commands():
    if request.method == 'POST':
        request_action = request.args["action"]
        with open("settings.json", "r") as json_file:
            settings = json.load(json_file)
            client = MongoClient(settings["ActionDB"])
            db = client.actions
            if request_action == "list":
                actions = db.actions.find()
                response_text = " ".join([action["name"] for action in actions])
                return Response(response_text, mimetype="text/plain")
            elif request_action == "get":
                action = db.actions.find_one({"name":request.args["name"]})
                return Response(str(action), mimetype="text/plain")
            elif request_action == "new":
                action = db.actions.find_one({"name":request.args["name"]})
                if action == None:
                    new_command = {"keywords":request.args["keywords"].split(","), "code":request.args["code"], "name":request.args["name"], "lang":request.args["lang"]}
                    db.actions.insert(new_command)
                    return Response("Success", mimetype="text/plain")
                return Response("Name already exists", mimetype="text/plain")
            elif request_action == "delete":
                action = db.actions.find_one({"name":request.args["name"]})
                db.actions.delete_one({"_id":action["_id"]})
                return Response("deleted " + action["name"], mimetype="text/plain")
            else:
                return Response(action + " not supported", mimetype="text/plain")
            return Response(action + " executed. But an error occurred.", mimetype="text/plain")
    return Response("Only POST is accepted", mimetype="text/plain")
        
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    main()