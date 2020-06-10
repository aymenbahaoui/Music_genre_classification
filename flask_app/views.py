import random
import os
from flask import Flask, request, jsonify, render_template, url_for, flash, redirect
from Music_classification_service import Music_Classification_Service
#import numpy as np
import youtube_to_wav

# instantiate flask app
app = Flask(__name__)
mcs = Music_Classification_Service()


@app.route("/index")
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # get file from POST request and save it
    # audio_file = request.files["file"]
    # file_name = str(random.randint(0, 100000))
    # audio_file.save(file_name)

    video = request.form["video"]
    if video == '':
        flash('No selected file')
        return redirect(request.url)
    else:
        result = []
        vidName = youtube_to_wav.youtube_to_wav(video)
        genre = mcs.predict(vidName)
        os.remove(vidName)
        for i in range(len(genre)):
            result.append((mcs._mapping[i], float("{:.4f}".format(genre[i]))))
        return render_template("predict.html", result=result, name=vidName)

