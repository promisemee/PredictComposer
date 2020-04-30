#!/usr/bin/python3

from flask import Flask, render_template, request, flash

import tensorflow as tf
import keras

from werkzeug.utils import secure_filename

from pydub import AudioSegment
import os
import tempfile

from tensorflow.keras.models import load_model
from api import predict_composer, make_spectogram, most_frequent, get_others

template_dir = 'web/'
upload_dir = 'web/temp/'
static_dir = 'web/static/'

ALLOWED_EXTENSIONS = {'wav', 'mp3'}

app = Flask(__name__, 
	template_folder=template_dir,
	static_folder=static_dir)

app.config['UPLOAD_FOLDER'] = upload_dir

@app.route("/")
@app.route("/index")
def index():
	return render_template('index.html')

@app.route("/upload", methods=['POST'])
def predict():
	if request.method == 'POST':
		file = request.files['uploadfile']

		#no file selected
		if file.filename == '':
			flash('No Selected File')
			return render_template('index.html')

		#save uploaded file
		filename = secure_filename(file.filename)
		file.save(os.path.join(upload_dir+'audio/', filename))
		file_test = upload_dir+"audio/"+filename
			
		#convert to wav file if file is mp3
		if file.filename.rsplit('.', 1)[1].lower() == 'mp3':
			mp3_audio = AudioSegment.from_file(file_test)  # read mp3
			file_test = upload_dir+'audio/audio.wav'
			mp3_audio.export(file_test, format="wav")  # convert to wav

		#get spectrogram
		make_spectogram(upload_dir+'image/', file_test)

		#predict by model
		model_path = 'model/cnrn.h5'
		test_sample_path = 'web/temp/image/'
		model = load_model(model_path)
		predictList = predict_composer(model, test_sample_path)
		predict = most_frequent(predictList)
		others = get_others(predict, predictList)

		return render_template("result.html", prediction = predict, 
			otherlist=others, len = len(others))

if __name__ == '__main__':
	app.run()