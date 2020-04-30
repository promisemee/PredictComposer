#!/usr/bin/python3
# coding: utf-8

import keras
import wave
import random
import matplotlib.pyplot as plt
import librosa.display
import contextlib
import os
from keras.preprocessing import image
import numpy as np

composerList = ['Ludwig van Beethoven', 'Wolfgang Amadeus Mozart', 'Johann Sebastian Bach', 'Franz Schubert', 'Frederic Chopin']
composerList.sort()

def most_frequent(List):
	return max(set(List), key = List.count)

def make_spectogram(upload_path, file_test):
	#get duration of audio file
	with contextlib.closing(wave.open(file_test,'r')) as f:
		frames = f.getnframes()
		rate = f.getframerate()
		duration = frames / float(rate)

	#get 5 spectogram
	for x in range(5):
		sec = random.randrange(0, int(duration-20))
		x_test, sr_test = librosa.load(file_test, offset=sec, duration=20)
  
		X = librosa.stft(x_test)
		Xdb = librosa.amplitude_to_db(abs(X))
		plt.figure(figsize=(14, 5))
		librosa.display.specshow(Xdb, sr=sr_test, x_axis='time', y_axis='hz')
		file_name = "test_"+str(x)+".png"
    
		plt.axis('off')
		plt.margins(0,0)
		plt.gca().xaxis.set_major_locator(plt.NullLocator())
		plt.gca().yaxis.set_major_locator(plt.NullLocator())
		plt.savefig(os.path.join(upload_path, file_name), bbox_inches='tight', pad_inches = 0)
		plt.close()


def predict_composer(model, test_sample_path):
	composer = ""
	predictList = []
	for x in range(5):
		img = image.load_img(test_sample_path+'test_'+str(x)+'.png', target_size=(32, 96))
		img = np.expand_dims(img, axis=0)
		y_prob = model.predict(img)
		y_pred = y_prob.argmax(axis=-1)
		predictList.append(composerList[y_pred[0]])

	return predictList

def get_others(predict, predictList):
	leftlist = predictList
	x = leftlist.count(predict)
	for i in range(x):
		leftlist.remove(predict)
	leftlist = list(set(leftlist))
	return leftlist
