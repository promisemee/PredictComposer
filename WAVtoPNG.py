#!/usr/bin/env python
# coding: utf-8

import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import random
import os

#load csv file
path = '/home/dain/IndividualProject/'
df = pd.DataFrame(data=pd.read_csv(path+'maestro-v2.0.0.csv',error_bad_lines=False))
df = df[["canonical_composer", "audio_filename","duration"]]
List = ['Ludwig van Beethoven','Franz Schubert','Wolfgang Amadeus Mozart','Frédéric Chopin', 'Johann Sebastian Bach']
df = df[df['canonical_composer'].isin(List)]


#load audio file and make spectrogram
audio_path = '/media/dain/TOSHIBA EXT/maestro-v2.0.0/'
image_path = '/home/dain/IndividualProject/train_image2/'

composer_list = []
filename_list = []
i = 0

data = pd.DataFrame(columns=["composer", "image_file"])

for index, row in df.iterrows():
    
    file_test = audio_path+row["audio_filename"]
    composer = row["canonical_composer"]

    if not os.path.exists(file_test):
        continue

    if composer in ['Frédéric Chopin', 'Franz Schubert']:
        repeat = 30
    elif composer in ["Ludwig van Beethoven", "Johann Sebastian Bach"]:
        repeat = 40
    elif composer in [ "Wolfgang Amadeus Mozart"]:
        repeat = 70
    else :
        repeat = 0
    for x in range(repeat):
        sec = random.randrange(0, int(row["duration"])-20)
        x_test, sr_test = librosa.load(file_test, offset=sec, duration=20)
  
        X = librosa.stft(x_test)
        Xdb = librosa.amplitude_to_db(abs(X))
        plt.figure(figsize=(14, 5))
        librosa.display.specshow(Xdb, sr=sr_test, x_axis='time', y_axis='hz')
    
        filename = 'train_dataset'+str(i)+'.png'
        plt.axis('off')
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(image_path+filename, bbox_inches='tight', pad_inches = 0)
    
        filename_list.append(filename)
        composer_list.append(composer)
    
        i+=1
        
        plt.close()

#make dataframe of spectrogram
data["composer"] = composer_list
data["image_file"] = filename_list

#save datframe to csv
file_path = "/home/dain/IndividualProject/temp/train_data.csv"
data.to_csv(file_path, mode='w', index=False)

