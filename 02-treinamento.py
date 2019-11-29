# coding= utf8 
# -*- coding: ascii -*-


import cv2
import numpy as np
from PIL import Image
import os

# Path para o conjunto de imagens com os rostos (necessario ter o diretorio criado de antemao)
path = 'conjunto_de_treino'

# O LBPH (do OpenCV) classifica as texturas de forma simples e eficiente, rotula 
# os pixels de uma imagem ao limitar a vizinhança de cada pixel e no fim, cada histograma 
# criado é usado para representar cada imagem do conjunto de dados de treinamento
reconhecedor = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

def getImagensERotulos(path):

    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []

    for imagePath in imagePaths:

        PIL_img = Image.open(imagePath).convert('L') # converte para nivel de cinza
        img_numpy = np.array(PIL_img,'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    return faceSamples,ids

print ("\n Treinando. Espere alguns segundos...")
faces,ids = getImagensERotulos(path)
reconhecedor.train(faces, np.array(ids))

# Salva o modelo em trainer/trainer.yml
reconhecedor.write('trainer/trainer.yml') 

# Printa o numero de rostos treinados e finalizado o programa
print("\n Saindo do programa")
