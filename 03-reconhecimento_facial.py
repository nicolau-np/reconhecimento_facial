# coding= utf8 
# -*- coding: ascii -*-

# Importando OpenCV
import cv2
import numpy as np
import os 

reconhecedor = cv2.face.LBPHFaceRecognizer_create()
reconhecedor.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"

# O classificador tem que ser treinado, mas com a ajuda do OpenCV ele acaba fazendo isso para nós
faceCascade = cv2.CascadeClassifier(cascadePath);

# inicia o contador do id
id = 0
fonte_letra = cv2.FONT_HERSHEY_SIMPLEX

# nomes associados ao ID passado
names = ['Valdimiro Pinto', 'Filson Felipe', 'Esmeralde Andrade',
         'Dulce Fontes', 'Really', 'Raul', 'HTML']

# Initialize and start realtime video capture
camera = cv2.VideoCapture(0)
camera.set(3, 640) # largura
camera.set(4, 480) # altura

# Define o tamanho minimo da janela para ser reconhecido como o rosto
minW = 0.1*camera.get(3)
minH = 0.1*camera.get(4)

while True:
    # Captura cada quadro da câmera
    ret, img =camera.read()
    
    # Converte para uma Nível de cinza
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
    
    # funcao do classificador que detecta rostos na imagem
    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )
    #Este trecho irá desenhar o retangulo no rosto da pessoa
    for(x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)

        id, probabilidade = reconhecedor.predict(gray[y:y+h,x:x+w])
        if (probabilidade < 60):
            id = names[id]
            probabilidade = "  {0}%".format(round(100 - probabilidade))
        else:
            id = "desconhecido"
            probabilidade = "  "

        cv2.putText(img, str(id), (x+5,y-5), fonte_letra, 1, (255,255,255), 2)
        cv2.putText(img, str(probabilidade), (x+5,y+h-5), fonte_letra, 1, (255,255,0), 1)  
        
    # Mostra o vídeo na tela do dispositivo
    cv2.imshow('Camera',img) 

    k = cv2.waitKey(10) & 0xff # Pressione 'ESC' para sair da camera
    if k == 27:
        break

# fim
print("\n Finalizando programa...")
# Encerra a captura dos quadros
camera.release()
cv2.destroyAllWindows()
