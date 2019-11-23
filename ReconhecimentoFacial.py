# coding= utf8 
# -*- coding: ascii -*-
# Importando OpenCV
import cv2

cap = cv2.VideoCapture(0)
# O classificador tem que ser treinado, mas com a ajuda do OpenCV ele acaba fazendo isso para nós
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while(True):
    
    # Captura cada quadro da câmera
    ret, frame = cap.read()

    # Início da operação do quadro
    # Converte para uma Nível de cinza porque o Classificador (utilizados para classificar 
    #ou descrever padrões ou objetos a partir de um conjunto de propriedades oucaracterísticas.)
    # funciona em Nível de cinza (o valor de cada pixel é uma única amostra de um espaço de cores)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    print(len(faces))
    
    #Este trecho irá desenhar retangulos no rosto da pessoa
    for (x,y,w,h) in faces:
         cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
         roi_gray = gray[y:y+h, x:x+w]
         roi_color = frame[y:y+h, x:x+w]
         

    # Mostra o vídeo na tela do dispositivo
    cv2.imshow('Captura rosto',frame)
    
    # Aguarda a tecla q ser pressionada para encerrar o aplicativo (estranhamente não funciona toda vez)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Encerra a captura dos quadros
cap.release()
cv2.destroyAllWindows()
