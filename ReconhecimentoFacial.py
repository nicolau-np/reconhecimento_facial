# coding= utf8 
# -*- coding: ascii -*-

import cv2

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while(True):
    
    # Captura cada quadro da câmera
    ret, frame = cap.read()

    # Início da operação do quadro
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    print(len(faces))
    
    
    for (x,y,w,h) in faces:
         cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
         roi_gray = gray[y:y+h, x:x+w]
         roi_color = frame[y:y+h, x:x+w]
         

    # Mostra o vídeo na tela do dispositivo
    cv2.imshow('Captura rosto',frame)
    
    # Aguarda a tecla q ser pressionada para encerrar o aplicativo
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Encerra a captura dos quadros
cap.release()
cv2.destroyAllWindows()
