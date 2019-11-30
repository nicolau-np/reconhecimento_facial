# coding= utf8 
# -*- coding: ascii -*-

# Importando OpenCV
import cv2
import os

camera = cv2.VideoCapture(0)
camera.set(3, 640) # largura
camera.set(4, 480) # altura

# O classificador tem que ser treinado, mas com a ajuda do OpenCV ele acaba fazendo isso para nós
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Cada pessoa terá para si um valor numerico associado chamado id do usuario
id_rosto = input('\n Digite o valor do seu ID e pressione <Enter> =  ')

print("\n Iniciando captura facial. Olhe para a camera e espera um pouco...")
#Iniciando o contador individual de amostras faciais
contador = 0

while(True):
	# Captura cada quadro da câmera
    ret, img = camera.read()
    
    # Converte para uma Nível de cinza porque o Classificador (utilizados para classificar 
    #ou descrever padrões ou objetos a partir de um conjunto de propriedades ou características.)
    # funciona em Nível de cinza (o valor de cada pixel é uma única amostra de um espaço de cores)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # funcao do classificador que detecta rostos na imagem
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    
    #Este trecho irá desenhar o retangulo no rosto da pessoa
    for (x,y,w,h) in faces:
		# "desenha" o quadrado no rosto
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        contador += 1

        # Salva a captura de imagem no folder destinado
        cv2.imwrite("conjunto_de_treino/User." + str(id_rosto) + '.' + str(contador) + ".jpg", gray[y:y+h,x:x+w])
        
        # Mostra o vídeo na tela do dispositivo
        cv2.imshow('Sorria', img)

    k = cv2.waitKey(100) & 0xff # Pressionar ESC para sair da camera
    if k == 27:
        break
    elif contador >= 70: # Tira 70 amostras do rosto e para a webcam
         break

# finalização
print("\n Saindo do programa...")
# Encerra a captura dos quadros
camera.release()
cv2.destroyAllWindows()


