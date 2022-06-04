
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
from engine import reconhece_face, get_rostos
import face_recognition as fr

rostos_conhecidos, nomes_dos_rostos = get_rostos()

#inicio da detecção de mascara
def detectaMascara(frame, detectFace, detectMasc):

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    detectFace.setInput(blob)
    detections = detectFace.forward()

    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > args["confidence"]:

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            
#converte a cor das imagens
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:

        preds = detectMasc.predict(faces)

    return (locs, preds)


ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
                default="face_detector",
                help="caminho para o diretório do modelo do detector facial")
ap.add_argument("-m", "--model", type=str,
                default="mask_detector.model",
                help="caminho para o modelo de detector de máscara facial treinado")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="probabilidade mínima para filtrar detecções fracas")
args = vars(ap.parse_args())

print("[INFO] carregando modelo de detector de rosto...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
                                "res10_300x300_ssd_iter_140000.caffemodel"])
detectFace = cv2.dnn.readNet(prototxtPath, weightsPath)


print("[INFO] carregando modelo de detector de máscara facial...")
detectMasc = load_model(args["model"])


print("[INFO] iniciando fluxo de vídeo...")

# Especifica a camera que será utilizada para captura
vs = VideoStream(src=0).start()

time.sleep(2.0)

while True:

    frame = vs.read()
    frame = imutils.resize(frame, width=800)
#pegando a localização de faces da camera
    localizacao_dos_rostos = fr.face_locations(frame)
    rosto_desconhecidos = fr.face_encodings(frame, localizacao_dos_rostos)

    (locs, preds) = detectaMascara(frame, detectFace, detectMasc)

#teste de uso de máscara
    for (box, pred) in zip(locs, preds):

        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        label = "Com Mascara" if mask > withoutMask else "Sem Mascara"
        color = (0, 255, 0) if label == "Com Mascara" else (0, 0, 255)

        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 3)
        
#teste para cada rosto que encontrar na câmera
    for (top, right, bottom, left), rosto_desconhecido in zip(localizacao_dos_rostos, rosto_desconhecidos):
        resultados = fr.compare_faces(rostos_conhecidos, rosto_desconhecido)
        print(resultados)
        
#definindo quais a distancia do rostos estão na tela para os que tem no banco
        face_distances = fr.face_distance(
            rostos_conhecidos, rosto_desconhecido)
#pegando o melhor resultado da distancia e apresentando
        melhor_id = np.argmin(face_distances)
        if resultados[melhor_id]:
            nome = nomes_dos_rostos[melhor_id]
#se não tiver nenhum rosto distante do banco então igual a desconhecido
        else:
            nome = "Desconhecido"

        font = cv2.FONT_HERSHEY_COMPLEX

        # Texto 
        cv2.putText(frame, nome, (left + 6, bottom - 6),
                    font, 1, (127, 255, 0), 5)
    
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
#para finalizar
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
