import torch
import numpy as np
import cv2
from datetime import datetime
import pandas as pd

model = torch.hub.load('ultralytics/yolov5', 'custom', path='./best.pt')
 

img = cv2.imread('Image01.jpg')

results = model(img)

if len(results.xyxy[0].numpy()) > 0:
    detection = results.xyxy[0].numpy()  # Converte o tensor para um array NumPy
    
    xmin = int(detection[0][0])
    ymin = int(detection[0][1])
    xmax = int(detection[0][2])
    ymax = int(detection[0][3])
    
    new_img = np.array(img)
    
    # make rectangles in all objects found
    for result in results.xyxy[0].numpy():
        xmin = int(result[0])
        ymin = int(result[1])
        xmax = int(result[2])
        ymax = int(result[3])
        
        new_img = cv2.rectangle(new_img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 4)
    
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    imagem_salva = "imagem_final_"+ timestamp + ".jpg"
    
    cv2.imwrite(imagem_salva, new_img)
    print(results.pandas().xyxy[0])
else:
    print("Nenhuma detecção encontrada.")