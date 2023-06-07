import os
import sys
import cv2 as cv
import numpy as np
import torch
import time
import math
import pathlib
import requests
import random as rng
import threading
import win32ui
import win32api
import win32con
import win32gui




cv.ocl.setUseOpenCL(True)
model = cv.dnn.readNet("reloaded.onnx") #credito otorgado a Intelligent Systems Lab Org ,adaptado a partir del modelo midas v2.1 small, enlace https://github.com/isl-org/MiDaS
model.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
model.setPreferableTarget(cv.dnn.DNN_TARGET_OPENCL)


#modelo detector de profundidad


net = cv.dnn_DetectionModel("yolov3.cfg","yolov3.weights") #credito otorgado por el modelo a la organizacion Root Kit, enlace https://github.com/RootKit-Org/AI-Aimbot
net.setPreferableBackend(cv.dnn.DNN_BACKEND_DEFAULT)
net.setPreferableBackend(cv.dnn.DNN_TARGET_CPU)
net.setInputSize(320,320) #notice this is the same size as the images well take from the screen  #es:notese que este es el tamaño aceptado por el modelo descargado
net.setInputScale(1.0/127.5)

#modelo detector de objetivos


vid =cv.VideoCapture("bareimage.avi")
out = cv.VideoWriter("perceptronv1.avi",cv.VideoWriter_fourcc(*"DIVX"),30,(800,600))
frames =int(vid.get(cv.CAP_PROP_FRAME_COUNT))
 
#codigo para correr el modelo en grabaciones 


for frame_idx in range(int(vid.get(cv.CAP_PROP_FRAME_COUNT))):
    start=time.time()
    ret,img = vid.read()
    emptymask = np.zeros((320,320,3),dtype=np.uint8)

    scrsht = cv.resize(img,(320,320))
    classIds,confs,bbox=net.detect(frame=scrsht)
    confs = list(map(float,confs))
    classIds=list(classIds)
    #these steps are the detection and the "score" or confidence on wether or not its right about it being classified correctly

     #es: aqui en este paso es donde se implementa la red neuronal con tal de conseguir objetivos similares a lo que se busca (personajes enemigos humanoides) y las variables determinan su nivel de confianza entre todas las detecciones

    indices= cv.dnn.NMSBoxes(bbox,confs,score_threshold=0.12,nms_threshold=0.2)#filters out extra or overlaying target boxes based on confidence values and sizes
 
 #es: interpreta cuadros o objetivos similares que estan superpuestos como uno solo y los filtra con un cierto nivel de tolerancia que se ajusto en este modelo a mano para evitar el "temblor" del recuadro del objetivo

    for i in indices: #this entire for loop serves to filter out non humanoid objects detected and to save the coordinates of the targets as an argument passed on later to a separate function
      
     if classIds[i]==0:
      box=bbox[i]
      x,y,w,h=box
      PROPORTION = (w/h)*6
      if PROPORTION<1:
        PROPORTION=1
      SIZE=math.floor(( w*2 + w*w)**(1./3))
      CENTER=(math.floor(x+(w/2)),y)
      tx=math.floor(x+(w/2))
      ty= y+math.floor(2*PROPORTION)+math.floor(h/(2*SIZE))
      cv.circle(emptymask,(tx,ty),5,(0,255,0),1)
      cv.rectangle(emptymask,(x,y),(x+w,y+h),(0,255,0),1)
      cv.line(emptymask,(x,ty),(x+w,ty),(0,255,0),1)
      cv.line(emptymask,(tx,y),(tx,y+h),(0,255,0),1)

   #dibuja recuados y lineas sobre el objetivo
    emptymask=cv.resize(emptymask,(400,300))
   #transforma la escala de la imagen al tamaño requerido
    edge = cv.convertScaleAbs(img,2,2)
    edge = cv.GaussianBlur(edge,(3,3),0)
    kernel = np.array([[-1, 0, -1],[1,4, 1],[-1, 1, -1]])
    edge= cv.filter2D(edge, -1, kernel)

    
    edge = cv.cvtColor(edge, cv.COLOR_RGB2GRAY)
    edge1 = cv.Canny(edge, 130,130)
    edge1 = cv.resize(edge1,(400,300))
      
    
   #aplica filtros a la imagen

    img1 = cv.resize(img,(256,256),cv.INTER_CUBIC)
    img1 = cv.dnn.blobFromImage(img1,1/255,(256,256),(123.675,116.28,103.53),True, False)
    model.setInput(img1)
    output = model.forward()
    output=np.reshape(output,(256,256))
    output =cv.normalize(output,None,0,1,norm_type =cv.NORM_MINMAX)
    output = cv.cvtColor(output  , cv.COLOR_GRAY2BGR)

   #aplica la percepcion de profundidad al modelo
    
    output = cv.resize(output,(400,300),cv.INTER_AREA)
    

    output2=cv.normalize(output,dst=None,alpha=0,beta=255,norm_type=cv.NORM_MINMAX,dtype=cv.CV_8U)

    
    interest= cv.Canny(output2, 20,20)
    interest2 = cv.bitwise_or(edge1,interest)
    

    img = cv.resize(img,(400,300),cv.INTER_AREA)
    
    UVmap = cv.cvtColor(interest2,cv.COLOR_GRAY2BGR)
    UVmap = cv.applyColorMap(UVmap,cv.COLORMAP_PLASMA)
    
    #transforma la imagen en escala de color de calor


    #combina y aplica mapas de trazado de superficie
    Mastermap = cv.bitwise_xor(UVmap,output2)
    Heatmap = cv.applyColorMap (Mastermap ,cv.COLORMAP_MAGMA)

    #combina percepcion de profundidad con el trazado
    Heatmap = cv.bitwise_or(Heatmap,emptymask)
    stack1 = cv.hconcat([img,output2])
    stack2 = cv.hconcat([UVmap,Heatmap])

    stackF= cv.vconcat([stack1,stack2])
    stackF = cv.normalize(stackF,dst=None,alpha=0,beta=255,norm_type=cv.NORM_MINMAX,dtype=cv.CV_8UC3)

    #corrige el formato de imagen
    print (np.shape(stackF))

    cv.imshow("final comparison",stackF) 
    cv.waitKey(1)
    out.write(stackF)
    end= time.time()-start
    estimated= end*frames
    frames=frames-1
    print(str(estimated)+" seconds aproximately left.")
    #codigo para medir el tiempo estimado de renderizado faltante
out.release()

    
