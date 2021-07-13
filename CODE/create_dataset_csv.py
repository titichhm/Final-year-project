# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""



import cv2
import os
import pandas as pd

import numpy as np

# file = open('sample.csv','a')



# file.close()



path_img = "E:/0. freelance_project-latest/5_ LUNG CANCER DETECTIOn_ CNN/dataset/test_set/lung_scc"




list_dir = os.listdir(path_img)

file = open('sample.csv','a')

for i in list_dir:    

    img_read = cv2.imread(path_img +"/" + str(i))
    
    print("----path ",path_img +"/" + str(i)+"\n")
    # img_read = cv2.imread("E:/0. freelance_project-latest/5_ LUNG CANCER DETECTIOn_ CNN/dataset/training_set/colon_aca/colonca1.jpeg")
    
    image = cv2.resize(img_read, (64,64))
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY )
    
    
    x = image.reshape(1,4096)
    
    a = ""
    for i in range(0,len(x[0])-1):
        a = a + str(x[0][i]) + "," + "4"
        
        # if i ==len(x[0])-1:
        #     a = a + "\n"

   
    file.write(a+'\n')
    
file.close()






# for i in range(0,len(x[0])-1):
#     a = a + str(x[0][i]) + "," + "0"
# a = a + "\n" 

# print(a)
    
    
    
    
    # file.write('orange,3')



# cv2.imshow("DISPLAY", image)
# cv2.waitKey()
print(image.shape)


