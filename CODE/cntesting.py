from tensorflow.keras.models import load_model
classifier = load_model('saved_model_10.h5')

from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
img1 = image.load_img('C:/_working/9_ LUNG CANCER DETECTIOn_ CNN/DECISION_TREE/test_images/colonca1.jpeg', target_size=(64, 64))
img = image.img_to_array(img1)
img = img/255
# create a batch of size 1 [N,H,W,C]
img = np.expand_dims(img, axis=0)
prediction = classifier.predict(img) #gives all class prob.


#['colon_aca', 'colon_n', 'lung_aca', 'lung_n', 'lung_scc']


colon_aca=prediction[0,0]*100
colon_n=prediction[0,1]*100
lung_aca=prediction[0,2]*100
lung_n=prediction[0,3]*100
lung_scc=prediction[0,4]*100

value1 =print('i am {} % sure about colon_aca '.format(colon_aca))
value2 =print('i am {} % sure about colon_n '.format(colon_n))
value3 =print('i am {} % sure about lung_aca '.format(lung_aca))
value4 =print('i am {} % sure about lung_n '.format(lung_n))
value5 =print('i am {} % sure about lung_scc '.format(lung_scc))


colon_aca='colon_aca= '+str(colon_aca)
colon_n='colon_n= '+str(colon_n)
lung_aca='lung_aca= '+str(lung_aca)
lung_n='lung_n= '+str(lung_n)
lung_scc='lung_scc= '+str(lung_scc)

if (prediction[0,0]*100)>55.00:
    plt.text(20, 62,colon_aca,color='red',fontsize=18,bbox=dict(facecolor='white',alpha=0.8))
elif (prediction[0,1]*100)>55.00:
    plt.text(20, 62,colon_n,color='red',fontsize=18,bbox=dict(facecolor='white',alpha=0.8))
elif (prediction[0,2]*100)>55.00:
    plt.text(20, 62,lung_aca,color='red',fontsize=18,bbox=dict(facecolor='white',alpha=0.8))
elif (prediction[0,2]*100)>55.00:
    plt.text(20, 62,lung_n,color='red',fontsize=18,bbox=dict(facecolor='white',alpha=0.8))
elif (prediction[0,2]*100)>55.00:
    plt.text(20, 62,lung_scc,color='red',fontsize=18,bbox=dict(facecolor='white',alpha=0.8))
else:
    plt.text(20, 62,"NONE",color='red',fontsize=18,bbox=dict(facecolor='white',alpha=0.8))
    
plt.imshow(img1)
plt.show()