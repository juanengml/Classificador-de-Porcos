
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import cv2

model = ResNet50(weights='imagenet')
cap = cv2.VideoCapture("porcos.mp4")

def classifier(image_pig):
 img_path = image_pig
 img = image.load_img(img_path, target_size=(224, 224))
 x = image.img_to_array(img)
 x = np.expand_dims(x, axis=0)
 x = preprocess_input(x)
 preds = model.predict(x)
 return decode_predictions(preds, top=3)[0]



while True:
    ret, frame = cap.read()
    img = cv2.resize(frame,(440,240), interpolation = cv2.INTER_AREA)
    cv2.imshow("frame",frame)
    #cv2.imwrite("frame.jpg",frame)
    #print(classifier("frame.jpg"))
    #cv2.imshow("frame",frame)
    
cap.release()
cv2.destroyAllWindows()
