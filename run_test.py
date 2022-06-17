from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
import sys
from matplotlib import pyplot as plt

img_path = sys.argv[1]

pred_model = load_model("wildlifeClassifier.h5")

with open('classes.txt') as f:
    file_content = f.readlines()

classes = [x.strip() for x in file_content]


img = image.load_img(img_path,target_size=(300,300,3))

x = image.img_to_array(img)
x = np.expand_dims(x,axis=0)

final_image = np.vstack([x])
pred_class = pred_model.predict(final_image,batch_size=1)
pred_index = np.argmax(pred_class[0])
print(f' Input Animal is ======== >> {classes[pred_index]}')



plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
im = plt.imread(img_path)
fig, ax = plt.subplots()
im = ax.imshow(im, extent=[0, 300, 0, 300])
plt.show()