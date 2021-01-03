import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os, numpy as np
from pathlib import Path

#below will disable GPU
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#fixes CUDA issue on GPU
#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
#config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

class plant_classify:
    
    abs_path = Path(__file__).parent.resolve()
    class_names = ['Apple scab', 'Apple Black rot', 'Apple Cedar apple rust', 'Apple healthy',
    'Blueberry healthy', 'Cherry (including sour) Powdery mildew', 'Cherry (including sour) healthy',
    'Corn (maize) Cercospora leaf spot Gray spot', 'Corn (maize) Common rust',
    'Corn (maize) Northern Leaf Blight', 'Corn (maize) healthy', 'Grape Black rot', 'Grape Esca (Black Measles)',
    'Grape Leaf blight (Isariopsis Spot)', 'Grape healthy', 'Orange Haunglongbing (Citrus greening)',
    'Peach Bacterial spot', 'Peach healthy', 'Pepper, bell Bacterial spot', 'Pepper, bell healthy',
    'Potato Early blight', 'Potato Late blight', 'Potato healthy', 'Raspberry healthy', 'Soybean healthy',
    'Squash Powdery mildew', 'Strawberry Leaf scorch', 'Strawberry healthy', 'Tomato Bacterial spot',
    'Tomato Early blight', 'Tomato Late blight', 'Tomato Leaf Mold', 'Tomato Septoria leaf spot',
    'Tomato Spider mites Two-spotted spider mite', 'Tomato Target Spot', 'Tomato Yellow Leaf Curl Virus',
    'Tomato mosaic virus', 'Tomato healthy']
    def __init__(self, folder = "New Plant Diseases Dataset(Augmented)"):
        self.folder = folder

    def getData(self):
        path_train = os.path.join(plant_classify.abs_path,self.folder, "train")
        path_valid = os.path.join(plant_classify.abs_path,self.folder, "valid")
        train = keras.preprocessing.image_dataset_from_directory(path_train)
        self.class_names = self._filterLabels(train.class_names)
        val = keras.preprocessing.image_dataset_from_directory(path_valid)
        #normalize pixel values to [0,1]
        train = train.map(self._process)
        val = val.map(self._process)
        return (train,val)

    #filter class names to something user friendly
    def _filterLabels(self, classes):
        for i, lbl in enumerate(classes):
            lbl = list(dict.fromkeys(lbl.split("_")))
            lbl = list(filter(("").__ne__, lbl))
            lbl = " ".join(lbl)
            classes[i] = lbl
        return classes

    def _process(self,image,label):
        image = tf.cast(image/255. ,tf.float32)
        return image,label

    def train(self,train,validation):
        model = keras.Sequential()
        #we add some convolution layers to find some cool features, and maxpooling to reduce image size
        model.add(layers.Conv2D(16,(3,3),activation='relu',input_shape=(256,256,3)))
        model.add(layers.MaxPool2D(4,4))
        model.add(layers.Conv2D(32,(3,3),activation='relu'))
        model.add(layers.MaxPool2D(4,4))
        model.add(layers.Conv2D(32,(3,3)))
        #make image into a flat array
        model.add(layers.Flatten())
        #some cool dense layers, maybe we get lucky and it can classify the correct plant disease
        model.add(layers.Dense(64,activation='relu'))
        model.add(layers.Dense(32,activation='relu'))
        #final output layer
        model.add(layers.Dense(len(self.class_names)))
        model.build((256,256,3))
        model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

        model.summary()
        #train model
        history = model.fit(train,validation_data=validation,epochs=10)
        return model

    #returns a model, loaded from a file name relative to project root folder
    def loadModel(self,file="model.h5"):
        path = os.path.join(plant_classify.abs_path,file)
        model = tf.keras.models.load_model(path)
        return model

    #returns numpy array of given image path that is resized and normalized
    def loadImage(self,file):
        image = tf.keras.preprocessing.image.load_img(file)
        image = keras.preprocessing.image.img_to_array(image)
        image = keras.preprocessing.image.smart_resize(image,(256,256))
        image = np.array([image])/255.
        return image

    def feed(self, model, image):
        scores = model.predict(image)
        prediction = np.argmax(scores)
        label = plant_classify.class_names[prediction]
        return (scores[0,prediction],label)

    #saves a model to the root directory with the given or default name
    def save(self,model, name = "model"):
        index = 1
        path = os.path.join(plant_classify.abs_path,"{}.h5".format(name))
        while os.path.isfile(path):
            id = "{}{}.h5".format(name,index)
            index+=1
            path = os.path.join(plant_classify.abs_path,id)
        model.save(path)