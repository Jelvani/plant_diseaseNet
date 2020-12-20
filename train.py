import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import matplotlib.pyplot as plt
from pathlib import Path

#below will disable GPU
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#fixes CUDA issue on GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

class plant_classify:
    
    abs_path = Path(__file__).parent.resolve()

    def __init__(self, folder = "New Plant Diseases Dataset(Augmented)"):
        self.folder = folder
        
    def getData(self):
        path_train = os.path.join(plant_classify.abs_path,self.folder, "train")
        path_valid = os.path.join(plant_classify.abs_path,self.folder, "valid")
        data = keras.preprocessing.image_dataset_from_directory(path_train)
        val = keras.preprocessing.image_dataset_from_directory(path_valid)
        #normalize pixel values to [0,1]
        data = data.map(self.__process)
        val = val.map(self.__process)
        return (data,val)

    def __process(self,image,label):
        image = tf.cast(image/255. ,tf.float32)
        return image,label

    def train(self,train,validation):
        classes = train.class_names
        model = keras.Sequential()
        #we add some convolution layers to find some cool features, and maxpooling to reduce image size
        model.add(layers.Conv2D(16,(3,3),input_shape=(256,256,3)))
        model.add(layers.MaxPool2D(4,4))
        model.add(layers.Conv2D(16,(3,3)))
        model.add(layers.MaxPool2D(4,4))
        model.add(layers.Conv2D(16,(3,3)))
        model.add(layers.MaxPool2D(4,4))
        #make image into a flat array
        model.add(layers.Flatten(input_shape=(256, 256, 3)))
        #some cool dense layers, maybe we get lucky and it can classify the correct plant disease
        model.add(layers.Dense(32,activation='relu'))
        model.add(layers.Dense(32,activation='relu'))
        #final output layer
        model.add(layers.Dense(len(classes)))
        model.build((256,256,3))
        model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

        model.summary()
        #train model
        history = model.fit(train,validation_data=validation,epochs=10)
        #save entire trained model to file
        
        return model
    def save(model, name = "model"):
        index = 1
        path = os.path.join(plant_classify.abs_path,name)
        while os.path.isfile(path):
            name = "{}{}".format(name,index)
            index+=1
            

        model.save(os.path.join(,"model.h5"))
    
    
