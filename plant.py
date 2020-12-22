import train
import easygui

plant = train.plant_classify()
plant.getData()
model = plant.loadModel("model.h5")
image = plant.loadImage(easygui.fileopenbox())
prediction = plant.feed(model,image)
print(prediction)