import train

def training():
    global train
    plant = train.plant_classify()
    train, val = plant.getData()
    model = plant.train(train,val)
    plant.save(model)

def test():
    plant = train.plant_classify()
    plant.getData()
    model = plant.loadModel('model1.h5')
    image = plant.loadImage('C:/Users/Admin/upper-leaf-mold-tomato.jpg')
    print(plant.feed(model,image))
if __name__ == "__main__":
    test()