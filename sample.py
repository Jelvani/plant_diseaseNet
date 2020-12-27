import train

plant = train.plant_classify()
train, val = plant.getData()
model = plant.train(train,val)
plant.save(model)