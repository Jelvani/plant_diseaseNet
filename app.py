#import train
import tensorflow as tf
import os
from pathlib import Path
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

app = Flask(__name__,template_folder='templates')
'''
app.config['UPLOAD_FOLDER'] = (os.path.join(Path(__file__).parent.resolve(),'uploads'))


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

#save file to folder from POST request
@app.route('/',methods = ['GET', 'POST'])
def upload():
    #if we are sent an image
    if request.method == 'POST':
        if 'file' in request.files:
            f = request.files['file']
            filename = secure_filename(f.filename)
            if filename == '':
                return render_template("index.html")
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            plant = train.plant_classify()
            plant.getData()
            model = plant.loadModel('model1.h5')
            image = plant.loadImage(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return render_template("index.html", name = plant.feed(model,image))
    return render_template("index.html")
'''
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 33507))
    print(port)
    app.run(host="0.0.0.0", debug=True,port=port)
