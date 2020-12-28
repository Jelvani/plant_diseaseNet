# plant_diseaseNet
## Attempt at Classifiying Plant Diseases
This repo contains a plant disease classifier, trained using tensorflow and a CNN model.
***
## Dataset
The used dataset, not included in this repo due to large size, can be found [here](https://www.kaggle.com/vipoooool/new-plant-diseases-dataset)

## Getting Started
- Usage examples can be found in [sample.py](sample.pys)
- To get started, it is a good idea to create a python virtual environment using [venv](https://docs.python.org/3/tutorial/venv.html). To create your own virtual environment, open the project folder and run:
    ```sh
    #create the venv
    python3 -m venv env
    #now you can activate the venv within a shell process with:
    #for windows
    env\Scripts\activate.bat
    #for Unix/Mac
    #source env/bin/activate
    ```
    Of course you can proceed without a virtual environment and install all the required packages to your system installation of python. For this, we use a `requirements.txt` file, which contains all the python packages to run this project.
    ```sh
    #install all python dependencies
    pip install -r requirements.txt
    ```