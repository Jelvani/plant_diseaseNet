# plant_diseaseNet
## Attempt at Classifiying Plant Diseases
This repo contains a plant disease classifier, trained using tensorflow and a CNN model.
***
## Dataset
The used dataset, not included in this repo due to large size, can be found [here](https://www.kaggle.com/vipoooool/new-plant-diseases-dataset)

## Getting Started
- Usage examples can be found in [sample.py](sample.py)
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

## Deployment to Heroku

*As of 1/2/2021, specifying `tensorflow>=2` in `requirements.txt` is not compatible with Heroku in my testing. Upon importing tensorflow, the python process terminates from a `SIGILL` signal. You can either use a version of `tensorflow<2`, or if you want to use tensorflow 2, use `tf-nightly`. I use the latter.*

[Heroku](https://www.heroku.com/about) is cloud hosting platform that we can use to host our web application and server. Installation instructions for the Heroku CLI can be found [here](https://devcenter.heroku.com/articles/heroku-cli).
- Upon installation and login, a user can navigate to the root folder of this repo, and run:

    ```sh
    #Create Heroku app and add its remote git repo to our local one
    heroku create
    #now we simply push our repo to the Heroku app
    git push heroku main
    ```
    Your app can be accessed by the randomly generated URL under the Heroku domain.

## File Structure

Here I will briefly discuss what some of the files are doing in the repo.
***
* `Procfile`
    * A plain text file that tells Heroku what to launch on startup. As we would run our application locally by running `app.py`, this is the executable we also specify to run in our Procfile.
* `requirements.txt`
    * A plain text file that contains all of our python packages that our app depends on. This is used to install all packages when we create our local python virtual environnement, and conveniently, Heroku also uses this file to install the python packages for each app.
* `runtime.txt`
    * A plain text file that contains the python runtime version Heroku should use. It is optional, since Heroku will always assign a default python runtime version, but by using this file, we can lock Heroku to use a certain version.
    