# Disaster Response Pipeline Project

## Table of Contents

1. [Project Motivation](#motivation)
2. [Installation](#installation)
3. [Licensing, Authors, and Acknowledgements](#licensing)

## Project Motivation<a name="motivation"></a>
In this project I use the [datasets](https://github.com/evansdoe/disaster-response-pipeline/tree/main/data) from real life disasters provided by [Figure Eight](https://appen.com/) to perform the following tasks:

1. Build an ETL (Extract, Transform, Load) Pipeline to repair the data.

2. Build a supervised learning model using a machine learning Pipeline.

3. Build a web app that does the following:

    - Takes an input message and gets the classification results of the input in several categories.
    - Displays visualisations of the training datasets.

## Installation <a name="installation"></a>
The installation instructions are as follows.

### Dependencies
* [Python (>=3.6)](https://www.python.org/downloads/)
* [Pandas](https://pandas.pydata.org/)
* [Numpy](https://numpy.org/)
* [Sqlalchemy](https://www.sqlalchemy.org/)
* [Sys](https://docs.python.org/3/library/sys.html)
* [Plotly](https://plotly.com/python/)
* [Sklearn](https://sklearn.org/)
* [Joblib](https://joblib.readthedocs.io/en/latest/)
* [Flask](https://flask.palletsprojects.com/en/2.0.x/)

### Download and Installation
```console
foo@bar:~ $ git clone https://github.com/evansdoe/disaster-response-pipeline.git
foo@bar:~ $ cd disaster-response-pipeline
foo@bar:disaster-response-pipeline $  
```
While in the project's root directory `disaster-response-pipeline` run the ETL pipeline that cleans and stores data in database.
```console
foo@bar:disaster-response-pipeline $ python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```
Next, run the ML pipeline that trains the classifier and saves it.
```console
foo@bar:disaster-response-pipeline $ python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
```

Next, change directory into the `app` directory and run the Python file `run.py`.
```console
foo@bar:disaster-response-pipeline $ cd app
foo@bar:app $ python run.py
```

Finally, go to http://0.0.0.0:3001/ or http://localhost:3001/ in your web-browser.

Type a message input box and click on the `Classify Message` button to see how the various categories that your message falls into.


## Licensing and Acknowledgements<a name="licensing"></a>

Big credit goes to [Figure Eight](https://appen.com/) for the relabelling the datasets and also to the teaching staffs at [Udacity](https://www.udacity.com/). Finally, the repository is distributed under the [GNU GPLv3 license](https://github.com/evansdoe/stackoverflow_2020_survey/blob/main/LICENSE).
