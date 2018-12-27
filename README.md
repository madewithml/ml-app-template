# productionML

A cookie cutter template to wrap your machine learning code with REST APIs. Check out the [document-classification](https://github.com/GokuMohandas/document-classification) repository to see an example of how this cookiecutter can be used.

### Set Up
```bash
pip install cookiecutter invoke requests
cookiecutter gh:practicalAI/productionML
```

### Steps
1. Place data in **datasets** or have it in S3, etc.
2. Define machine learning scripts in **ml**.
    1. Load data from datasets using *load.py*.
    2. Split data after loading it using *split.py*.
    3. Preprocess data after splitting it using *preprocess.py*.
    4. Define vocabularies for the data using *vocabulary.py*.
    5. Define the vectorizer that uses the vocabulary to vectorizer the data using *vectorizer.py*.
    6. Define the dataset object that can load the data into batches using *dataset.py*.
    7. Create a model using *model.py*.
    8. Define training objects using *training.py*.
    9. Define inference objects using *inference.py*.
    10. Define training and inference operations in *utils.py* along with other utility functions.
3. Define API scripts in **api**.
    1. Create endpoints in *api.py*.
    2. Define utility functions for the api endpoints in *utils.py*.
4. Define training and inference configuration files in **configs**.
5. Initialize the API in *application.py*.
6. Set up the API configuration in *config.py*.
7. Create utility functions (logging, etc.) in *utils.py*.
8. Define pip requirements in *requirements.txt*.
9. Package the service in *setup.py*.
10. Add unit tests.
