# {{cookiecutter.service_name}}

### Content
- **📁 datasets**: directory to hold datasets
- **📁 {{cookiecutter.service_name}}**:
    - *📄 application.py*: application script
    - *📄 config.py*: application configuration
    - *📄 utils.py*: application utilities
    - **📁 api**: holds all API scripts
        - *📄 api.py*: API call definitions
        - *📄 utils.py*: utility functions
    - **📁 config**: configuration files
        - *📄 train.json*: training configurations
        - *📄 infer.json*: inference configurations
    - **📁 ml**:
        - *📄 dataset.py*: dataset/dataloader
        - *📄 inference.py*: inference operations
        - *📄 load.py*: load the data
        - *📄 model.py*: model architecture
        - *📄 preprocess.py*: preprocess the data
        - *📄 split.py*: split the data
        - *📄 training.py*: train the model
        - *📄 utils.py*: utility functions
        - *📄 vectorizer.py*: vectorize the processed data
        - *📄 vocabulary.py*: vocabulary to vectorize data
- *📄 .gitignore*: gitignore file
- *📄 LICENSE*: license of choice (default is MIT)
- *📄 requirements.txt*: python package requirements
- *📄 setup.py*: custom package setup