# Document Classification

### Set up with virtualenv
```
virtualenv -p python3.6 venv
source venv/bin/activate
pip install -r requirements.txt
python setup.py develop
python document_classification/application.py
```

### Set up with Dockerfile
```
```

### API endpoints
- Health check `GET /api`
```bash
curl --header "Content-Type: application/json" \
     --request GET \
     http://localhost:5000/
```

- Training `POST /train`
```bash
curl --header "Content-Type: application/json" \
     --request POST \
     --data '{"config_filepath": "/Users/goku/Documents/productionML/document_classification/configs/train.json"}' \
     http://localhost:5000/train
```

- List of experiments `GET /experiments`
```bash
curl --header "Content-Type: application/json" \
     --request GET \
     http://localhost:5000/experiments
```

- Experiment info `GET /experiment_info/<experiment_id>`
```bash
curl --header "Content-Type: application/json" \
     --request GET \
     http://localhost:5000/info/latest
```

- Delete an experiment
```bash
curl --header "Content-Type: application/json" \
     --request GET \
     http://localhost:5000/delete/1545593561_8371ca74-06e9-11e9-b8ca-8e0065915101
```

### Content
- **datasets**:
- **document_classification**:
    - **api**:
        - *api.py*:
        - *utils.py*:
    - **config**:
        - *train.json*:
        - *infer.json*:
    - **experiments**:
    - **logs**:
        - *flask.log*:
        - *ml.log*:
    - **ml**:
        - *dataset.py*:
        - *inference.py*:
        - *load.py*:
        - *model.py*:
        - *preprocess.py*:
        - *split.py*:
        - *training.py*:
        - *utils.py*:
        - *vectorizer.py*:
        - *vocabulary.py*:
    - *application.py*:
    - *config.py*:
    - *utils.py*:
- *.gitignore*:
- *Dockerfile*:
- *LICENSE*:
- *requirements.txt*:
- *setup.py*:



