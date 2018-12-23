# API

### Endpoints
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