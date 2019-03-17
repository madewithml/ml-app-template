# {{ cookiecutter.service_name }}

This repository was made using the [practicalAI boilerplate](https://github.com/practicalAI/boilerplate) template.

### Directory structure
```
{{ cookiecutter.service_name }}/
├── src/                                - source files
|   ├── api/                              - holds all API scripts
|   |   ├── endpoints.py                    - API endpoint definitions
|   |   └── utils.py                        - utility functions for endpoints
|   ├── configs/                          - configuration files
|   |   ├── logging.json                    - logger configuration
|   |   └── training.json                   - training configuration
|   ├── datasets/                         - directory to hold datasets
|   |   └── news.csv                        - data file
|   ├── tests/                            - tests
|   |   ├── e2e/                            - integration tests
|   |   ├── unit/                           - unit tests
|   ├── {{ cookiecutter.package_name }}/  - ML files
|   |   ├── dataset.py                      - dataset
|   |   ├── model.py                        - model functions
|   |   ├── utils.py                        - utility functions
|   |   ├── vectorizer.py                   - vectorize the processed data
|   |   └── vocabulary.py                   - vocabulary to vectorize data
|   ├── application.py                    - application script
|   ├── config.py                         - application configuration
|   ├── requirements.txt                  - python package requirements
|   ├── setup.py                          - custom package setup
|   ├── wsgi.py                           - application initialization
├── .dockerignore                       - dockerignore file
├── .gitignore                          - gitignore file
├── Dockerfile                          - Dockerfile for the application
├── CODE_OF_CONDUCT.md                  - code of conduct
├── CODEOWNERS                          - code owner assignments
├── LICENSE                             - license description
└── README.md                           - repository readme
```
