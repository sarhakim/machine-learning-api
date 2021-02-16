# machine-learning-api
Create a machine learning api to train a model.

To run the ml_api. 

### Install dependencies
```
poetry install --no-root
```
This will create a virtual environment with the dependencies. 

### Run the api
```
python ml_api/api.py --port 5000
```

### Configuration

The api depends on a postgres database configured in ml_api/config.py`. 

### Ref 
- https://www.datacamp.com/community/tutorials/machine-learning-models-api-python
