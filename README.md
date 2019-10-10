# factoid-QA

## Setup:
- Install dependencies `pip install -r dependencies.txt`

## Architecture:
- Conv layer
- ReLU activation
- Pool layer
- Conv layer
- ReLU activation
- Pool layer
- Fully connected layer
- Softmax layer

## Execute:
`python predict.py` for prediction

## Keras REST API :
- run the server `run_keras_server.py`
- Submit a request via cURL `curl -X POST -F image=@imagename.jpg 'http://localhost:5000/predict'`
- Submita a request via Python `python simple_request.py`
