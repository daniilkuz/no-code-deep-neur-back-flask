from flask import Flask, request
from flask_cors import CORS
from urllib.request import urlopen
import pickle
import time
import random
# import base64
# import data_url
# from process_file import process_file
# from train import neural_network_train, sample_from_model
from train import Model
import json
import numpy

app = Flask(__name__)
# CORS(app)
CORS(app, origins="*")
# CORS(app, origins='https://example.com')

@app.route('/', methods=['POST'])
def upload_file():
    try:
        if request.json['layers'] and len(request.json['layers']) > 0:
            print("layers - ", request.json['layers'])
            layers=[]
            for layer in request.json['layers']:
                layers.append(int(layer['neurons']))
        else:
            return "error, there are no layers", 400
        if request.json['data']:
            fileName = request.json['data']['name']
            data_url = request.json['data']['file']
            with urlopen(data_url) as response:
                file = response.read()
            with open('files/'+fileName, 'wb') as f:
                f.write(file)
            if request.json['params'] and len(request.json['params'])>0:
                params = request.json['params']
            
            words = open('files/'+fileName, "r").read().splitlines() #still don´t know how to use without saving
            # modelName = neural_network_train(name = fileName, layers = layers)
            
            # C, layers, itos = neural_network_train(name = fileName, layers = layers)
            # result = sample_from_model(C, layers, itos)
            


            m = Model()
            if params:
                result = m.neural_network_train(words = words, neurons_per_layer = layers,
                                                         block_size = int(params['Block size']), 
                                                         n_embd=int(params['Number of embeddings']),
                                                         max_steps = int(params['Number of iterations']), 
                                                         batch_size=int(params['Batch size']))                    
            else:
                print("train with default params")
                result = m.neural_network_train(words = words, neurons_per_layer = layers, max_steps = 10)
            name = m.save_model()
            result["name"] = name
            # m = Model(C=C, layers=layers, itos=itos)
            # result = m.sample_from_model(n_samples=10)
            # result = {'name': name, 'train_loss': train_loss, 'val_loss': val_loss}
            # print(type(layers))
            # modelName = "model" + generateId() + ".pickle"
            print('result: ', result)
        else:
            return "error, the file is missing!", 400
            
        return result, 200
        
    except Exception as e:
        print("caught error: ", e)
        return "500 Internal Server Error: Oops, something went wrong. We're sorry, but our system encountered an unexpected error that prevented us from fulfilling your request. You did nothing wrong, and we're working to resolve the issue as quickly as possible. Please retry the request in a few minutes or contact our website administrator if the issue persists. Thank you for your patience.", 500


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.json['data']:
            data = request.json['data']
            with open(data, 'rb') as f:
                model = pickle.load(f)
            # print(model)
            C  = model["C"]
            layers = model["layers"]
            itos = model['itos']
            block_size = model['block_size']
            m = Model(C=C, layers=layers, itos=itos, block_size = block_size)
            result = m.sample_from_model(n_samples=20)
            return result, 200
        else:
            return "error, there is no model", 400

    except Exception as e:
        print("caught error: ", e)
        return "500 Internal Server Error: Oops, something went wrong. We're sorry, but our system encountered an unexpected error that prevented us from fulfilling your request. You did nothing wrong, and we're working to resolve the issue as quickly as possible. Please retry the request in a few minutes or contact our website administrator if the issue persists. Thank you for your patience.", 500

if __name__ == '__main__':
    app.run()