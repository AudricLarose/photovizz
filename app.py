from flask import Flask,request, jsonify
import keras
import tensorflow as tf
import joblib
import numpy as np
import tensorflow as tf

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()


def tensorQuest(r,g,b):
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # Test model on random input data.
    input_shape = input_details[0]['shape']
    input_data = np.array([[r, g, b]], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    cyphers=[int(data) for data in output_data[0]]
    print(output_data)
    return cyphers



app=Flask(__name__)
@app.route('/')
def index():
    return "hello"

@app.route('/predict',methods=['GET'])
def predict():
    r = int(request.args['r'])
    g = int(request.args['g'])
    b = int(request.args['b'])

    print(r, g, b)
    label = ["Blanc", "Bleu", "Cyan", "Gris", "Jaune", "Marron", "Orange", "Rose", "Rouge", "Vert", "Violet"]
    maxcount=0
    maxiter=0;
    data_result =tensorQuest(r,g,b)
    for i in range (len(label)) :
        if maxcount < int(data_result[i]) :
            maxcount=data_result[i]
            maxiter =i


    return jsonify(prediction=label[maxiter])

if __name__=="__main__":
    app.run(debug=True)