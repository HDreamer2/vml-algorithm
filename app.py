from flask import Flask, request, jsonify
import os
from Linear_Regression import LinearRegression
import pandas as pd
import json

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


@app.route('/upload-csv', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # 在这里添加对CSV文件的处理逻辑
    # ...

    return jsonify({'message': 'File uploaded successfully', 'file_path': file_path}), 200


@app.route('/linear-regression/train', methods=['POST'])
def linear_regression_train():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    data = request.form

    features = json.loads(data.get('features'))
    label = data.get('label')
    epochs = int(data.get('epoch'))

    data = pd.read_csv(file_path)
    LinearRegression(data, features, label, epochs)

    return jsonify({'message': 'Training started successfully'}), 200

#暂时不用这个接口
@app.route('/linear-regression/get-epoch-data', methods=['GET'])
def get_epoch_data():
    # 假设epoch_data是全局变量或存储在某处
    # global epoch_data
    # return jsonify(epoch_data), 200
    return


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
