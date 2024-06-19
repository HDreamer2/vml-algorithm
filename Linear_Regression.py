import pandas as pd
import torch
import random
import numpy as np
# # 获取当前工作目录
# current_working_directory = os.getcwd()
# # 拼接文件路径
# file_path = os.path.join(current_working_directory, 'uploads', 'generated_data.csv')
from Constant import *
import requests

epoch_data = []
def read_data_set(csv_file, features, label):

    samples = csv_file[features].values.tolist()
    labels = csv_file[label].values.tolist()
    samples = torch.tensor(samples, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)

    return samples, labels


def read_batch_data(batch_size, samples, labels):
    num_sample = len(samples)

    indices = list(range(num_sample))

    random.shuffle(indices)

    for i in range(0, num_sample, batch_size):

        batch_indices = torch.tensor(
            indices[i : min(i + batch_size, num_sample)]
        )

        yield samples[batch_indices], labels[batch_indices]


def linear_model(X, w, b):
    return torch.matmul(X, w) + b


def transfer_data(epoch, w, b, loss):
    '''
    need to complete
    :param epoch:
    :param w:
    :param b:
    :param loss:
    :return:
    '''
    data = {
        'epoch': epoch,
        'weights': w.detach().numpy().tolist(),
        'bias': b.detach().numpy().tolist(),
        'loss': loss.item()
    }
    epoch_data.append(data)

    response = requests.post(LINEAR_REGRESSION_GET_EPOCH_DATA, json=data)


def LinearRegression(csv_file, features, label, epochs, learn_rate = 0.0005, batch_size = 5):
    model = linear_model
    samples, labels = read_data_set(csv_file, features, label)

    w = torch.normal(0, 0.01, size=(len(features), 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    loss = torch.nn.MSELoss()

    trainer = torch.optim.SGD([w, b], lr=learn_rate)

    for epoch in range(epochs):
        for X, Y in read_batch_data(batch_size, samples, labels):
            l = loss(model(X, w, b), Y)
            trainer.zero_grad()
            l.backward()
            trainer.step()

        l = loss(model(samples, w, b), labels)
        transfer_data(epoch + 1, w, b, l)


# data = pd.read_csv("C:/Users/admin/Desktop/generated_data.csv")
# l1 = ["Attribute_1", "Attribute_2", "Attribute_3"]
# l2 = ["Label"]
# LinearRegression(data, l1, l2, 30)
