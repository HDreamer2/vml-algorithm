import random

import requests
import torch
import pandas as pd

from Constant import *


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


def logistic_model(X, w, b):
    return torch.sigmoid(torch.matmul(X, w) + b)

def transfer_data(epoch, w, b, loss, feature_weights):
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
        'loss': loss.item(),
        'featureWeights': feature_weights.detach().numpy().tolist()
    }
    response = requests.post(LOGISTIC_REGRESSION_GET_EPOCH_DATA, json=data)



def LogisticRegression(csv_file, features, label, epochs, learn_rate = 0.0005, batch_size = 5):
    model = logistic_model
    samples, labels = read_data_set(csv_file, features, label)

    w = torch.normal(0, 0.01, size=(len(features), 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    loss = torch.nn.BCELoss()

    trainer = torch.optim.SGD([w, b], lr=learn_rate)

    feature_weights = torch.zeros(w.shape)
    for epoch in range(epochs):
        for X, Y in read_batch_data(batch_size, samples, labels):
            l = loss(model(X, w, b), Y)
            trainer.zero_grad()
            l.backward()
            trainer.step()

        # 最后一轮epoch的时候
        if epoch == epochs - 1:
            feature_weights = abs(w) / sum(abs(w))
        l = loss(model(samples, w, b), labels)
        transfer_data(epoch + 1, w, b, l, feature_weights)


# data = pd.read_csv("./uploads/random_data.csv")
# l1 = ["feature_0", "feature_1", "feature_2", "feature_3", "feature_4"]
# l2 = ["label"]
# LogisticRegression(data, l1, l2, 30)
