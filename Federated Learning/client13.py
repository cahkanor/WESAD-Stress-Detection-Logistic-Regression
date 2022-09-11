import warnings
import flwr as fl
import numpy as np
import pickle
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

import csv
from sklearn.metrics import precision_recall_fscore_support, precision_score, recall_score, accuracy_score, f1_score

import utils

if __name__ == "__main__":
    client_nr=13
    # Load training
    file_to_read = open("features/training_features"+str(client_nr)+".pickle", "rb")
    features1 = pickle.load(file_to_read)
    file_to_read.close()

    file_to_read = open("features/training_labels"+str(client_nr)+".pickle", "rb")
    labels1 = pickle.load(file_to_read)
    file_to_read.close()

    X_train = features1.reset_index(drop=True)
    y_train = labels1.reset_index(drop=True)

    # Load testing
    file_to_read = open("features/testing_features"+str(client_nr)+".pickle", "rb")
    features1 = pickle.load(file_to_read)
    file_to_read.close()

    file_to_read = open("features/testing_labels"+str(client_nr)+".pickle", "rb")
    labels1 = pickle.load(file_to_read)
    file_to_read.close()

    X_test = features1.reset_index(drop=True)
    y_test = labels1.reset_index(drop=True)
    

    # Create LogisticRegression Model
    model = LogisticRegression(
        penalty="l2",
          # local epoch
        warm_start=True,  # prevent refreshing weights when fitting
    )

    # Setting initial parameters, akin to model.compile for keras models
    utils.set_initial_params(model)

    # Define Flower client
    class MnistClient(fl.client.NumPyClient):
        def get_parameters(self):  # type: ignore
            return utils.get_model_parameters(model)

        def fit(self, parameters, config):  # type: ignore
            utils.set_model_params(model, parameters)
            # Ignore convergence failure due to low local epochs
            #print("Mulaiii fit")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
            print(f"Training finished for round {config['rnd']}")
            
            return list(utils.get_model_parameters(model)), len(X_train), {}

        def evaluate(self, parameters, config):  # type: ignore
            #print("Mulaiii evaluate")
            utils.set_model_params(model, parameters)
            loss = log_loss(y_test, model.predict_proba(X_test))
            accuracy = model.score(X_test, y_test)
            print("Evaluate")
            #print(accuracy)
            y_pred = model.predict(X_test)
            acc=accuracy_score(y_test, y_pred)
            prec=precision_score(y_test, y_pred, pos_label=2)
            rec=recall_score(y_test, y_pred, pos_label=2)
            f1=f1_score(y_test, y_pred, pos_label=2)
            print(acc, prec, rec, f1)
            # open the file in the write mode
            with open('result_client13.csv', 'a') as f:
                writer = csv.writer(f)

                # write a row to the csv file
                writer.writerow([acc, prec, rec, f1])
            #print(loss)
            return loss, len(X_test), {"accuracy": accuracy}

    # Start Flower client
    fl.client.start_numpy_client("localhost:8080", client=MnistClient())
