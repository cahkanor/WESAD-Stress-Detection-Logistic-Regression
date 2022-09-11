import flwr as fl
import utils
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from typing import Dict
import pickle
import pandas as pd


def fit_round(rnd: int) -> Dict:
    """Send round number to client."""
    return {"rnd": rnd}


def get_eval_fn(model: LogisticRegression):
    """Return an evaluation function for server-side evaluation."""

    # Load test data here to avoid the overhead of doing it in `evaluate` itself
    
    # Load testing
    file_to_read = open("features/all_testing_features.pickle", "rb")
    features1 = pickle.load(file_to_read)
    file_to_read.close()

    file_to_read = open("features/all_testing_labels.pickle", "rb")
    labels1 = pickle.load(file_to_read)
    file_to_read.close()

    X_test = pd.concat(features1)
    y_test = pd.concat(labels1)

    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # The `evaluate` function will be called after every round
    def evaluate(parameters: fl.common.Weights):
        # Update model with the latest parameters
        utils.set_model_params(model, parameters)
        loss = log_loss(y_test, model.predict_proba(X_test))
        accuracy = model.score(X_test, y_test)
        return loss, {"accuracy": accuracy}

    return evaluate


# Start Flower server for five rounds of federated learning
if __name__ == "__main__":
    model = LogisticRegression()
    utils.set_initial_params(model)
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=15,
        min_fit_clients=15,
        eval_fn=get_eval_fn(model),
        on_fit_config_fn=fit_round,
        fraction_eval=1,
    )
    fl.server.start_server(
        server_address="localhost:8080",
        strategy=strategy,
        config={"num_rounds": 30},
    )
