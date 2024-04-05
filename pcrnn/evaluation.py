import numpy as np
import torch
from sklearn.linear_model import LinearRegression

from pcrnn.models.pcrnn import RNNCell, PCRNN
from pcrnn.models.baselines import BaselineRNN, NaiveForecaster
from pcrnn.data_utils import create_linreg_dataset


def evaluate_model(model, X_test, y_test, true_grad, torch_device, scaler=None):
    """Evaluate model performance on test data.

    Parameters
    ----------
    model
        Model to evaluate, must be some class from wtg-pinn.models
    X_test
        Input test data.
    y_test
        Target test data.
    true_grad
        True rescaled gradients.
    torch_device
        Torch device.
    scaler, optional
        MinMaxScaler class instance used to transform data, default None.

    Returns
    -------
        Numpy arrays of model predictions, squared prediction errors, squared errors of computed gradients
    """

    if issubclass(type(model), torch.nn.Module):
        model.eval()
        # --- Predictions ---
        model_pred = model.predict(torch.tensor(X_test).float().to(torch_device)).detach().cpu().numpy().flatten()
        if scaler:
            model_pred = scaler.inverse_transform(model_pred)
        pred_error = (y_test - model_pred)**2

        # --- Gradients ---
        if model.__class__ == PCRNN:
            computed_grad = model.get_gradients(torch.tensor(X_test).float().to(torch_device)).detach().cpu().numpy().flatten()
            grad_comp_error = (true_grad - computed_grad)**2
        else:
            grad_comp_error = np.zeros_like(pred_error) * np.nan

    elif type(model) == LinearRegression:
        X_test_linreg = create_linreg_dataset(X_test)
        model_pred = model.predict(X_test_linreg)
        if scaler:
            model_pred = scaler.inverse_transform(model_pred)
        pred_error = (y_test - model_pred)**2
        grad_comp_error = np.zeros_like(pred_error) * np.nan

    elif type(model) == NaiveForecaster:
        X_test_linreg = create_linreg_dataset(X_test)
        model_pred = model.predict(X_test_linreg)
        if scaler:
            model_pred = scaler.inverse_transform(model_pred)
        pred_error = (y_test - model_pred)**2
        grad_comp_error = np.zeros_like(pred_error) * np.nan

    return model_pred, pred_error, grad_comp_error
