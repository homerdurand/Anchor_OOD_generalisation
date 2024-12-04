import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
import numpy as np


class IRMLinearRegression2:
    def __init__(self, input_dim, output_dim, lambda_irm=1.0, n_epochs=1000, learning_rate=0.01, patience=20, tol=1e-4, verbose=False):
        """
        Multi-output Invariant Risk Minimization Linear Regression.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lambda_irm = lambda_irm
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.tol = tol
        self.verbose = verbose

        # Linear model for multi-output regression
        self.phi = nn.Parameter(torch.eye(input_dim, input_dim))  # Learnable representation
        self.w = nn.Linear(input_dim, output_dim, bias=False)  # Linear predictor
        self.optimizer = optim.Adam([self.phi, self.w.weight], lr=learning_rate)

    def irm_loss(self, X, Y, envs):
        """
        Compute the IRM loss: risk minimization + invariance penalty for multi-output regression.
        """
        mse_loss = nn.MSELoss()
        penalty = 0
        total_loss = 0

        for env in torch.unique(envs):
            mask = envs == env
            X_env, Y_env = X[mask], Y[mask]

            # Transform input with learnable phi
            X_transformed = X_env @ self.phi
            predictions = self.w(X_transformed)

            # Compute environment-specific loss
            error = mse_loss(predictions, Y_env)
            total_loss += error

            # Compute gradient penalty for invariance
            grad = torch.autograd.grad(error, self.w.weight, create_graph=True)[0]
            penalty += grad.pow(2).mean()

        # Combine loss and penalty
        return total_loss + self.lambda_irm * penalty

    def fit(self, X_train, Y_train, envs_train):
        """
        Train the IRM model using early stopping.
        """
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        Y_tensor = torch.tensor(Y_train, dtype=torch.float32)
        envs_tensor = torch.tensor(envs_train, dtype=torch.int64)

        # Early stopping variables
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.n_epochs):
            self.optimizer.zero_grad()
            loss = self.irm_loss(X_tensor, Y_tensor, envs_tensor)
            loss.backward()
            self.optimizer.step()

            # Calculate mean squared error for early stopping
            with torch.no_grad():
                predictions = self.predict(X_train)
                mse = mean_squared_error(Y_train, predictions)

            # Early stopping logic
            if mse < best_loss - self.tol:
                best_loss = mse
                patience_counter = 0  # Reset patience
                best_phi = self.phi.detach().clone()
                best_w = self.w.state_dict()
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                if self.verbose:
                    print(f"Early stopping at epoch {epoch}/{self.n_epochs}.")
                break

            if self.verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}/{self.n_epochs}, Loss: {loss.item():.4f}, MSE: {mse:.4f}")

        # Restore best parameters
        self.phi = nn.Parameter(best_phi)
        self.w.load_state_dict(best_w)

    def predict(self, X):
        """
        Make predictions on new data. Returns NumPy array.
        """
        self.w.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            X_transformed = X_tensor @ self.phi
            predictions = self.w(X_transformed)
        return predictions.numpy()

    def get_representation(self, X):
        """
        Get the learned representation of the input data.
        """
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            return (X_tensor @ self.phi).numpy()
