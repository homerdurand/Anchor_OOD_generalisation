import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.cluster import KMeans


# Define the Conditional Variance Penalty model
class CVPLinearRegression:
    def __init__(self, input_dim, output_dim, lambda_cvp=1.0, n_epochs=1000, learning_rate=0.01, patience=20, tol=1e-4, verbose=False):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lambda_cvp = lambda_cvp  # The regularization strength for conditional variance penalty
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.tol = tol
        self.verbose = verbose
        self.model = nn.Linear(input_dim, output_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def cvp_loss(self, X, Y, envs):
        """
        Compute the Conditional Variance Penalty Loss: risk minimization + conditional variance regularization.
        """
        mse_loss = nn.MSELoss()
        # Standard MSE loss
        loss = mse_loss(self.model(X), Y)
        
        # Conditional variance regularization: penalize high variance across environments
        unique_envs = np.unique(envs)
        for env in unique_envs:
            mask = envs == env
            X_env, Y_env = X[mask], Y[mask]
            
            # Predictions for this environment
            predictions = self.model(X_env)
            
            # Calculate variance of predictions across different environments for the same input
            prediction_variance = torch.var(predictions, dim=0)
            loss += self.lambda_cvp * torch.sum(prediction_variance)

        return loss

    def fit(self, X_train, Y_train, envs_train):
        """
        Train the model using Conditional Variance Penalty with early stopping.
        """
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        Y_tensor = torch.tensor(Y_train, dtype=torch.float32)
        
        # Early stopping parameters
        best_loss = float('inf')
        patience_counter = 0
        
        # Train the model
        for epoch in range(self.n_epochs):
            self.model.train()
            self.optimizer.zero_grad()
            
            # Compute the CVP loss
            loss = self.cvp_loss(X_tensor, Y_tensor, envs_train)
            
            # Backpropagate
            loss.backward()
            self.optimizer.step()
            
            # Calculate MSE for early stopping
            with torch.no_grad():
                Y_pred = self.model(X_tensor)
                mse = mean_squared_error(Y_tensor.numpy(), Y_pred.numpy())
            
            # Early stopping check
            if mse < best_loss - self.tol:
                best_loss = mse
                patience_counter = 0  # Reset counter if MSE improves
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                if self.verbose:
                    print(f"Early stopping at epoch {epoch}/{self.n_epochs} (MSE did not improve).")
                break

            if self.verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}/{self.n_epochs}, Loss: {loss.item():.4f}, MSE: {mse:.4f}")

    def set_sklearn_model(self):
        """
        Set the coefficients and intercept from the learned CVP model to an sklearn LinearRegression model.
        """
        sklearn_model = LinearRegression()
        
        # Set coefficients and intercept in the sklearn model
        with torch.no_grad():
            sklearn_model.coef_ = self.model.weight.detach().numpy()
            sklearn_model.intercept_ = self.model.bias.detach().numpy()

        return sklearn_model
