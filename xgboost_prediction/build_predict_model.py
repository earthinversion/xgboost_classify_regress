
import pandas as pd
from sklearn import metrics
from xgboost import XGBClassifier, XGBRegressor, Booster, DMatrix
from sklearn.model_selection import train_test_split
import xgboost as xgb
import matplotlib.pyplot as plt
import shap
import os
import numpy as np




## Learning Curve
def plot_learning_curve(params, X_train, X_test, y_train, y_test, num_boost_round=100, eval_metric='logloss'):
    """
    Plot learning curve for XGBoost model with training and test data.

    :param params: Dictionary of XGBoost parameters.
    :param X_train: Training features.
    :param X_test: Test features.
    :param y_train: Training labels.
    :param y_test: Test labels.
    :param num_boost_round: Number of boosting rounds.
    :param eval_metric: Evaluation metric (default is 'logloss'). Others include rmse, mae, and mape, error, etc.
    """
    # Convert data to DMatrix format
    dtrain = DMatrix(X_train, label=y_train)
    dtest = DMatrix(X_test, label=y_test)

    # Define evaluation results dictionary
    evals_result = {}

    # Add eval_metric to params dictionary
    params['eval_metric'] = eval_metric

    # Train the model
    model = xgb.train(params, dtrain, evals=[(dtrain, 'train'), (dtest, 'test')],
                      evals_result=evals_result, num_boost_round=num_boost_round,
                      early_stopping_rounds=10, verbose_eval=False)

    # Extract learning curves
    epochs = len(evals_result['train'][eval_metric])
    x_axis = range(0, epochs)

    # Plot learning curves
    fig = plt.figure(figsize=(10, 6), dpi=150)
    plt.plot(x_axis, evals_result['train'][eval_metric], label='Train')
    plt.plot(x_axis, evals_result['test'][eval_metric], label='Test')
    plt.xlabel('Boosting Round')
    plt.ylabel(eval_metric.capitalize())
    plt.title(f'XGBoost {eval_metric.capitalize()} Learning Curve')
    plt.legend()

    return fig

# Function to train and save the XGBoost model with SHAP visualization
def train_and_save_model(x_train, y_train, modelpath, params):
    """
    Train an XGBoost model, save it, and visualize SHAP values.

    :param x_train: Training features.
    :param x_test: Test features.
    :param y_train: Training labels.
    :param y_test: Test labels.
    :param modelpath: Path to save the trained model.
    :param params: Dictionary of model parameters.
    :param feature_index: Index of the feature to plot SHAP dependence plot.
    :param plot_type: Type of SHAP plot to display (default is 'shap_summary').
    """
    
    # Train XGBoost model on the entire training set and save
    model = XGBClassifier(**params)
    model.fit(x_train, y_train)

    # Save model in JSON format for compatibility
    model.save_model(modelpath)
    print(f'>> Model saved at {modelpath}')

## SHAP
def plot_shap_values(modelpath, x_test, feature_index=0, plot_type='shap_summary', top_n_features=10, shap_threshold=0.01, filter_features=True):
    """
    Plot SHAP values for the XGBoost model and return the figure if available.

    :param modelpath: Path to the trained XGBoost model.
    :param x_test: Test dataset features.
    :param feature_index: Index of the feature for dependence and interaction plots.
    :param plot_type: Type of SHAP plot ('shap_summary', 'shap_force', 'shap_dependence', or 'shap_interaction').
    :param top_n_features: Number of top features to display (default is 10).
    :param shap_threshold: Minimum mean absolute SHAP value for a feature to be included in the plot.
    """
    # Load the trained model using XGBClassifier
    model = XGBClassifier()
    model.load_model(modelpath)

    # Convert x_test to DataFrame if it is an array, and add feature names if necessary
    if isinstance(x_test, np.ndarray):
        x_test = pd.DataFrame(x_test, columns=[f"Feature_{i}" for i in range(x_test.shape[1])])

    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_test)


    # Filter features based on mean absolute SHAP values
    if filter_features: 
        mean_shap_values = np.abs(shap_values).mean(axis=0)
        important_features = np.where(mean_shap_values > shap_threshold)[0]
        filtered_x_test = x_test.iloc[:, important_features]
        filtered_shap_values = shap_values[:, important_features]
    else:
        filtered_x_test = x_test
        filtered_shap_values = shap_values

    fig = None  # Initialize figure as None

    if plot_type == 'shap_summary':
        # Summary Plot (top N features global importance)
        print(f">> SHAP Summary Plot (Top {top_n_features} Global Feature Importance):")
        # shap.summary_plot(
        #     filtered_shap_values, 
        #     filtered_x_test, 
        #     plot_type="layered_violin", 
        #     max_display=top_n_features, 
        #     color='coolwarm'
        # )

        ## simple summary plot
        # shap.summary_plot(filtered_shap_values, filtered_x_test, max_display=top_n_features)
        shap.summary_plot(filtered_shap_values, filtered_x_test, max_display=top_n_features, plot_type="violin")

        fig = plt.gcf()  # Get the current figure

    elif plot_type == 'shap_force':
        # Force Plot (individual prediction for the first instance)
        print(">> SHAP Force Plot (Individual Prediction):")
        force_plot = shap.force_plot(explainer.expected_value, filtered_shap_values[0, :], filtered_x_test.iloc[0, :])

        # Save to HTML file for visualization outside Jupyter Notebook
        html_path = "shap_force_plot.html"
        shap.save_html(html_path, force_plot)
        print(f"Force plot saved to {html_path}")
        return html_path  # Return the path instead of a figure

    elif plot_type == 'shap_dependence':
        # Dependence Plot for a specific feature
        print(f">> SHAP Dependence Plot for feature {feature_index}:")
        shap.dependence_plot(feature_index, filtered_shap_values, filtered_x_test)
        fig = plt.gcf()

    elif plot_type == 'shap_interaction':
        # Interaction Plot for a specific feature
        print(f">> SHAP Interaction Plot for feature {feature_index}:")
        shap.dependence_plot(feature_index, filtered_shap_values, filtered_x_test, interaction_index='auto')
        fig = plt.gcf()

    return fig


# # Function to train and save the XGBoost model
# def train_and_save_model(x_train, x_test, y_train, y_test, modelpath, params, output_learning_curve=False):
    
#     if output_learning_curve:
#         # Plot the learning curve
#         fig = plot_learning_curve(params, x_train, x_test, y_train, y_test, num_boost_round=100)

#     # Train XGBoost model on the entire training set and save
#     model = XGBClassifier(**params)
#     model.fit(x_train, y_train)

#     # Save model in JSON format for compatibility
#     model.save_model(modelpath)
#     print(f'>> Model saved at {modelpath}')

#     if output_learning_curve:
#         return fig

def get_model_prediction_multiclass(modelpath, x_test, y_test, feature_count):
    # Load the trained model
    model = Booster()
    model.load_model(modelpath)  # Load model using Booster

    # Convert x_test to DMatrix for compatibility with Booster's predict method
    dmatrix_test = DMatrix(x_test)

    # Predict for multiclass: directly get predicted classes
    y_test_pred = model.predict(dmatrix_test)

    # Ensure the prediction output is in the expected format
    if y_test_pred.ndim > 1:
        # If y_test_pred is a 2D array, take the argmax across columns to get class labels
        y_test_pred = np.argmax(y_test_pred, axis=1)
    else:
        # Otherwise, cast to int if necessary
        y_test_pred = y_test_pred.astype(int)
    
    # Ensure that y_test is also a 1D array for comparison
    y_test = y_test.ravel()

    # Calculate accuracy and misclassified samples
    test_accuracy = metrics.accuracy_score(y_test, y_test_pred)
    print(f'>> XGBoost Test Accuracy ({feature_count} features): {test_accuracy:.6f}')
    count_misclassified = (y_test != y_test_pred).sum()
    print(f'>> Misclassified Samples ({feature_count} features): {count_misclassified}')

    return test_accuracy, count_misclassified




def get_model_prediction(modelpath, x_test, y_test, feature_count):
    # Load the trained model
    model = Booster()
    model.load_model(modelpath)  # Load model using Booster

    # Convert x_test to DMatrix for compatibility with Booster's predict method
    dmatrix_test = DMatrix(x_test)

    # Predict
    y_test_pred = (model.predict(dmatrix_test) > 0.5).astype(int)  # For binary classification

    # Calculate accuracy and misclassified samples
    test_accuracy = metrics.accuracy_score(y_test, y_test_pred)
    print(f'>> XGBoost Test Accuracy ({feature_count} features): {test_accuracy:.6f}')
    count_misclassified = (y_test != y_test_pred).sum()
    print(f'>> Misclassified Samples ({feature_count} features): {count_misclassified}')

    return test_accuracy, count_misclassified



# Function to load and predict using the XGBoost model
def predict_with_model(X, y, modelpath, feature_count, params, test_size=0.2, 
                       output_shap_plot=False, plot_type='shap_summary', multiclass=False, 
                       shap_threshold=0.01, filter_features=True):
    
    # Split data into training and testing sets
    # x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # Check if model file exists; train if not
    print(f">> Model file {modelpath} not found. Training model....")
    
    if not os.path.exists(modelpath):
        train_and_save_model(x_train, y_train, modelpath, params)
        
    

    ## Get model prediction
    if multiclass:
        test_accuracy, count_misclassified = get_model_prediction_multiclass(modelpath, x_test, y_test, feature_count)
    else:
        test_accuracy, count_misclassified = get_model_prediction(modelpath, x_test, y_test, feature_count)
    
    if output_shap_plot:
        fig = plot_shap_values(modelpath=modelpath, x_test=x_test, plot_type=plot_type, top_n_features=10, shap_threshold=shap_threshold, filter_features=filter_features)

        # Show the figure if it's a matplotlib figure
        if fig:
            return fig

        # If the output is a file path (for force plot)
        elif isinstance(fig, str):
            print(f"Open the force plot in your browser: {fig}")


def train_and_save_model_regression(x_train, y_train, modelpath, params):
    """
    Train an XGBoost model for regression, save it, and visualize SHAP values.
    """
    # Use XGBRegressor for regression
    model = XGBRegressor(**params)
    model.fit(x_train, y_train)

    # Save the model
    model.save_model(modelpath)
    print(f'>> Model saved at {modelpath}')

def predict_regression_model(X, y, modelpath, feature_count, params, test_size=0.2, output_shap_plot=False, plot_type='shap_summary'):
    """
    Predict using an XGBoost regression model, calculate performance metrics, and optionally display SHAP values.
    
    :param X: Features.
    :param y: Target variable.
    :param modelpath: Path to save/load the model.
    :param feature_count: Number of features.
    :param params: Model parameters.
    :param test_size: Fraction of data for testing.
    :param output_shap_plot: If True, plots SHAP summary.
    :param plot_type: Type of SHAP plot ('shap_summary', 'shap_dependence').
    """
    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # Check if model file exists; train if not
    if not os.path.exists(modelpath):
        print(f">> Model file {modelpath} not found. Training model....")
        train_and_save_model_regression(x_train, y_train, modelpath, params)
    
    # Load the model
    model = XGBRegressor()
    model.load_model(modelpath)
    print(">> Model loaded successfully.")

    # Perform predictions
    y_pred = model.predict(x_test)
    
    # Calculate RMSE
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    print(f'>> Test RMSE: {rmse:.4f}')

    # Optionally, plot SHAP values
    if output_shap_plot:
        explainer = shap.Explainer(model, x_train)
        shap_values = explainer(x_test)

        if plot_type == 'shap_summary':
            print(">> SHAP Summary Plot:")
            shap.summary_plot(shap_values, x_test, plot_type="violin")
        elif plot_type == 'shap_dependence':
            # Plot SHAP dependence plot for a specific feature
            shap.dependence_plot(0, shap_values.values, x_test)  # Modify '0' to change the feature index

    return rmse
