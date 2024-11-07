from xgboost import plot_importance
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

def plot_feature_importance(modelpath, importance_type='weight', max_num_features=10, title=None, dpi=300):
    # Initialize an XGBClassifier and load the model
    model = XGBClassifier()
    model.load_model(modelpath)  # Load the JSON model

    # Feature importance plot
    plt.figure(figsize=(10, 8), dpi=dpi)
    plot_importance(model, importance_type=importance_type, max_num_features=max_num_features, title=title)
    plt.show()

    return plt

def plot_tree_structure(modelpath, tree_index=0, rankdir='LR', ax=None, dpi=300):
    from xgboost import plot_tree

    # Initialize an XGBClassifier and load the model
    model = XGBClassifier()
    model.load_model(modelpath)  # Load the JSON model

    # Plot the tree structure
    # Create a figure with a higher DPI
    plt.figure(figsize=(10, 8), dpi=dpi)
    plot_tree(model, num_trees=tree_index, rankdir=rankdir, ax=ax)
    plt.show()

    return plt

