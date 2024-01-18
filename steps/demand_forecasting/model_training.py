import logging
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from zenml import step
from zenml.steps.sklearn import SKLearnBaseStep

class ModelTraining:
    """
    Model training class which trains an XGBoost model on the provided data.
    """

    def __init__(self, X, y) -> None:
        """Initialize the model training class."""
        self.X = X
        self.y = y

    def train_model(self) -> XGBRegressor:
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        # Create an XGBoost model
        model = XGBRegressor()

        # Train the model
        model.fit(X_train, y_train)

        # Evaluate the model
        score = model.score(X_test, y_test)
        logging.info(f"Model Accuracy: {score}")

        return model


@step
def train_model(X, y) -> XGBRegressor:
    """
    Args:
        X: pd.DataFrame or np.ndarray
            Features for training the model.
        y: pd.Series or np.ndarray
            Target variable for training the model.
    Returns:
        model: XGBRegressor
            Trained XGBoost model.
    """
    try:
        model_training = ModelTraining(X, y)
        model = model_training.train_model()
        return model
    except Exception as e:
        logging.error(e)
        raise e
