
from ml.model import load_model

model = None


def inferance_model1(input_data: list[float]):
    global model
    if model is None:
        model = load_model(r"model/model2.pkl")
    y_pred = model.predict(input_data)
    return y_pred
