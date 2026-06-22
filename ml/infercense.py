
from ml.model import load_model

model = load_model(r"model/model2.pkl")


def inferance_model1(input_data: list[float]):
    
    y_pred = model.predict(input_data)
    return y_pred
