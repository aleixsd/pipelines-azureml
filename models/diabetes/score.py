import json
import numpy
from azureml.core.model import Model
from sklearn.externals import joblib


###############################
def init():
    global model
    model_path = Model.get_model_path(model_name = 'sklearn_regression_model')
    model = joblib.load(model_path)

def run(input_data):
    try:
        data = json.loads(input_data)['data']
        data = numpy.array(data)
        result = model.predict(data)
        # you can return any datatype as long as it is JSON-serializable
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error
