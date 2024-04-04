import joblib
# import uvicorn
import inference
from fastapi import FastAPI
from pydantic import BaseModel

# App creation and model loading
app = FastAPI()
model = joblib.load("./model.joblib")


class ClassificationModel(BaseModel):
    """
    Input features validation for the ML model
    """
    word1: str
    word2: str


@app.post('/predict')
def predict(word_pair: ClassificationModel):
    """
    :param iris: input data from the post request
    :return: predicted iris type
    """
    features = [[
        word_pair.word1,
        word_pair.word2
    ]]
    prediction = inference.predict
    return {
        "prediction": prediction
    }


if __name__ == '__main__':
    # Run server using given host and port
    uvicorn.run(app, host='127.0.0.1', port=80)