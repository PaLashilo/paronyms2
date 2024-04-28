import joblib
import uvicorn
import inference
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import urllib.parse


# App creation and model loading
app = FastAPI()

html_form = """
<!DOCTYPE html>
<html>
<head>
    <title>Предсказание слов</title>
    <meta charset="utf-8">
</head>
<body>
    <h1>Предсказание слов</h1>
    <form id="predictionForm" action="/predict" method="post" encoding="utf-8">
        <label for="word1">Слово 1:</label>
        <input type="text" id="word1" name="word1"><br><br>

        <label for="word2">Слово 2:</label>
        <input type="text" id="word2" name="word2"><br><br>

        <input type="submit" value="Предсказать">
    </form>
    <div id="prediction"></div>
    <script>
        const form = document.getElementById('predictionForm');
        console.log(1);
        form.addEventListener('submit', async (event) => {
            event.preventDefault();
            const formData = new FormData(form);
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            console.log(2);
            const data = await response.json();
            console.log(data);
            document.getElementById('prediction').innerText = data.prediction;
        });
    </script>
</body>
</html>
"""

class ClassificationModel(BaseModel):
    """
    Input features validation for the ML model
    """
    word1: str
    word2: str


@app.post('/predict')
def predict(word_pair: dict):
    """
    :param word_pair: input data from the post request
    :return: predicted result
    """
    model_data = ClassificationModel(**word_pair)
    prediction = inference.predict(urllib.parse.unquote(model_data.word1), urllib.parse.unquote(model_data.word2))
    return {
        "prediction": prediction
    }


@app.get("/", response_class=HTMLResponse)
async def read_form():
    return HTMLResponse(content=html_form, status_code=200)


if __name__ == '__main__':
    # Run server using given host and port
    uvicorn.run(app, host='127.0.0.1', port=80)


