FROM python:latest

WORKDIR /home/

COPY train.py /home/ 
COPY inference.py /home/ 
COPY server.py /home/
COPY best_params.json /home/
COPY config.json /home/
COPY Data/paronym.csv /home/Data

COPY requirements.txt /home/

RUN python3 -m pip install -r requirements.txt
RUN python3 ./train.py

EXPOSE 80

CMD ["unicorn", "server:app", "--reload", "--host", "0.0.0.0", "--port", "80"]