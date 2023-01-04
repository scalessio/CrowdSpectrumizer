FROM python:3.7
ADD . /Deployment
WORKDIR ./Deployment
RUN pip install tsfresh
RUN pip install -r requirements.txt

EXPOSE 5005
# if we want to start only the container from the image we need to setup the CMD otherwise the docker compose do it fot us
CMD ["python" , "./FlaskApp/TDTCApp.py"]