FROM ubuntu:16.04

MAINTAINER John C. Hale "john.cody.hale@gmail.com"


RUN apt -y update &&\
    apt -y install python3 python3-pip

RUN python3 -m pip install --upgrade pip

 
ADD ./python_requirements.txt /
RUN python3 -m pip install -r python_requirements.txt

ADD ./server.py /
ADD ./gender_predictor.pkl /
ADD ./height_predictor.pkl /
ADD ./weight_predictor.pkl /

CMD [ "python3", "-u", "./server.py" ]
