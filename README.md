# Web-Distributed convolutional neural networks 

## Installation:
1. install docker for windows or rabbitmq-server for Ubuntu
2. set your pythonenv to 3.5 (can use anconda)
3. install tensorflow or tensorflow-gpu
4. install keras 2.0.5 and above
5. install pika

## Getting started:
1. if runing on windows: run the following command in powershell:
	docker run -d -p 5672:5672 -p 15672:15672 rabbitmq:management
2. open propmt (can use anaconda) and enter python 3.5 env (in anaconda: activate <envname>) and cd to project directory
4. run: python server.py
5. run: python client.py <clinet name>

RabbitMQ admin page:
1. localhost:15672
2. user/password: admin/admin or guest/guest
