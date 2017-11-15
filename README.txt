project installation:
	1. install docker for windows
	2. install python 3.5 (can use anconda)
	3. install keras
	4. install pika

run:
	1. run the following command in powershell:
		docker run -d -p 5672:5672 -p 15672:15672 rabbitmq:management
	2. open propmt (can use anaconda) and enter python 3.5 env (in anaconda: activate <envname>)
	3. cd to project directory
	4. run: python server.py [-dataset, defult=cifar10] [-logPrint] [-noAdmin] [-fn: file name dest] [-test: how mant batch to save each time]
	5. run: python client.py [name] [-host, defult=loclhost] [-logPrint]

RabbitMQ admin page
===================
localhost:15672

user/password: guest/guest



