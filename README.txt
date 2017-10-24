1. install docker for windows
2. run the following command in powershell:
	docker run -d -p 5672:5672 -p 15672:15672 rabbitmq:management

3. open anaconda
4. run the following set of commands:
	activate py35
	pip install pika

5. run the server.py
6. run the client:
	python client.py NAME

admin page
==========
localhost:15672

user/password: guest/guest