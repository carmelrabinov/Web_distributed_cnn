# Web-Distributed convolutional neural networks 

## Abstruct
The purpose of this project is to minimize the training time of a convolutional neural network using data parallelism techniques, by distributing data over the Internet web to multiple machines that will contribute to the training process.
This project was performed in the Networked Software System Laboratory in the Electrical Engineering faculty in the Technicon, and was written by Amir Livne and Carmel Rabinovitz as part of project A curse.   

## Project results:
The project was implemented using the CIFAR10 dataset and all results are with respect to this dataset.
The weight og the cnn were updated using async SGD method as in the following figure:

![alt text](https://raw.githubusercontent.com/carmelrabinov/Web_distributed_cnn/master/pictures/async_sgd.png)

We achived best performance using 3 client as seen in the following figures:

![alt text](https://raw.githubusercontent.com/carmelrabinov/Web_distributed_cnn/master/results/graphs/accuracy_per_time.png)

![alt text](https://raw.githubusercontent.com/carmelrabinov/Web_distributed_cnn/master/results/graphs/accuracy_per_epoch.png)

![alt text](https://raw.githubusercontent.com/carmelrabinov/Web_distributed_cnn/master/results/graphs/time_to_70.png)

for detail results please read the PDF document

## Installation:
1. install docker for windows or rabbitmq-server for Ubuntu
2. set your python env to 3.5
3. install tensorflow
4. install keras 2.0.5 and above
5. install pika

## Getting started:
1. if runing on windows: run the following command in powershell:
	docker run -d -p 5672:5672 -p 15672:15672 rabbitmq:management
2. open propmt (can use anaconda) and enter python 3.5 env (in anaconda: activate <envname>) and cd to project directory
4. run: python server.py
5. run: python client.py <clinet name> from each client machine

RabbitMQ admin page:
1. localhost:15672
2. user/password: admin/admin or guest/guest