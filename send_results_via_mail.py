# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 15:42:59 2017

@author: carmelr
"""

from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from smtplib import SMTP
import smtplib
import sys


recipients = ['carmelrab@gmail.com']
emaillist = [elem.strip().split(',') for elem in recipients]
msg = MIMEMultipart()
#msg['Subject'] = str(sys.argv[1])
msg['Subject'] = 'project A test results'
msg['From'] = 'ProjectA.results@gmail.com'
msg['Reply-to'] = 'ProjectA.results@gmail.com'
 
msg.preamble = 'Multipart massage.\n'
 
part = MIMEText("Hi, please find the attached file")
msg.attach(part)

#filename = str(sys.argv[2])
filename = 'D:\\TECHNION\\projectA\\tasks.txt'
part = MIMEApplication(open(filename,"rb").read())

part.add_header('Content-Disposition', 'attachment', filename=filename)
msg.attach(part)
 
server = smtplib.SMTP("smtp.gmail.com:587")
server.ehlo()
server.starttls()
server.login("ProjectA.results@gmail.com", "carmelamir")
 
server.sendmail(msg['From'], emaillist , msg.as_string())