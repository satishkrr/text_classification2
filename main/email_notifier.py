from collections import namedtuple
from datetime import datetime

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

import csv
import pprint
import smtplib


import logging
logging.info('Script starting...')

from creds import *

#================= Email content=====================#
body =  """

Model: {model_name}  \r\r\n
Epoch: {epoch}  \r\r\n
Accuracy: {accuracy}  \r\r\n
Loss: {loss}  \r\r\n

"""

def send_email(model_name, hpconfig_hash, epoch, accuracy, loss, attachment):

    server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
    status = server.ehlo()
    logging.info('establishing connection... -- {}'.format(status))
    status = server.login(GMAIL_USER, GMAIL_PASSWD)
    logging.info('log in... --{}'.format(status))


    send_to = ','.join(['selva.developer@gmail.com'])
    sent_from = 'selva.developer@gmail.com'

    msg = MIMEMultipart('alternative')
    msg['Subject'] = 'Model Monitor > {} > {} '.format(model_name, hpconfig_hash)
    msg['From'] = sent_from
    msg['To'] = send_to

    msg['Reply-to'] = sent_from
    
    msgText = MIMEText('{}\ncid:{}'.format(body.format(model_name = model_name,
                                                                                        epoch = epoch,
                                                                                        accuracy = accuracy,
                                                                                        loss = loss),
                                                                            attachment), 'html')  
    msg.attach(msgText)

    """
    fp = open(attachment, 'rb')                                                    
    img = MIMEText(str(fp.read()))
    fp.close()
    img.add_header('Content-ID', '<{}>'.format(attachment))
    msg.attach(img)
    """
    logging.info(pprint.pformat(body))
    logging.info('sending the above message to {}'.format(send_to))
    status = server.sendmail(sent_from, send_to.split(','), msg.as_string())
    logging.info('message sent -- {}'.format(status))

    status = server.close()
    logging.info('closing connection... -- {}'.format(status))
