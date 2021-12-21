from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from smtplib import SMTP_SSL

SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 465
SMTP_USER = 'scolpigemail@gmail.com'
SMTP_PASSWORD = '' # 자신의 비번을 입력

def send_mail(name, recvs, cc, hidden_cc, contents, attachment=False):
    msg = MIMEMultipart('alternative')

    if attachment:
        msg = MIMEMultipart('mixed')

    msg['From'] = SMTP_USER
    msg['To'] = recvs
    msg['CC'] = cc
    msg['Subject'] = name + '님, 메일이 도착했습니다.'

    text = MIMEText(contents)
    msg.attach(text)

    # if attachment:
    #     from email.mime.base import MIMEBase
    #     from email import encoders
    #
    #     file_data = MIMEBase('application', 'octet-stream')
    #     f = open(attachment, 'rb')

    targets = ','.join((recvs, cc, hidden_cc))
    smtp = SMTP_SSL(SMTP_SERVER, SMTP_PORT)
    smtp.login(SMTP_USER, SMTP_PASSWORD)
    print('login')
    smtp.sendmail(SMTP_USER, targets.split(','),
                  msg.as_string())
    smtp.close()
send_mail('scolpigemail@gmail.com', 'scolpigemail@gmail.com',
          '', '', '테스트입니다.')


