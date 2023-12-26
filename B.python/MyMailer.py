import os.path
from smtplib import SMTP
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart

def sendMail(from_addr, to_addr, subject, content, files=[]):
    content_type = 'plain'
    username = 'hyk2202@gmail.com' # 네이버 : 아이디 , 구글 : 메일주소
    password = 'ppaqsmytiefdsdvb' # 네이버 : 개인 비밀번호 or 애플리케이션 비밀번호(2차 보호 사용시) , 구글 : 앱 비밀번호

    smtp = 'smtp.gmail.com'
    port = 587 # 구글 발송 서버주소와 포트번호 ( 고정값)
    # 네이버 : smtp.naver.com / 465

    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = from_addr
    msg['To'] = to_addr

    msg.attach(MIMEText(content, content_type))

    if files:
        for f in files:
            # 바이너리 형식으로 읽기( 첨부파일이 txt파일이 아닐시 바이너리로 읽어야함)
            # 바이너리로 읽을시 encoding 없음( 이미 바이너리로 변환완료)
            with open(f, 'rb') as a_file:
                basename = os.path.basename(f)
                part = MIMEApplication(a_file.read(),Name = basename)

                part['Content-Disposition'] = 'attachment; filename="%s"' % basename
            msg.attach(part)

    mail = SMTP(smtp)
    mail.ehlo()
    mail.starttls()
    mail.login(username, password)
    mail.sendmail(from_addr, to_addr, msg.as_string())
    mail.quit()

# __name__은 코드를 직접 실행할 경우 __name__이라는 이름의 변수에 "__main__"이라는 값이 대입된다.
# 모듈로 실행되는경우 '해당 모듈의 이름으로 대입한다.'


if __name__ == '__main__':
    from_addr = 'hyk2202@gmail.com' # '이름 <메일주소>' 형식은 발신/수신인의 표시 이름을 설정하는 방법
    # 네이버의 경우 발신자에 한해서 이름을 지정할 수 없다
    to_addr = '메일받는사람 <hanyoul0107@naver.com>'
    subject = '메일 제목'
    content = '메일 내용'
    files = ['hello.txt','world.txt']

    sendMail(from_addr,to_addr,subject,content,files)
    