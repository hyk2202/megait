class Member:
    def __init__(self,username,email):
        print('----- 생성자가 실행되었습니다. -----')
        self.username =username
        self.email = email
    
    def view_info(self):
        print(f'이름 : {self.username} / 이메일 : {self.email}')
