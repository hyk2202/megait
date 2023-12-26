def PRINT():

    if __name__ =='__main__':
        print('이 코드는 직접 실행될 때만 실행됩니다.')
        print(__name__)
    else:
        print('이 코드는 이 파일이 모듈 형태로 호출될 때만 실행됩니다.')
        print(__name__)