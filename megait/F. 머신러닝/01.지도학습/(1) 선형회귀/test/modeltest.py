# %%
import sys
sys.path.append('C:\\Users\\hyk\\Desktop\\산대특\\megait')

from helper.regrassion import *
from helper.util import *
from helper.plot import *
from helper.analysis import *
import joblib
import json
from flask import Flask, request

#%%
# 학습이 완료된 모델 객체
fit = joblib.load('C:\\Users\\hyk\\Desktop\\산대특\\megait\\E. 추론통계(머신러닝)\\04. 지도학습 - 회귀\\test\\mymodel.pkl')


# y=fit.predict([[1,160,1,1]])
# print(y)


# Flask 메인 객체 생성
# -> __name__은 이 소스파일 이름
app = Flask(__name__)

@app.route('/hello/world', methods = ['GET'])
def helloworld():
    # URL에 포함된 변수 추출
    # ?age=40&children=1&smoke=Y/N&ob=Y/N
    age = request.args.get('age')
    children = request.args.get('children')
    smoke = request.args.get('smoke')
    ob = request.args.get('ob')

    # 원하는 형태로 변환
    p_age = int(age)**2
    p_children = int(children)
    p_smoke = 1 if smoke == "Y" else 0
    p_ob = 1 if ob == "Y" else 0

    # 변환된 내용을 활용해서 예측값 얻기
    y = fit.predict([[p_age,p_children,p_smoke,p_ob]])
    return json.dumps({'y':float(y)})

if __name__ == "__main__": 
    app.run(host='127.0.0.1', port = 3001, debug= True)