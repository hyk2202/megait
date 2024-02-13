# 경고메시지 출력 off
import warnings
warnings.filterwarnings(action='ignore')

# ---------------------
# 기본 라이브러리 참조
# ---------------------
# 파이썬 기본 시스템 모듈 -> 그래프 초기화 과정에서 OS종류에 따른 글꼴 선택을 위함
import sys

# 수치 계산 및 배열 자료형 제공 모듈
import numpy as np

# 데이터 시각화 관련 모듈
import seaborn as sb
from matplotlib import pyplot as plt

# 데이터 프레임
from pandas import read_csv, read_excel, DataFrame, melt, pivot_table, concat, merge, MultiIndex

# 데이터 프레임을 테이블 형태로 출력
from tabulate import tabulate        

# 머신러닝을 위한 데이터 분할 기능
from sklearn.model_selection import train_test_split

# 데이터 표준화
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# -----------------------------
# 선형회귀 관련 라이브러리 참조
# -----------------------------

# 선형회귀모델
from sklearn.linear_model import LinearRegression

# 성능 평가 관련 기능
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

#  결과보고 관련 기능
from sklearn.feature_selection import f_regression
from scipy.stats import t, f 
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

# 회귀분석을 수행하는 통계패키지
from statsmodels.formula.api import ols



# -----------------
# 그래프 초기 설정
# -----------------
plt.rcParams["font.family"] = 'AppleGothic' if sys.platform == 'darwin' else 'Malgun Gothic'
plt.rcParams["font.size"] = 9
plt.rcParams["figure.figsize"] = (15, 6)
plt.rcParams["axes.unicode_minus"] = False

print('Library load fin :)')