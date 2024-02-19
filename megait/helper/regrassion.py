import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

from tabulate import tabulate
from pandas import DataFrame, Series

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.api import het_breuschpagan

from scipy.stats import t, f
from helper.util import my_pretty_table
from helper.plot import my_residplot, my_qqplot

def my_linear_regrassion(x_train : DataFrame, y_train : Series, x_test : DataFrame, y_test : Series, use_plot : bool = True, report=True, resid_test = False, figsize=(10, 5), dpi=200) -> LinearRegression :
    """선형회귀분석을 수행하고 결과를 출력한다.

    Args:
        x_train (DataFrame): 독립변수에 대한 훈련 데이터
        y_train (Series): 종속변수에 대한 훈련 데이터
        x_test (DataFrame): 독립변수에 대한 검증 데이터
        y_test (Series): 종속변수에 대한 검증 데이터
        use_plot (bool, optional): 시각화 여부. Defaults to True.
        report (bool, optional): 회귀분석 결과를 보고서로 출력할지 여부. Defaults to True.
        resid_test (bool, optional): 잔차의 가정을 확인할지 여부. Defaults to False.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 5).
        dpi (int, optional): 그래프의 해상도. Defaults to 200.

    Returns:
        LinearRegression: 회귀분석 모델
    """
    xnames = x_train.columns
    yname = y_train.name
    size = len(xnames)

    # 분석모델 생성
    model = LinearRegression(n_jobs=-1) # n_jobs : 사용하는 cpu 코어의 개수 // -1은 최대치
    fit = model.fit(x_train, y_train)

    expr = f"{yname} = "

    for i, v in enumerate(xnames):
        expr += f"{fit.coef_[i]:0.3f} * {v} + " 

    expr += f"{fit.intercept_:0.3f}" 
    print("[회귀식]")
    print(expr, end="\n\n")

    # 성능지표 저장용 리스트
    result_data = []
    y_train_pred = fit.predict(x_train)
    y_test_pred = fit.predict(x_test)
    target = [[x_train, y_train, y_train_pred], [x_test, y_test, y_test_pred]]

    for i, v in enumerate(target):
        result = {
            "결정계수(R2)": r2_score(v[1], v[2]),
            "평균절대오차(MAE)": mean_absolute_error(v[1], v[2]),
            "평균제곱오차(MSE)": mean_squared_error(v[1], v[2]),
            "평균오차(RMSE)": np.sqrt(mean_squared_error(v[1], v[2])),
            "평균 절대 백분오차 비율(MAPE)": np.mean(np.abs((v[1] - v[2]) / v[1]) * 100),
            "평균 비율 오차(MPE)": np.mean((v[1] - v[2]) / v[1] * 100)
        }
        result_data.append(result)
    
    result_df = DataFrame(result_data, index=["훈련데이터", "검증데이터"])
    my_pretty_table(result_df)
    
    if report:
        my_linear_regrassion_report(fit, x_train, y_train, x_test, y_test)

    if use_plot:
        for i,v in enumerate(xnames):
            fig, ax = plt.subplots(1, 2, figsize=(figsize[0]*2,figsize[1]), dpi=dpi)
            fig.subplots_adjust(hspace=0.3)
            for j,w in enumerate(target):
                sb.regplot(x=w[0][v], y=w[1], ci=95, ax=ax[j], label='관측치')
                sb.regplot(x=w[0][v], y=w[1], ci=0, ax=ax[j], label='추정치')
                ax[j].set_title(f"{'훈련데이터' if j == 0 else '검증데이터'}: {yname} vs {v}")
                ax[j].legend()
                ax[j].grid()

            plt.show()
            plt.close()
    
        # 잔차 가정 확인  
    if resid_test:
        print("\n\n[훈련데이터에 대한 잔차의 가정 확인] ==============================")
        my_resid_test(x_train, y_train, y_train_pred, figsize=figsize, dpi=dpi)
        
        print("\n\n[검증데이터에 대한 잔차의 가정 확인] ==============================")
        my_resid_test(x_test, y_test, y_test_pred, figsize=figsize, dpi=dpi)

    # 도출된 결과를 회귀모델 객체에 포함시킴
    fit.x_train = x_train
    fit.y_train = y_train
    fit.x_test = x_test
    fit.y_test = y_test
    fit.y_train_pred = y_train_pred
    fit.y_test_pred = y_test_pred
    fit.train_resid = y_train - y_train_pred
    fit.test_resid = y_test - y_test_pred

    return fit

def my_linear_regrassion_report(fit : LinearRegression, x_train : DataFrame, y_train : Series, x_test : DataFrame, y_test: Series) -> None:
    """선형회귀분석 결과를 보고한다.    

    Args:
        fit (LinearRegression): 선형회귀 객체
        x_train (DataFrame): 독립변수에 대한 훈련 데이터
        y_train (Series): 종속변수에 대한 훈련 데이터
        x_test (DataFrame): 독립변수에 대한 검증 데이터
        y_test (Series): 종속변수에 대한 검증 데이터
    """

    xnames = x_train.columns
    yname = y_train.name
    size = len(xnames)

    y_train_pred = fit.predict(x_train)
    y_test_pred = fit.predict(x_test)
    target = [[x_train, y_train, y_train_pred], [x_test, y_test, y_test_pred]]
    for i, v in enumerate(target):
        print(f"[{'훈련' if i == 0 else '검증'}데이터에 대한 결과보고]")
        
        target_x, target_y, target_y_pred = v
        
        # 잔차
        resid = target_y - target_y_pred

        # 절편과 계수를 하나의 배열로 결합
        params = np.append(fit.intercept_, fit.coef_)

        # 검증용 독립변수에 상수항 추가
        design_x = target_x.copy()
        design_x.insert(0, '상수', 1)

        dot = np.dot(design_x.T,design_x)   # 행렬곱
        inv = np.linalg.inv(dot)            # 역행렬
        dia = inv.diagonal()                # 대각원소

        # 제곱오차
        MSE = (sum((target_y-target_y_pred)**2)) / (len(design_x)-len(design_x.iloc[0]))

        se_b = np.sqrt(MSE * dia)           # 표준오차
        ts_b = params / se_b                # t값

        # 각 독립수에 대한 pvalue
        p_values = [2*(1-t.cdf(np.abs(i),(len(design_x)-len(design_x.iloc[0])))) for i in ts_b]

        # VIF
        vif = [variance_inflation_factor(target_x, list(target_x.columns).index(v)) for v in target_x.columns]

        # 표준화 계수
        train_df = target_x.copy()
        train_df[target_y.name] = target_y
        scaler = StandardScaler()
        std = scaler.fit_transform(train_df)
        std_df = DataFrame(std, columns=train_df.columns)
        std_x = std_df[xnames]
        std_y = std_df[yname]
        std_model = LinearRegression()
        std_fit = std_model.fit(std_x, std_y)
        beta = std_fit.coef_

        # 결과표 구성하기
        result_df = DataFrame({
            "종속변수": [yname] * size,
            "독립변수": xnames,
            "B(비표준화 계수)": np.round(params[1:], 4),
            "표준오차": np.round(se_b[1:], 3),
            "β(표준화 계수)": np.round(beta, 3),
            "t": np.round(ts_b[1:], 3),
            "유의확률": np.round(p_values[1:], 3),
            "VIF": vif,
        })

        #result_df
        my_pretty_table(result_df)
        
        resid = target_y - target_y_pred        # 잔차
        dw = durbin_watson(resid)               # 더빈 왓슨 통계량
        r2 = r2_score(target_y, target_y_pred)  # 결정계수(설명력)
        rowcount = len(target_x)                # 표본수
        featurecount = len(target_x.columns)    # 독립변수의 수

        # 보정된 결정계수
        adj_r2 = 1 - (1 - r2) * (rowcount-1) / (rowcount-featurecount-1)

        # f값
        f_statistic = (r2 / featurecount) / ((1 - r2) / (rowcount - featurecount - 1))

        # Prob (F-statistic)
        p = 1 - f.cdf(f_statistic, featurecount, rowcount - featurecount - 1)

        tpl = f"𝑅^2({r2:.3f}), Adj.𝑅^2({adj_r2:.3f}), F({f_statistic:.3f}), P-value({p:.4g}), Durbin-Watson({dw:.3f})"
        print(tpl, end="\n\n")

        # 결과보고
        tpl = f"{yname}에 대하여 {','.join(xnames)}로 예측하는 회귀분석을 실시한 결과, 이 회귀모형은 통계적으로 유의{'하다' if p <= 0.05 else '하지 않다'}(F({len(target_x.columns)},{len(target_x.index)-len(target_x.columns)-1}) = {f_statistic:0.3f}, p {'<=' if p <= 0.05 else '>'} 0.05)."

        print(tpl, end = '\n\n')

        # 독립변수 보고
        for n in xnames:
            item = result_df[result_df['독립변수'] == n]
            coef = item['B(비표준화 계수)'].values[0]
            pvalue = item['유의확률'].values[0]

            s = f"{n}의 회귀계수는 {coef:0.3f}(p {'<=' if pvalue <= 0.05 else '>'} 0.05)로, {yname}에 대하여 {'유의미한' if pvalue <= 0.05 else '유의하지 않은'} 예측변인인 것으로 나타났다."

            print(s)
            
        print("")

    # 도출된 결과를 회귀모델 객체에 포함시킴 --> 객체 타입의 파라미터는 참조변수로 전달되므로 fit 객체에 포함된 결과값들은 이 함수 외부에서도 사용 가능하다.
    fit.r2 = r2
    fit.adj_r2 = adj_r2
    fit.f_statistic = f_statistic
    fit.p = p
    fit.dw = dw
        
def my_resid_normality(y: Series, y_pred: Series) -> None:
    """MSE값을 이용하여 잔차의 정규성 가정을 확인한다.

    Args:
        y (Series): 종속변수
        y_pred (Series): 예측값
    """
    mse = mean_squared_error(y, y_pred)
    resid = y - y_pred
    mse_sq = np.sqrt(mse)

    r1 = resid[ (resid > -mse_sq) & (resid < mse_sq)].count() / resid.count() * 100
    r2 = resid[ (resid > -2*mse_sq) & (resid < 2*mse_sq)].count() / resid.count() * 100
    r3 = resid[ (resid > -3*mse_sq) & (resid < 3*mse_sq)].count() / resid.count() * 100

    mse_r = [r1, r2, r3]
    
    print(f"루트 1MSE 구간에 포함된 잔차 비율: {r1:1.2f}% ({r1-68})")
    print(f"루트 2MSE 구간에 포함된 잔차 비율: {r2:1.2f}% ({r2-95})")
    print(f"루트 3MSE 구간에 포함된 잔차 비율: {r3:1.2f}% ({r3-99})")
    
    normality = r1 >= 68 and r2 >= 95 and r3 >= 99
    print(f"잔차의 정규성 가정 충족 여부: {normality}")

def my_resid_equal_var(x: DataFrame, y: Series, y_pred: Series) -> None:
    """잔차의 등분산성 가정을 확인한다.

    Args:
        x (DataFrame): 독립변수
        y (Series): 종속변수
        y_pred (Series): 예측값
    """
    # 독립변수 데이터 프레임 복사
    x_copy = x.copy()
    
    # 상수항 추가
    x_copy.insert(0, "const", 1)
    
    # 잔차 구하기
    resid = y - y_pred
    
    # 등분산성 검정
    bs_result = het_breuschpagan(resid, x_copy)
    bs_result_df = DataFrame(bs_result, columns=['values'], index=['statistic', 'p-value', 'f-value', 'f p-value'])

    print(f"잔차의 등분산성 가정 충족 여부: {bs_result[1] > 0.05}")
    my_pretty_table(bs_result_df)

def my_resid_independence(y: Series, y_pred: Series) -> None:
    """잔차의 독립성 가정을 확인한다.

    Args:
        y (Series): 종속변수
        y_pred (Series): 예측값
    """
    dw = durbin_watson(y - y_pred)
    print(f"Durbin-Watson: {dw}, 잔차의 독립성 가정 만족 여부: {1.5 < dw < 2.5}")
    
def my_resid_test(x: DataFrame, y: Series, y_pred: Series, figsize: tuple=(10, 4), dpi: int=150) -> None:
    """잔차의 가정을 확인한다.

    Args:
        x (Series): 독립변수
        y (Series): 종속변수
        y_pred (Series): 예측값
    """

    # 잔차 생성
    resid = y - y_pred
    
    print("[잔차의 선형성 가정]")
    my_residplot(y, y_pred, lowess=True, figsize=figsize, dpi=dpi)
    
    print("\n[잔차의 정규성 가정]")
    my_qqplot(y, figsize=figsize, dpi=dpi)
    my_residplot(y, y_pred, mse=True, figsize=figsize, dpi=dpi)
    my_resid_normality(y, y_pred)
    
    print("\n[잔차의 등분산성 가정]")
    my_resid_equal_var(x, y, y_pred)
    
    print("\n[잔차의 독립성 가정]")
    my_resid_independence(y, y_pred)