import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt

from tabulate import tabulate        
from pandas import DataFrame, Series

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

from scipy.stats import t, f 
from helper.util import my_pretty_table

def my_linear_regrassion(x_train : DataFrame, y_train : Series, x_test : DataFrame, y_test : Series, use_plot : bool = True, report=True) -> LinearRegression :
    """선형회귀분석을 수행하고 결과를 출력한다.

    Args:
        x_train (DataFrame): 독립변수에 대한 훈련 데이터
        y_train (Series): 종속변수에 대한 훈련 데이터
        x_test (DataFrame): 독립변수에 대한 검증 데이터
        y_test (Series): 종속변수에 대한 검증 데이터
        use_plot (bool, optional): 시각화 여부. Defaults to True.

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
            fig, ax = plt.subplots(1, 2, figsize=(15, 4), dpi=150)
            fig.subplots_adjust(hspace=0.3)
            for j,w in enumerate(target):
                sb.regplot(x=w[0][v], y=w[1], ci=95, ax=ax[j], label='관측치')
                sb.regplot(x=w[0][v], y=w[1], ci=0, ax=ax[j], label='추정치')
                ax[j].set_title(f"{'훈련데이터' if j == 0 else '검증데이터'}: {yname} vs {v}")
                ax[j].legend()
                ax[j].grid()

            plt.show()
            plt.close()

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
        