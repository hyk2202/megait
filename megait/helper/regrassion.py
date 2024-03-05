import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

from tabulate import tabulate
from pandas import DataFrame, Series, concat

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.api import het_breuschpagan
from sklearn.model_selection import GridSearchCV

from scipy.stats import t, f
from helper.util import my_pretty_table, my_trend, my_train_test_split
from helper.plot import my_residplot, my_qqplot, my_learing_curve

def my_auto_linear_regrassion(df:DataFrame, yname:str, cv:int=5, learning_curve: bool = True, degree : int = 1, plot: bool = True, report=True, resid_test=False, figsize=(10, 4), dpi=150, sort: str = None,order: str = None,p_value_num:float=0.05) -> LinearRegression:
    """선형회귀분석을 수행하고 결과를 출력한다.

    Args:
        df (DataFrame) : 회귀분석을 수행할 데이터프레임.
        yname (str) : 종속변수
        cv (int, optional): 교차검증 횟수. Defaults to 0.
        degree (int, optional): 다항회귀분석의 차수. Defaults to 1.
        plot (bool, optional): 시각화 여부. Defaults to True.
        report (bool, optional): 회귀분석 결과를 보고서로 출력할지 여부. Defaults to True.
        resid_test (bool, optional): 잔차의 가정을 확인할지 여부. Defaults to False.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 4).
        dpi (int, optional): 그래프의 해상도. Defaults to 150.
        order (bool, optional): 독립변수 결과 보고 표의 정렬 기준 (v, p)
        p_value_num (float, optional) : 회귀모형의 유의확률. Drfaults to 0.05
    Returns:
        LinearRegression: 회귀분석 모델
    """

    x_train, x_test, y_train, y_test = my_train_test_split(df, yname, test_size=0.2)
    
    xnames = x_train.columns
    yname = y_train.name
    size = len(xnames)

    # 분석모델 생성
    model = LinearRegression(n_jobs=-1) # n_jobs : 사용하는 cpu 코어의 개수 // -1은 최대치

    # 교차검증 설정
    if cv > 0:
        params = {}
        grid = GridSearchCV(model, param_grid=params, cv=cv, n_jobs=-1)
        fit = grid.fit(x_train, y_train)
        model = fit.best_estimator_
        fit.best_params = fit.best_params_
        
        result_df = DataFrame(grid.cv_results_['params'])
        result_df['mean_test_score'] = grid.cv_results_['mean_test_score']
        
        # print("[교차검증]")
        # my_pretty_table(result_df.sort_values(by='mean_test_score', ascending=False))
        # print("")

    fit = model.fit(x_train, y_train)
    x = x_test
    y = y_test
    y_pred = fit.predict(x)

    resid = y - y_pred

    # 절편과 계수를 하나의 배열로 결합
    params = np.append(fit.intercept_, fit.coef_)

    # 검증용 독립변수에 상수항 추가
    design_x = x.copy()
    design_x.insert(0, '상수', 1)

    dot = np.dot(design_x.T,design_x)   # 행렬곱
    inv = np.linalg.inv(dot)            # 역행렬
    dia = inv.diagonal()                # 대각원소

    # 제곱오차
    MSE = (sum((y-y_pred)**2)) / (len(design_x)-len(design_x.iloc[0]))

    se_b = np.sqrt(MSE * dia)           # 표준오차
    ts_b = params / se_b                # t값

    # 각 독립수에 대한 pvalue
    p_values = [2*(1-t.cdf(np.abs(i),(len(design_x)-len(design_x.iloc[0])))) for i in ts_b]

    # VIF
    if len(x.columns) > 1:
        vif = [variance_inflation_factor(x, list(x.columns).index(v)) for i, v in enumerate(x.columns)]
    else:
        vif = 0

    # 표준화 계수
    train_df = x.copy()
    train_df[y.name] = y
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
        "종속변수": [yname] * len(xnames),
        "독립변수": xnames,
        "B(비표준화 계수)": np.round(params[1:], 4),
        "표준오차": np.round(se_b[1:], 3),
        "β(표준화 계수)": np.round(beta, 3),
        "t": np.round(ts_b[1:], 3),
        "유의확률": np.round(p_values[1:], 3),
        "VIF": vif,
    })
    if order:
        order = order.upper()
        if order == 'V':
            result_df.sort_values('VIF',inplace=True)
        elif  order == 'P':
            result_df.sort_values('유의확률',inplace=True)
        #result_df
    # my_pretty_table(result_df)
        
    resid = y - y_pred        # 잔차
    dw = durbin_watson(resid)               # 더빈 왓슨 통계량
    r2 = r2_score(y, y_pred)  # 결정계수(설명력)
    rowcount = len(x)                # 표본수
    featurecount = len(x.columns)    # 독립변수의 수

    # 보정된 결정계수
    adj_r2 = 1 - (1 - r2) * (rowcount-1) / (rowcount-featurecount-1)

    # f값
    f_statistic = (r2 / featurecount) / ((1 - r2) / (rowcount - featurecount - 1))

    # Prob (F-statistic)
    p = 1 - f.cdf(f_statistic, featurecount, rowcount - featurecount - 1)

    tpl = f"𝑅^2({r2:.3f}), Adj.𝑅^2({adj_r2:.3f}), F({f_statistic:.3f}), P-value({p:.4g}), Durbin-Watson({dw:.3f})"
    # print(tpl, end="\n\n")

    # 결과보고
    tpl = f"{yname}에 대하여 {','.join(xnames)}로 예측하는 회귀분석을 실시한 결과, 이 회귀모형은 통계적으로 유의{'하다' if p <= 0.05 else '하지 않다'}(F({len(x.columns)},{len(x.index)-len(x.columns)-1}) = {f_statistic:0.3f}, p {'<=' if p <= p_value_num else '>'} 0.05)."

    # # print(tpl, end = '\n\n')

    # 독립변수 보고
    for n in xnames:
        item = result_df[result_df['독립변수'] == n]
        coef = item['B(비표준화 계수)'].values[0]
        pvalue = item['유의확률'].values[0]

        s = f"{n}의 회귀계수는 {coef:0.3f}(p {'<=' if pvalue <= p_value_num else '>'} 0.05)로, {yname}에 대하여 {'유의미한' if pvalue <= p_value_num else '유의하지 않은'} 예측변인인 것으로 나타났다."

        # print(s)
        
    # print("")
    if result_df["VIF"].max() >= 10:
        # print('-'*50)
        # print('뺀 변수 :',result_df['독립변수'][result_df['VIF'].idxmax()])
        # print('-'*50)
        return my_auto_linear_regrassion(df.drop(result_df['독립변수'][result_df['VIF'].idxmax()],axis=1), yname, cv, degree,plot,report,resid_test, figsize, dpi, order,p_value_num )
    else:
        if result_df["유의확률"].max() >= p_value_num:
            # print('-'*50)
            # print('뺀 변수 :',result_df['독립변수'][result_df['유의확률'].idxmax()])
            # print('-'*50)
            return my_auto_linear_regrassion(df.drop(result_df['독립변수'][result_df['유의확률'].idxmax()],axis=1), yname,cv, degree,plot,report,resid_test, figsize, dpi, order,p_value_num )
    
    x_train, x_test, y_train, y_test = my_train_test_split(df, yname, test_size=0.2)
    
    xnames = x_train.columns
    yname = y_train.name
    size = len(xnames)

    # 분석모델 생성
    model = LinearRegression(n_jobs=-1) # n_jobs : 사용하는 cpu 코어의 개수 // -1은 최대치

    # 교차검증 설정
    if cv > 0:
        params = {}
        grid = GridSearchCV(model, param_grid=params, cv=cv, n_jobs=-1)
        fit = grid.fit(x_train, y_train)
        model = fit.best_estimator_
        fit.best_params = fit.best_params_
        
        result_df = DataFrame(grid.cv_results_['params'])
        result_df['mean_test_score'] = grid.cv_results_['mean_test_score']
        
        print("[교차검증]")
        my_pretty_table(result_df.sort_values(by='mean_test_score', ascending=False))
        print("")

    fit = model.fit(x_train, y_train)
    x = x_test
    y = y_test
    y_pred = fit.predict(x)
    expr = "{yname} = ".format(yname=yname)

    for i, v in enumerate(xnames):
        expr += "%0.3f * %s + " % (fit.coef_[i], v)

    expr += "%0.3f" % fit.intercept_
    print("[회귀식]")
    print(expr, end="\n\n")
    resid = y - y_pred

    # 절편과 계수를 하나의 배열로 결합
    params = np.append(fit.intercept_, fit.coef_)

    # 검증용 독립변수에 상수항 추가
    design_x = x.copy()
    design_x.insert(0, '상수', 1)

    dot = np.dot(design_x.T,design_x)   # 행렬곱
    inv = np.linalg.inv(dot)            # 역행렬
    dia = inv.diagonal()                # 대각원소

    # 제곱오차
    MSE = (sum((y-y_pred)**2)) / (len(design_x)-len(design_x.iloc[0]))

    se_b = np.sqrt(MSE * dia)           # 표준오차
    ts_b = params / se_b                # t값

    # 각 독립수에 대한 pvalue
    p_values = [2*(1-t.cdf(np.abs(i),(len(design_x)-len(design_x.iloc[0])))) for i in ts_b]

    # VIF
    if len(x.columns) > 1:
        vif = [variance_inflation_factor(x, list(x.columns).index(v)) for i, v in enumerate(x.columns)]
    else:
        vif = 0

    # 표준화 계수
    train_df = x.copy()
    train_df[y.name] = y
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
        "종속변수": [yname] * len(xnames),
        "독립변수": xnames,
        "B(비표준화 계수)": np.round(params[1:], 4),
        "표준오차": np.round(se_b[1:], 3),
        "β(표준화 계수)": np.round(beta, 3),
        "t": np.round(ts_b[1:], 3),
        "유의확률": np.round(p_values[1:], 3),
        "VIF": vif,
    })
    if order:
        order = order.upper()
        if order == 'V':
            result_df.sort_values('VIF',inplace=True)
        elif  order == 'P':
            result_df.sort_values('유의확률',inplace=True)
        # result_df
    my_pretty_table(result_df)
        
    resid = y - y_pred        # 잔차
    dw = durbin_watson(resid)               # 더빈 왓슨 통계량
    r2 = r2_score(y, y_pred)  # 결정계수(설명력)
    rowcount = len(x)                # 표본수
    featurecount = len(x.columns)    # 독립변수의 수

    # 보정된 결정계수
    adj_r2 = 1 - (1 - r2) * (rowcount-1) / (rowcount-featurecount-1)

    # f값
    f_statistic = (r2 / featurecount) / ((1 - r2) / (rowcount - featurecount - 1))

    # Prob (F-statistic)
    p = 1 - f.cdf(f_statistic, featurecount, rowcount - featurecount - 1)

    tpl = f"𝑅^2({r2:.3f}), Adj.𝑅^2({adj_r2:.3f}), F({f_statistic:.3f}), P-value({p:.4g}), Durbin-Watson({dw:.3f})"
    print(tpl, end="\n\n")

    # 결과보고
    tpl = f"{yname}에 대하여 {','.join(xnames)}로 예측하는 회귀분석을 실시한 결과, 이 회귀모형은 통계적으로 유의{'하다' if p <= 0.05 else '하지 않다'}(F({len(x.columns)},{len(x.index)-len(x.columns)-1}) = {f_statistic:0.3f}, p {'<=' if p <= p_value_num else '>'} 0.05)."

    print(tpl, end = '\n\n')

    # 독립변수 보고
    for n in xnames:
        item = result_df[result_df['독립변수'] == n]
        coef = item['B(비표준화 계수)'].values[0]
        pvalue = item['유의확률'].values[0]

        s = f"{n}의 회귀계수는 {coef:0.3f}(p {'<=' if pvalue <= p_value_num else '>'} 0.05)로, {yname}에 대하여 {'유의미한' if pvalue <= p_value_num else '유의하지 않은'} 예측변인인 것으로 나타났다."

        print(s)
        
    print("")
    return fit
    
def my_linear_regrassion(x_train: DataFrame, y_train: Series, x_test: DataFrame = None, y_test: Series = None, cv: int = 5,  learning_curve: bool = True, degree : int = 1, plot: bool = True, report=True, resid_test=False, figsize=(10, 4), dpi=150, sort: str = None,order: str = None,p_value_num:float=0.05 ) -> LinearRegression:
    """선형회귀분석을 수행하고 결과를 출력한다.

    Args:
        x_train (DataFrame): 독립변수에 대한 훈련 데이터
        y_train (Series): 종속변수에 대한 훈련 데이터
        x_test (DataFrame): 독립변수에 대한 검증 데이터. Defaults to None.
        y_test (Series): 종속변수에 대한 검증 데이터. Defaults to None.
        cv (int, optional): 교차검증 횟수. Defaults to 5.
        learning_curve (bool, optional): 학습곡선을 출력할지 여부. Defaults to True.
        degree (int, optional): 다항회귀분석의 차수. Defaults to 1.
        plot (bool, optional): 시각화 여부. Defaults to True.
        report (bool, optional): 회귀분석 결과를 보고서로 출력할지 여부. Defaults to True.
        resid_test (bool, optional): 잔차의 가정을 확인할지 여부. Defaults to False.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 4).
        dpi (int, optional): 그래프의 해상도. Defaults to 150.
        order (bool, optional): 독립변수 결과 보고 표의 정렬 기준 (v, p)
        p_value_num (float, optional) : 회귀모형의 유의확률. Drfaults to 0.05
    Returns:
        LinearRegression: 회귀분석 모델
    """
    xnames = x_train.columns
    yname = y_train.name
    size = len(xnames)

    # 분석모델 생성

    # 교차검증 설정
    if cv > 0:
        params = {}
        prototype_estimator = LinearRegression(n_jobs=-1)
        grid = GridSearchCV(prototype_estimator, param_grid=params, cv=cv, n_jobs=-1)
        grid.fit(x_train, y_train)
        result_df = DataFrame(grid.cv_results_['params'])
        result_df['mean_test_score'] = grid.cv_results_['mean_test_score']
        print("[교차검증]")
        my_pretty_table(result_df.sort_values(by='mean_test_score', ascending=False))
        print("")

        estimator = grid.best_estimator_
        estimator.best_params = grid.best_params_
    else:
        estimator = LinearRegression(n_jobs=-1)
        estimator.fit(x_train, y_train)        
    y_pred = estimator.predict(x_test) if x_test is not None else estimator.predict(x_train)
    
    # 도출된 결과를 회귀모델 객체에 포함시킴
    estimator.x = x_test if x_test is not None else x_train
    estimator.y = y_test if y_test is not None else y_train
    estimator.y_pred = y_pred if y_test is not None else estimator.predict(x_train)
    estimator.resid = y_test - y_pred if y_test is not None else y_train - estimator.predict(x_train)

    #------------------------------------------------------
    # 성능평가
    if x_test is not None and y_test is not None:
        my_regrassion_result(estimator, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, learning_curve=learning_curve, cv=cv, figsize=figsize, dpi=dpi)
    else:
        my_regrassion_result(estimator, x_train=x_train, y_train=y_train, learning_curve=learning_curve, cv=cv, figsize=figsize, dpi=dpi)

    #------------------------------------------------------
    # 보고서 출력
    if report:
        print("")
        my_regrassion_report(estimator, estimator.x, estimator.y, sort=sort, plot=plot, degree=degree, figsize=figsize, dpi=dpi)
    
    #------------------------------------------------------
    # 잔차 가정 확인  
    if resid_test:
        print("\n\n[잔차의 가정 확인] ==============================")
        my_resid_test(estimator.x, estimator.y, estimator.y_pred, figsize=figsize, dpi=dpi)

    return estimator

def my_ridge_regrassion(x_train: DataFrame, y_train: Series, x_test: DataFrame = None, y_test: Series = None, cv: int = 5, learning_curve: bool = True, report=False, plot: bool = False, degree: int = 1, resid_test=False, figsize=(10, 5), dpi: int = 100, sort: str = None, params: dict = {'alpha': [0.01, 0.1, 1, 10, 100]}) -> LinearRegression:
    """릿지회귀분석을 수행하고 결과를 출력한다.

    Args:
        x_train (DataFrame): 독립변수에 대한 훈련 데이터
        y_train (Series): 종속변수에 대한 훈련 데이터
        x_test (DataFrame): 독립변수에 대한 검증 데이터. Defaults to None.
        y_test (Series): 종속변수에 대한 검증 데이터. Defaults to None.
        cv (int, optional): 교차검증 횟수. Defaults to 5.
        learning_curve (bool, optional): 학습곡선을 출력할지 여부. Defaults to True.
        report (bool, optional): 회귀분석 결과를 보고서로 출력할지 여부. Defaults to True.
        plot (bool, optional): 시각화 여부. Defaults to True.
        degree (int, optional): 다항회귀분석의 차수. Defaults to 1.
        resid_test (bool, optional): 잔차의 가정을 확인할지 여부. Defaults to False.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 5).
        dpi (int, optional): 그래프의 해상도. Defaults to 100.
        sort (bool, optional): 독립변수 결과 보고 표의 정렬 기준 (v, p)
        params (dict, optional): 하이퍼파라미터. Defaults to {'alpha': [0.01, 0.1, 1, 10, 100]}.
        
    Returns:
        Ridge: Ridge 모델
    """
    
    #------------------------------------------------------
    # 교차검증 설정
    if cv > 0:   
        # 분석모델 생성
        prototype_estimator = Ridge()     
        
        print("[%s 하이퍼파라미터]" % prototype_estimator.__class__.__name__)
        my_pretty_table(DataFrame(params))
        print("")
        
        grid = GridSearchCV(prototype_estimator, param_grid=params, cv=cv, n_jobs=-1)
        grid.fit(x_train, y_train)
        
        result_df = DataFrame(grid.cv_results_['params'])
        result_df['mean_test_score'] = grid.cv_results_['mean_test_score']
        
        print("[교차검증]")
        my_pretty_table(result_df.sort_values(by='mean_test_score', ascending=False))
        print("")
        
        estimator = grid.best_estimator_
        estimator.best_params = grid.best_params_
    else:
        # 분석모델 생성
        estimator = Ridge(**params) 
        estimator.fit(x_train, y_train)
    
    #------------------------------------------------------
    xnames = x_train.columns
    yname = y_train.name
    
    # 훈련 데이터에 대한 추정치 생성
    y_pred = estimator.predict(x_test) if x_test is not None else estimator.predict(x_train)
    
    # 도출된 결과를 회귀모델 객체에 포함시킴
    estimator.x = x_test if x_test is not None else x_train
    estimator.y = y_test if y_test is not None else y_train
    estimator.y_pred = y_pred if y_test is not None else estimator.predict(x_train)
    estimator.resid = y_test - y_pred if y_test is not None else y_train - estimator.predict(x_train)

    #------------------------------------------------------
    # 성능평가
    if x_test is not None and y_test is not None:
        my_regrassion_result(estimator, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, learning_curve=learning_curve, cv=cv, figsize=figsize, dpi=dpi)
    else:
        my_regrassion_result(estimator, x_train=x_train, y_train=y_train, learning_curve=learning_curve, cv=cv, figsize=figsize, dpi=dpi)

    #------------------------------------------------------
    # 보고서 출력
    if report:
        print("")
        my_regrassion_report(estimator, estimator.x, estimator.y, sort, plot=plot, degree=degree, figsize=figsize, dpi=dpi)
    
    #------------------------------------------------------
    # 잔차 가정 확인  
    if resid_test:
        print("\n\n[잔차의 가정 확인] ==============================")
        my_resid_test(estimator.x, estimator.y, estimator.y_pred, figsize=figsize, dpi=dpi)

    return estimator

def my_lasso_regrassion(x_train: DataFrame, y_train: Series, x_test: DataFrame = None, y_test: Series = None, cv: int = 5, learning_curve: bool = True, report=False, plot: bool = False, degree: int = 1, resid_test=False, figsize=(10, 5), dpi: int = 100, sort: str = None, params: dict = {'alpha': [0.01, 0.1, 1, 10, 100]}) -> LinearRegression:
    """라쏘회귀분석을 수행하고 결과를 출력한다.

    Args:
        x_train (DataFrame): 독립변수에 대한 훈련 데이터
        y_train (Series): 종속변수에 대한 훈련 데이터
        x_test (DataFrame): 독립변수에 대한 검증 데이터. Defaults to None.
        y_test (Series): 종속변수에 대한 검증 데이터. Defaults to None.
        cv (int, optional): 교차검증 횟수. Defaults to 5.
        learning_curve (bool, optional): 학습곡선을 출력할지 여부. Defaults to True.
        report (bool, optional): 회귀분석 결과를 보고서로 출력할지 여부. Defaults to True.
        plot (bool, optional): 시각화 여부. Defaults to True.
        degree (int, optional): 다항회귀분석의 차수. Defaults to 1.
        resid_test (bool, optional): 잔차의 가정을 확인할지 여부. Defaults to False.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 5).
        dpi (int, optional): 그래프의 해상도. Defaults to 100.
        sort (bool, optional): 독립변수 결과 보고 표의 정렬 기준 (v, p)
        params (dict, optional): 하이퍼파라미터. Defaults to {'alpha': [0.01, 0.1, 1, 10, 100]}.
        
    Returns:
        Lasso: Lasso 모델
    """
    
    #------------------------------------------------------
    # 교차검증 설정
    if cv > 0:   
        # 분석모델 생성
        prototype_estimator = Lasso()     
        
        print("[%s 하이퍼파라미터]" % prototype_estimator.__class__.__name__)
        my_pretty_table(DataFrame(params))
        print("")
        
        grid = GridSearchCV(prototype_estimator, param_grid=params, cv=cv, n_jobs=-1)
        grid.fit(x_train, y_train)
        
        result_df = DataFrame(grid.cv_results_['params'])
        result_df['mean_test_score'] = grid.cv_results_['mean_test_score']
        
        print("[교차검증]")
        my_pretty_table(result_df.sort_values(by='mean_test_score', ascending=False))
        print("")
        
        estimator = grid.best_estimator_
        estimator.best_params = grid.best_params_
    else:
        # 분석모델 생성
        estimator = Lasso(**params) 
        estimator.fit(x_train, y_train)
    
    #------------------------------------------------------
    xnames = x_train.columns
    yname = y_train.name
    
    # 훈련 데이터에 대한 추정치 생성
    y_pred = estimator.predict(x_test) if x_test is not None else estimator.predict(x_train)
    
    # 도출된 결과를 회귀모델 객체에 포함시킴
    estimator.x = x_test if x_test is not None else x_train
    estimator.y = y_test if y_test is not None else y_train
    estimator.y_pred = y_pred if y_test is not None else estimator.predict(x_train)
    estimator.resid = y_test - y_pred if y_test is not None else y_train - estimator.predict(x_train)

    #------------------------------------------------------
    # 성능평가
    if x_test is not None and y_test is not None:
        my_regrassion_result(estimator, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, learning_curve=learning_curve, cv=cv, figsize=figsize, dpi=dpi)
    else:
        my_regrassion_result(estimator, x_train=x_train, y_train=y_train, learning_curve=learning_curve, cv=cv, figsize=figsize, dpi=dpi)

    #------------------------------------------------------
    # 보고서 출력
    if report:
        print("")
        my_regrassion_report(estimator, estimator.x, estimator.y, sort, plot=plot, degree=degree, figsize=figsize, dpi=dpi)
    
    #------------------------------------------------------
    # 잔차 가정 확인  
    if resid_test:
        print("\n\n[잔차의 가정 확인] ==============================")
        my_resid_test(estimator.x, estimator.y, estimator.y_pred, figsize=figsize, dpi=dpi)

    return estimator

def my_regrassion_result(estimator: any, x_train: DataFrame = None, y_train: Series = None, x_test: DataFrame = None, y_test: Series = None, learning_curve: bool = True, cv: int = 10, figsize: tuple = (10, 5), dpi: int = 100) -> None:
    """회귀분석 결과를 출력한다.

    Args:
        estimator (any): 회귀분석 모델
        x_train (DataFrame): 독립변수에 대한 훈련 데이터
        y_train (Series): 종속변수에 대한 훈련 데이터
        x_test (DataFrame): 독립변수에 대한 검증 데이터. Defaults to None.
        y_test (Series): 종속변수에 대한 검증 데이터. Defaults to None.
        learning_curve (bool, optional): 학습곡선을 출력할지 여부. Defaults to False.
        cv (int, optional): 교차검증 횟수. Defaults to 10.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 5).
        dpi (int, optional): 그래프의 해상도. Defaults to 100.
    """
    
    scores = []
    score_names = []
    
    if x_train is not None and y_train is not None:
        y_train_pred = estimator.predict(x_train)

        # 성능평가
        result = {
            "결정계수(R2)": r2_score(y_train, y_train_pred),
            "평균절대오차(MAE)": mean_absolute_error(y_train, y_train_pred),
            "평균제곱오차(MSE)": mean_squared_error(y_train, y_train_pred),
            "평균오차(RMSE)": np.sqrt(mean_squared_error(y_train, y_train_pred)),
            "평균 절대 백분오차 비율(MAPE)": np.mean(np.abs((y_train - y_train_pred) / y_train) * 100),
            "평균 비율 오차(MPE)": np.mean((y_train - y_train_pred) / y_train * 100)
        }
        
        scores.append(result)
        score_names.append("훈련데이터")
        
    if x_test is not None and y_test is not None:
        y_test_pred = estimator.predict(x_test)

        # 성능평가
        result = {
            "결정계수(R2)": r2_score(y_test, y_test_pred),
            "평균절대오차(MAE)": mean_absolute_error(y_test, y_test_pred),
            "평균제곱오차(MSE)": mean_squared_error(y_test, y_test_pred),
            "평균오차(RMSE)": np.sqrt(mean_squared_error(y_test, y_test_pred)),
            "평균 절대 백분오차 비율(MAPE)": np.mean(np.abs((y_test - y_test_pred) / y_test) * 100),
            "평균 비율 오차(MPE)": np.mean((y_test - y_test_pred) / y_test * 100)
        }
        
        scores.append(result)
        score_names.append("검증데이터")
        

    print("[회귀분석 성능평가]")
    result_df = DataFrame(scores, index=score_names)
    my_pretty_table(result_df.T)
    
    # 학습곡선
    if learning_curve:
        print("\n[학습곡선]")
        yname = y_train.name
        
        if x_test is not None and y_test is not None:
            y_df = concat([y_train, y_test])
            x_df = concat([x_train, x_test])
        else:
            y_df = y_train.copy()
            x_df = x_train.copy()
            
        x_df[yname] = y_df 
        x_df.sort_index(inplace=True)
        
        if cv > 0:
            my_learing_curve(estimator, data=x_df, yname=yname, cv=cv, scoring='RMSE', figsize=figsize, dpi=dpi)
        else:
            my_learing_curve(estimator, data=x_df, yname=yname, scoring='RMSE', figsize=figsize, dpi=dpi)

def my_regrassion_report(estimator: any, x: DataFrame = None, y: Series = None, sort: str = None, plot: bool = False, degree: int = 1, figsize: tuple = (10, 5), dpi: int = 100, order : str = None, p_value_num:float=0.05 ) -> None:
    """선형회귀분석 결과를 보고한다.

    Args:
        fit (LinearRegression): 선형회귀 객체
        x (DataFrame): 독립변수에 대한 훈련 데이터
        y (Series): 종속변수에 대한 훈련 데이터
        sort (str, optional): 정렬 기준 (v, p). Defaults to None.
        plot (bool, optional): 시각화 여부. Defaults to False.
        degree (int, optional): 다항회귀분석의 차수. Defaults to 1.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 5).
        dpi (int, optional): 그래프의 해상도. Defaults to 100.
        order (bool, optional): 독립변수 결과 보고 표의 정렬 기준 (v, p)
        p_value_num (float, optional) : 회귀모형의 유의확률. Drfaults to 0.05
    """
    
    # 회귀식
    xnames = x.columns
    yname = y.name
    
    expr = "{yname} = ".format(yname=yname)

    for i, v in enumerate(xnames):
        expr += "%0.3f * %s + " % (estimator.coef_[i], v)

    expr += "%0.3f" % estimator.intercept_
    print("[회귀식]")
    print(expr, end="\n\n")
    
    
    print("[독립변수보고]")
    if x is None and y is None:
        x = estimator.x
        y = estimator.y
    
    y_pred = estimator.predict(x)
    xnames = x.columns
    yname = y.name

    # 잔차
    resid = y - y_pred

    # 절편과 계수를 하나의 배열로 결합
    params = np.append(estimator.intercept_, estimator.coef_)

    # 검증용 독립변수에 상수항 추가
    design_x = x.copy()
    design_x.insert(0, '상수', 1)

    dot = np.dot(design_x.T,design_x)   # 행렬곱
    inv = np.linalg.inv(dot)            # 역행렬
    dia = inv.diagonal()                # 대각원소

    # 제곱오차
    MSE = (sum((y-y_pred)**2)) / (len(design_x)-len(design_x.iloc[0]))

    se_b = np.sqrt(MSE * dia)           # 표준오차
    ts_b = params / se_b                # t값

    # 각 독립수에 대한 pvalue
    p_values = [2*(1-t.cdf(np.abs(i),(len(design_x)-len(design_x.iloc[0])))) for i in ts_b]

    # VIF
    if len(x.columns) > 1:
        vif = [variance_inflation_factor(x, list(x.columns).index(v)) for i, v in enumerate(x.columns)]
    else:
        vif = 0

    # 표준화 계수
    train_df = x.copy()
    train_df[y.name] = y
    scaler = StandardScaler()
    std = scaler.fit_transform(train_df)
    std_df = DataFrame(std, columns=train_df.columns)
    std_x = std_df[xnames]
    std_y = std_df[yname]
    std_estimator = LinearRegression(n_jobs=-1)
    std_estimator.fit(std_x, std_y)
    beta = std_estimator.coef_

    # 결과표 구성하기
    result_df = DataFrame({
        "종속변수": [yname] * len(xnames),
        "독립변수": xnames,
        "B(비표준화 계수)": np.round(params[1:], 4),
        "표준오차": np.round(se_b[1:], 3),
        "β(표준화 계수)": np.round(beta, 3),
        "t": np.round(ts_b[1:], 3),
        "유의확률": np.round(p_values[1:], 3),
        "VIF": vif,
    })
    
    if sort:
        if sort.upper() == 'V':
            result_df.sort_values('VIF', inplace=True)
        elif sort.upper() == 'P':
            result_df.sort_values('유의확률', inplace=True)
    

    #result_df
    my_pretty_table(result_df)
    print("")

    resid = y - y_pred        # 잔차
    dw = durbin_watson(resid)               # 더빈 왓슨 통계량
    r2 = r2_score(y, y_pred)  # 결정계수(설명력)
    rowcount = len(x)                # 표본수
    featurecount = len(x.columns)    # 독립변수의 수

    # 보정된 결정계수
    adj_r2 = 1 - (1 - r2) * (rowcount-1) / (rowcount-featurecount-1)

    # f값
    f_statistic = (r2 / featurecount) / ((1 - r2) / (rowcount - featurecount - 1))

    # Prob (F-statistic)
    p = 1 - f.cdf(f_statistic, featurecount, rowcount - featurecount - 1)

    tpl = "𝑅^2(%.3f), Adj.𝑅^2(%.3f), F(%.3f), P-value(%.4g), Durbin-Watson(%.3f)"
    print(tpl % (r2, adj_r2, f_statistic, p, dw), end="\n\n")

    # 결과보고
    tpl = "%s에 대하여 %s로 예측하는 회귀분석을 실시한 결과,\n이 회귀모형은 통계적으로 %s(F(%s,%s) = %0.3f, p %s %s)."

    result_str = tpl % (
        yname,
        ",".join(xnames),
        "유의하다" if p <= p_value_num else "유의하지 않다",
        len(x.columns),
        len(x.index)-len(x.columns)-1,
        f_statistic,
        "<=" if p <= p_value_num else ">",
        p_value_num)
        
    print(result_str, end="\n\n")

    # 독립변수 보고
    for n in xnames:
        item = result_df[result_df['독립변수'] == n]
        coef = item['B(비표준화 계수)'].values[0]
        pvalue = item['유의확률'].values[0]

        s = "%s의 회귀계수는 %0.3f(p %s %s)로, %s에 대하여 %s."
        k = s % (n,
                coef,
                "<=" if pvalue <= p_value_num else '>',
                yname,
                '유의미한 예측변인인 것으로 나타났다' if pvalue <= p_value_num else '유의하지 않은 예측변인인 것으로 나타났다',
                p_value_num
        )

        print(k)
        
    # 도출된 결과를 회귀모델 객체에 포함시킴 --> 객체 타입의 파라미터는 참조변수로 전달되므로 fit 객체에 포함된 결과값들은 이 함수 외부에서도 사용 가능하다.
    estimator.r2 = r2
    estimator.adj_r2 = adj_r2
    estimator.f_statistic = f_statistic
    estimator.p = p
    estimator.dw = dw
        
    # 시각화
    if plot:
        for i, v in enumerate(xnames):
            plt.figure(figsize=figsize, dpi=dpi)
            
            if degree == 1:
                sb.regplot(x=x[v], y=y, ci=95, label='관측치')
                sb.regplot(x=x[v], y=y_pred, ci=0, label='추정치')
            else:
                sb.scatterplot(x=x[v], y=y, label='관측치')
                sb.scatterplot(x=x[v], y=y_pred, label='추정치')
                
                t1 = my_trend(x[v], y, degree=degree)
                sb.lineplot(x=t1[0], y=t1[1], color='blue', linestyle='--', label='관측치 추세선')
                
                t2 = my_trend(x[v], y_pred, degree=degree)
                sb.lineplot(x=t2[0], y=t2[1], color='red', linestyle='--', label='추정치 추세선')
            
            plt.title(f"{yname} vs {v}")
            plt.legend()
            plt.grid()

            plt.show()
            plt.close()
        
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

def my_resid_equal_var(x: DataFrame, y: Series, y_pred: Series, p_value_num:float =0.05) -> None:
    """잔차의 등분산성 가정을 확인한다.

    Args:
        x (DataFrame): 독립변수
        y (Series): 종속변수
        y_pred (Series): 예측값
        p_value_num(float) : 유의확률
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

    print(f"잔차의 등분산성 가정 충족 여부: {bs_result[1] > p_value_num}")
    my_pretty_table(bs_result_df)

def my_resid_independence(y: Series, y_pred: Series) -> None:
    """잔차의 독립성 가정을 확인한다.

    Args:
        y (Series): 종속변수
        y_pred (Series): 예측값
    """
    dw = durbin_watson(y - y_pred)
    print(f"Durbin-Watson: {dw}, 잔차의 독립성 가정 만족 여부: {1.5 < dw < 2.5}")
    
def my_resid_test(x: DataFrame, y: Series, y_pred: Series, figsize: tuple=(10, 4), dpi: int=150, p_value_num:float = 0.05) -> None:
    """잔차의 가정을 확인한다.

    Args:
        x (Series): 독립변수
        y (Series): 종속변수
        y_pred (Series): 예측값
        p_value_num(float) : 유의확률
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
    my_resid_equal_var(x, y, y_pred, p_value_num)
    
    print("\n[잔차의 독립성 가정]")
    my_resid_independence(y, y_pred)