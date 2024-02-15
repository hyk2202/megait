from pandas import DataFrame
from scipy.stats import shapiro, normaltest, bartlett, levene, ttest_1samp, ttest_ind, ttest_rel, mannwhitneyu, pearsonr
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from helper.util import my_pretty_table

def my_normal_test(data: DataFrame, method: str = "n") -> None:
    """데이터프레임 내의 모든 컬럼에 대해 정규성 검정을 수행하고 결과를 출력한다.

    Args:
        data (DataFrame): 데이터프레임 객체
        method (str, optional): 정규성 검정 방법(n=normaltest, s=shapiro). Defaults to "n".
    """
    for c in data.columns:
        if method == "n":
            n = "normaltest"
            s, p = normaltest(data[c])
        else:
            n = "shapiro"
            s, p = shapiro(data[c])
            
        print(f"[{n}-{c}] statistic: {s:.3f}, p-value: {p:.3f}, 정규성 충족 여부: {p > 0.05}")

def my_equal_var_test(data: DataFrame, normal_dist: bool = True) -> None:
    """데이터프레임 내에 있는 모든 컬럼들에 대해 등분산성 검정을 수행하고 결과를 출력한다.

    Args:
        data (DataFrame): 데이터프레임 객체
        normal_dist (bool, optional): 정규성 검정 결과를 의미한다. True일 경우 정규분포를 따르는 데이터에 대한 등분산성 검정을 수행한다. Defaults to True.
    """
    fields: list = [data[x] for x in data.colunms]

    if normal_dist:
        n: str = "Bartlett"
        s, p = bartlett(*fields)
    else:
        n: str = "Levene"
        s, p = levene(*fields)
        
    print(f"{n} 검정: statistic: {s:.3f}, p-value: {p:.3f}, 등분산성 충족 여부: {p > 0.05}")

def my_normal_equal_var_1field(data: DataFrame, xname: str = 'x', hue: str = 'hue') -> None:
    """데이터프레임 내에 있는 한 종류의 명목형 변수에 따라 종속변수의 정규성과 등분산성을 검정하고 결과를 출력한다.

    Args:
        data (DataFrame): 데이터프레임 객체
        xname (str, optional): 종속변수의 컬럼명. Defaults to 'x'.
        hue (str, optional): 명목형 변수의 컬럼명. Defaults to 'hue'.
    """
    u1 = data[hue].unique()
    equal_var_fields: list = []
    normal_dist: bool = True
    report: list = []

    for i in u1:
        filtered_data = data[data[hue] == i][xname]
        equal_var_fields.append(filtered_data)
        s, p = normaltest(filtered_data)

        normalize = p > 0.05
        report.append({"field": i, "statistic": s, "p-value": p, "result": normalize})
        normal_dist = normal_dist and normalize

    if normal_dist:
        n = "Bartlett"
        s, p = bartlett(*equal_var_fields)
    else:
        n = "Levene"
        s, p = levene(*equal_var_fields)
        
    report.append({"field": n, "statistic": s, "p-value": p, "result": p > 0.05})
    report_df = DataFrame(report).set_index('field')
    my_pretty_table(report_df)
    
def my_normal_equal_var_2field(data: DataFrame, xname: str = 'x', hue: list = ['h1', 'h2']) -> None:
    """데이터프레임 내에 있는 두 종류의 명목형 변수에 따라 종속변수의 정규성과 등분산성을 검정하고 결과를 출력한다.

    Args:
        data (DataFrame): 데이터프레임 객체
        xname (str, optional): 종속변수의 컬럼명. Defaults to 'x'.
        hue (list, optional): 명목형 변수의 컬럼명을 저장하고 있는 리스트. Defaults to ['h1', 'h2'].
    """
    u1 = data[hue[0]].unique()
    u2 = data[hue[1]].unique()
    equal_var_fields: list = []
    normal_dist: bool = True
    report: list = []

    for i in u1:
        for j in u2:
            filtered_data = data[(data[hue[0]] == i) & (data[hue[1]] == j)][xname]
            equal_var_fields.append(filtered_data)
            s, p = normaltest(filtered_data)

            normalize: bool = p > 0.05
            report.append({"field": f"{i}, {j}", "statistic": s, "p-value": p, "result": normalize})
            normal_dist = normal_dist and normalize

    if normal_dist:
        n: str = "Bartlett"
        s, p = bartlett(*equal_var_fields)
    else:
        n: str = "Levene"
        s, p = levene(*equal_var_fields)
        
    report.append({"field": n, "statistic": s, "p-value": p, "result": p > 0.05})
    report_df = DataFrame(report).set_index('field')
    my_pretty_table(report_df)

def my_ttest_1samp(data: DataFrame, mean_value: int = 0) -> None:
    """데이터프레임 내에 있는 모든 컬럼에 대해 일표본 t-검정을 수행하고 결과를 출력한다.

    Args:
        data (DataFrame): 데이터프레임 객체
        mean_value (int, optional): 귀무가설의 기준값. Defaults to 0.
    """
    alternative: list = ["two-sided", "less", "greater"]
    result: list = []

    for c in data.columns:
        for a in alternative:
            s, p = ttest_1samp(data[c], mean_value, alternative=a)
            
            itp = None
            
            if a == "two-sided":
                itp = f"μ {'==' if p > 0.05 else '!='} {mean_value}"
            elif a == "less":
                itp = f"μ {'>=' if p > 0.05 else '<'} {mean_value}"
            else:
                itp = f"μ {'<=' if p > 0.05 else '>'} {mean_value}"
            
            result.append({
                "field": c,
                "alternative": a,
                "statistic": round(s, 3),
                "p-value": round(p, 3),
                "H0": p > 0.05,
                "H1": p <= 0.05,
                "interpretation": itp
            })
            
    rdf = DataFrame(result).set_index(["field", "alternative"])
    my_pretty_table(rdf)

def my_ttest_ind(data: DataFrame, xname: str, yname: str, equal_var: bool = True) -> None:
    """독립표본 t-검정을 수행하고 결과를 출력한다.

    Args:
        data (DataFrame): 데이터프레임 객체
        xname (str): 첫 번째 필드 이름
        yname (str): 두 번째 필드 이름
        equal_var (bool, optional): 등분산성 가정 여부. Defaults to True.
    """
    
    alternative: list = ["two-sided", "less", "greater"]
    result: list = []
    fmt: str = "μ({f0}) {0} μ({f1})"
    
    for a in alternative:
        s, p = ttest_ind(data[xname], data[yname], equal_var=equal_var, alternative=a)
        n = "t-test_ind" if equal_var else "Welch's t-test"
            
        # 검정 결과 해석
        itp = None
        
        if a == "two-sided":
            itp = fmt.format("==" if p > 0.05 else "!=", f0=xname, f1=yname)
        elif a == "less":
            itp = fmt.format(">=" if p > 0.05 else "<", f0=xname, f1=yname)
        else:
            itp = fmt.format("<=" if p > 0.05 else ">", f0=xname, f1=yname)
            
        result.append({
            "test": n,
            "alternative": a,
            "statistic": round(s, 3),
            "p-value": round(p, 3),
            "H0": p > 0.05,
            "H1": p <= 0.05,
            "interpretation": itp
        })
        
    rdf = DataFrame(result).set_index(["test", "alternative"])
    my_pretty_table(rdf)

def my_ttest_rel(data: DataFrame, xname: str, yname: str, equal_var: bool = True) -> None:
    """대응표본 t-검정 또는 Mann-Whitney U 검정을 수행하고 결과를 출력한다.

    Args:
        data (DataFrame): 데이터프레임 객체
        xname (str): 첫 번째 필드 이름
        yname (str): 두 번째 필드 이름
        equal_var (bool, optional): 등분산성 가정 여부. True인 경우 대응표본 T검정 수행 / False인 경우 Mann-Whitney 검정 수행. Defaults to True.
    """
    
    alternative: list = ["two-sided", "less", "greater"]
    result: list = []
    fmt: str = "μ({f0}) {0} μ({f1})"
    
    for a in alternative:
        if equal_var:
            s, p = ttest_rel(data[xname], data[yname], alternative=a)
            n = "t-test_paired"
        else:
            s, p = mannwhitneyu(data[xname], data[yname], alternative=a)
            n = "Mann-Whitney"
            
        # 검정 결과 해석
        itp = None
        
        if a == "two-sided":
            itp = fmt.format("==" if p > 0.05 else "!=", f0=xname, f1=yname)
        elif a == "less":
            itp = fmt.format(">=" if p > 0.05 else "<", f0=xname, f1=yname)
        else:
            itp = fmt.format("<=" if p > 0.05 else ">", f0=xname, f1=yname)
            
        result.append({
            "test": n,
            "alternative": a,
            "statistic": round(s, 3),
            "p-value": round(p, 3),
            "H0": p > 0.05,
            "H1": p <= 0.05,
            "interpretation": itp
        })
        
    rdf = DataFrame(result).set_index(["test", "alternative"])
    my_pretty_table(rdf)