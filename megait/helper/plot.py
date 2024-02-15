'''
def example(a, b=None, c="w" , d=[], *e, **f):
    print(a,b,c,d,e,f)
함수 정의시  
positional parameter(기본값 미지정) -> optional parameter(기본값 지정) -> keyword-only(*) / var-keyword parameter(**)
또는
keyword-only(*) / var-keyword parameter(**) -> positional parameter(기본값 미지정) -> optional parameter(기본값 지정) 
로 파라미터를 셋팅해야한다.
위 순서를 지키지 않고 만들고 싶으면 바뀌어지는 사이에 '*' 을 추가하여 해결한다.
'*' 이후에 오는 파라미터는 반드시 함수 호출시 명시해야한다.
'''

import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from math import sqrt
from scipy.stats import t
from pandas import DataFrame
from scipy.spatial import ConvexHull
from statannotations.Annotator import Annotator

def my_boxplot(df: DataFrame, orient : str = 'v', hue=None, figsize: tuple = (10, 4), dpi: int = 150, plt_title : str = None, plt_grid : bool = True, plt_xlabel : str = None, plt_ylabel : str = None) -> None:
    """데이터프레임 내의 모든 컬럼에 대해 상자그림을 그려서 분포를 확인한다.

    Args:
        df (DataFrame): 데이터프레임 객체
        orient('v','x' or 'h','y', optional): 박스플롯의 축을 결정. Defaults to 'v'
        hue (str, optional): 색상을 구분할 기준이 되는 컬럼명. Defaults to None.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 4).
        dpi (int, optional): 그래프의 해상도. Defaults to 150.
    """
    plt.figure(figsize=figsize, dpi=dpi)
    sb.boxplot(data=df, orient=orient, hue=hue)
    plt.grid(plt_grid)
    plt.title(plt_title)
    if plt_xlabel:plt.xlabel(plt_xlabel)
    plt.ylabel(plt_ylabel)
    plt.show()
    plt.close()

def my_histplot(df: DataFrame, xname: str = None, yname : str = None, hue: str = None, bins = 'auto', kde: bool = True, figsize: tuple=(10, 4), plt_title : str = None, plt_xlabel : str = None, plt_ylabel : str = None, plt_grid : bool = True, dpi: int = 150) -> None:
    """데이터프레임 내의 컬럼에 대해 히스토그램을 그려서 분포를 확인한다.

    Args:
        df (DataFrame): 데이터프레임 객체
        xname (str): x축에 사용할 컬럼명
        yname (str): y축에 사용할 컬럼명 x,y 두 축중 하나만 사용
        hue (str, optional): 색상을 구분할 기준이 되는 컬럼명. Defaults to None.
        bins (int or list ,optional): 히스토그램의 구간 수 혹은 리스트. Defaults to auto.
        kde (bool, optional): 커널밀도추정을 함께 출력할지 여부. Defaults to True.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 4).
        dpi (int, optional): 그래프의 해상도. Defaults to 150.
    """
    plt.figure(figsize=figsize, dpi=dpi)
    
    sb.histplot(data=df, x=xname, y=yname, hue=hue, kde=True, bins=bins)

        
    plt.grid(plt_grid)
    plt.title(plt_title)
    if plt_xlabel:plt.xlabel(plt_xlabel)
    plt.ylabel(plt_ylabel)
    plt.show()
    plt.close()

def my_scatterplot(df: DataFrame, xname: str = None, yname: str = None, hue=None, figsize: tuple=(10, 4), plt_title : str = None, plt_grid : bool = True, dpi: int = 150) -> None:
    """데이터프레임 내의 두 컬럼에 대해 산점도를 그려서 관계를 확인한다.

    Args:
        df (DataFrame): 데이터프레임 객체
        xname (str): x축에 사용할 컬럼명
        yname (str): y축에 사용할 컬럼명
        hue (str, optional): 색상을 구분할 기준이 되는 컬럼명. Defaults to None.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 4).
        dpi (int, optional): 그래프의 해상도. Defaults to 150.
    """
    plt.figure(figsize=figsize, dpi=dpi)
    sb.scatterplot(data=df, x=xname, y=yname, hue=hue)
    plt.grid(plt_grid)
    plt.title(plt_title)
    plt.show()
    plt.close()

def my_regplot(df: DataFrame, xname: str = None, yname: str = None, figsize: tuple=(10, 4),ci :int = 95, plt_title : str = None, plt_grid : bool = True, dpi: int = 150) -> None:
    """데이터프레임 내의 컬럼에 대해 회귀선을 포함한 산점도를 그려서 관계를 확인한다.

    Args:
        df (DataFrame): 데이터프레임 객체
        xname (str): x축에 사용할 컬럼명
        yname (str): y축에 사용할 컬럼명
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 4).
        dpi (int, optional): 그래프의 해상도. Defaults to 150.
        ci (int in [0,100] or None, optional) : 신뢰구간설정
    """
    plt.figure(figsize=figsize, dpi=dpi)
    sb.regplot(data=df, x=xname, y=yname, ci = ci)
    plt.grid(plt_grid)
    plt.title(plt_title)

    plt.show()
    plt.close()

def my_lmplot(df: DataFrame, xname: str = None, yname: str = None, hue : str = None, figsize: tuple = (10, 4), plt_title : str = None, plt_grid : bool = True, dpi: int = 150) -> None:
    """데이터프레임 내의 컬럼에 대해 회귀선을 포함한 산점도를 그려서 관계를 확인한다.

    Args:
        df (DataFrame): 데이터프레임 객체
        xname (str): x축에 사용할 컬럼명
        yname (str): y축에 사용할 컬럼명
        hue (str, optional): 색상을 구분할 기준이 되는 컬럼명. Defaults to None.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 4).
        dpi (int, optional): 그래프의 해상도. Defaults to 150.
    """
    g = sb.lmplot(data=df, x=xname, y=yname, hue=hue)
    g.fig.set_figwidth(figsize[0])
    g.fig.set_figheight(figsize[1])
    g.fig.set_dpi(dpi)
    plt.grid(plt_grid)
    plt.title(plt_title)
    plt.show()
    plt.close()

def my_pairplot(df: DataFrame, diag_kind: str = "auto", hue = None, figsize: tuple = (10, 4), kind :str ='scatter', plt_title : str = None, dpi: int = 150) -> None:
    """데이터프레임 내의 모든 컬럼에 대해 쌍별 관계를 시각화한다.

    Args:
        df (DataFrame): 데이터프레임 객체
        hue (str, optional): 색상을 구분할 기준이 되는 컬럼명. Defaults to None.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 4).
        diag_kind ( ['auto', 'hist', 'kde', None], optional) : 대각그래프에 들어갈 그래프 설정
        kind (['scatter', 'kde', 'hist', 'reg'], optional ): 그 외 그래프 설정
        dpi (int, optional): 그래프의 해상도. Defaults to 150.
    """
    sb.pairplot(df, hue=hue, diag_kind=diag_kind, kind = kind)
    plt.title(plt_title)
    plt.show()
    plt.close()

def my_countplot(df: DataFrame, xname: str = None, yname: str = None, hue=None, figsize: tuple = (10, 4), plt_title : str = None, plt_xlabel : str = None, plt_grid : bool = True, plt_ylabel : str = None, dpi: int = 150) -> None:
    """데이터프레임 내의 컬럼에 대해 카운트플롯을 그려서 분포를 확인한다.

    Args:
        df (DataFrame): 데이터프레임 객체
        xname (str): x축에 사용할 컬럼명
        hue (str, optional): 색상을 구분할 기준이 되는 컬럼명. Defaults to None.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 4).
        dpi (int, optional): 그래프의 해상도. Defaults to 150.
    """
    plt.figure(figsize=figsize, dpi=dpi)
    sb.countplot(data=df, x=xname, y=yname, hue=hue)
    plt.grid(plt_grid)
    plt.title(plt_title)
    if plt_xlabel : plt.xlabel(plt_xlabel)
    if plt.ylabel : plt.ylabel(plt_ylabel)
    plt.show()
    plt.close()

def my_barplot(df: DataFrame, xname: str = None, yname: str = None, hue = None, figsize: tuple = (10, 4), plt_title : str = None, plt_grid : bool = True, plt_xlabel : str = None, plt_ylabel : str = None, dpi: int = 150) -> None:
    """데이터프레임 내의 컬럼에 대해 바플롯을 그려서 분포를 확인한다.

    Args:
        df (DataFrame): 데이터프레임 객체
        xname (str): x축에 사용할 컬럼명
        yname (str): y축에 사용할 컬럼명
        hue (str, optional): 색상을 구분할 기준이 되는 컬럼명. Defaults to None.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 4).
        dpi (int, optional): 그래프의 해상도. Defaults to 150.
    """
    plt.figure(figsize=figsize, dpi=dpi)
    sb.barplot(data=df, x=xname, y=yname, hue=hue)
    plt.grid(plt_grid)
    plt.title(plt_title)
    if plt_xlabel : plt.xlabel(plt_xlabel)
    if plt.ylabel : plt.ylabel(plt_ylabel)
    plt.show()
    plt.close()

def my_boxenplot(df: DataFrame, xname: str = None, yname: str = None, hue = None, figsize: tuple = (10, 4), plt_title : str = None, plt_grid : bool = True, plt_xlabel : str = None, plt_ylabel : str = None, dpi: int = 150) -> None:
    """데이터프레임 내의 컬럼에 대해 박슨플롯을 그려서 분포를 확인한다.

    Args:
        df (DataFrame): 데이터프레임 객체
        xname (str): x축에 사용할 컬럼명
        yname (str): y축에 사용할 컬럼명
        hue (str, optional): 색상을 구분할 기준이 되는 컬럼명. Defaults to None.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 4).
        dpi (int, optional): 그래프의 해상도. Defaults to 150.
    """
    plt.figure(figsize=figsize, dpi=dpi)
    sb.boxenplot(data=df, x=xname, y=yname, hue=hue)
    plt.grid(plt_grid)
    plt.show()
    plt.close()

def my_violinplot(df: DataFrame, xname: str = None, yname: str = None, hue = None, figsize: tuple = ( 10, 4), plt_grid : bool = True, plt_title : str = None, dpi: int = 150) -> None:
    """데이터프레임 내의 컬럼에 대해 바이올린플롯(상자그림+커널밀도)을 그려서 분포를 확인한다.

    Args:
        df (DataFrame): 데이터프레임 객체
        xname (str): x축에 사용할 컬럼명
        yname (str): y축에 사용할 컬럼명
        hue (str, optional): 색상을 구분할 기준이 되는 컬럼명. Defaults to None.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 4).
        dpi (int, optional): 그래프의 해상도. Defaults to 150.
    """
    plt.figure(figsize=figsize, dpi=dpi)
    sb.violinplot(data=df, x=xname, y=yname, hue=hue)
    plt.grid(plt_grid)
    plt.title(plt_title)
    plt.show()
    plt.close()
        
def my_pointplot(df: DataFrame, xname: str = None, yname: str = None, hue = None, figsize: tuple = (10, 4), plt_grid : bool = True, plt_title : str = None, dpi: int = 150) -> None:
    """데이터프레임 내의 컬럼에 대해 포인트플롯을 그려서 분포를 확인한다.

    Args:
        df (DataFrame): 데이터프레임 객체
        xname (str): x축에 사용할 컬럼명
        yname (str): y축에 사용할 컬럼명
        hue (str, optional): 색상을 구분할 기준이 되는 컬럼명. Defaults to None.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 4).
        dpi (int, optional): 그래프의 해상도. Defaults to 150.
    """
    plt.figure(figsize=figsize, dpi=dpi)
    sb.pointplot(data=df, x=xname, y=yname, hue=hue)
    plt.grid(plt_grid)
    plt.title(plt_title)
    plt.show()
    plt.close()

def my_jointplot(df: DataFrame, xname: str = None, yname: str = None, hue = None, figsize: tuple = (10, 4), plt_title : str = None, dpi: int = 150) -> None:
    """데이터프레임 내의 컬럼에 대해 산점도와 히스토그램을 함께 그려서 관계를 확인한다.

    Args:
        df (DataFrame): 데이터프레임 객체
        xname (str): x축에 사용할 컬럼명
        yname (str): y축에 사용할 컬럼명
        hue (str, optional): 색상을 구분할 기준이 되는 컬럼명. Defaults to None.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 4).
        dpi (int, optional): 그래프의 해상도. Defaults to 150.
    """
    g = sb.jointplot(data=df, x=xname, y=yname, hue=hue)
    g.fig.set_figwidth(figsize[0])
    g.fig.set_figheight(figsize[1])
    g.fig.set_dpi(dpi)
    plt.title(plt_title)
    plt.show()
    plt.close()
    
def my_heatmap(data: DataFrame, cmap = 'coolwarm', figsize: tuple = (10, 4), plt_title : str = None, dpi: int = 150) -> None:
    """데이터프레임 내의 컬럼에 대해 히트맵을 그려서 관계를 확인한다.

    Args:
        data (DataFrame): 데이터프레임 객체
        cmap (str, optional): 칼라맵. Defaults to 'coolwarm'.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 4).
        dpi (int, optional): 그래프의 해상도. Defaults to 150.
    """
    plt.figure(figsize=figsize, dpi=dpi)
    sb.heatmap(data, annot=True, cmap=cmap, fmt='.2g')
    plt.title(plt_title)
    plt.show()
    plt.close()

def my_convex_hull(data: DataFrame, xname: str = None, yname: str = None, * , hue: str , cmap:str = 'coolwarm', plt_grid : bool = True, plt_title : str = None, figsize: tuple = (10, 4), dpi: int = 150):

    """데이터프레임 내의 컬럼에 대해 외곽선을 그려서 군집을 확인한다.

    Args:
        data (DataFrame): 데이터프레임 객체
        xname (str): x축에 사용할 컬럼명
        yname (str): y축에 사용할 컬럼명
        hue (str): 색상을 구분할 기준이 되는 컬럼명
        cmap (str, optional): 칼라맵. Defaults to 'coolwarm'.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 4).
        dpi (int, optional): 그래프의 해상도. Defaults to 150.
    """
    plt.figure(figsize=figsize, dpi=dpi)
    
    # 군집별 값의 종류별로 반복 수행
    for c in data[hue].unique():
        # 한 종류만 필터링한 결과에서 두 변수만 선택
        df_c = data.loc[data[hue] == c, [xname, yname]]
        
        # 외각선 좌표 계산
        hull = ConvexHull(df_c)
        
        # 마지막 좌표 이후에 첫 번째 좌표를 연결
        points = np.append(hull.vertices, hull.vertices[0])
        
        plt.plot(df_c.iloc[points, 0], df_c.iloc[points, 1], linewidth=1, linestyle=":")
        plt.fill(df_c.iloc[points, 0], df_c.iloc[points, 1], alpha=0.1)
        
    sb.scatterplot(data=data, x=xname, y=yname, hue=hue, palette=cmap)
    
    plt.grid(plt_grid)
    plt.title(plt_title)
    plt.show()
    plt.close()
    
def my_kde_confidence_interval(data: DataFrame, clevel=0.95, figsize: tuple = (10, 4), plt_grid : bool = True, plt_title : str = None, dpi: int = 150) -> None:
    """커널밀도추정을 이용하여 신뢰구간을 그려서 분포를 확인한다.

    Args:
        data (DataFrame): 데이터프레임 객체
        clevel (float, optional): 신뢰수준. Defaults to 0.95.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 4).
        dpi (int, optional): 그래프의 해상도. Defaults to 150.
    """
    plt.figure(figsize=figsize, dpi=dpi)

    # 데이터 프레임의 컬럼이 여러 개인 경우 처리
    for c in data.columns:
        column = data[c]
        #print(column)
        max = column.max()      # 최대값
        dof = len(column) - 1   # 자유도
        sample_mean = column.mean()  # 표본평균
        sample_std = column.std(ddof=1) # 표본표준편차
        sample_std_error = sample_std / sqrt(len(column)) # 표본표준오차
        #print(max, dof, sample_mean, sample_std, sample_std_error)
        
        # 신뢰구간
        cmin, cmax = t.interval(clevel, dof, loc=sample_mean, scale=sample_std_error)

        # 현재 컬럼에 대한 커널밀도추정
        sb.kdeplot(data=column)

        # 그래프 축의 범위
        xmin, xmax, ymin, ymax = plt.axis()

        # 신뢰구간 그리기
        plt.plot([cmin, cmin], [ymin, ymax], linestyle=':')
        plt.plot([cmax, cmax], [ymin, ymax], linestyle=':')
        plt.fill_between([cmin, cmax], y1=ymin, y2=ymax, alpha=0.1)

        # 평균 그리기
        plt.plot([sample_mean, sample_mean], [0, ymax], linestyle='--', linewidth=2)

        plt.text(x=(cmax-cmin)/2+cmin,
                y=ymax,
                s="[%s] %0.1f ~ %0.1f" % (column.name, cmin, cmax),
                horizontalalignment="center",
                verticalalignment="bottom",
                fontdict={"size": 10, "color": "red"})

    plt.ylim(ymin, ymax*1.1)
    plt.grid(plt_grid)
    plt.title(plt_title)
    plt.show()
    plt.close()

def my_pvalue1_anotation(data : DataFrame, target: str, hue: str, pairs: list, test: str = "t-test_ind", text_format: str = "star", loc: str = "outside", figsize: tuple=(10, 4), dpi: int=150) -> None:
    """데이터프레임 내의 컬럼에 대해 상자그림을 그리고 p-value를 함께 출력한다.

    Args:
        data (DataFrame): 데이터프레임 객체
        target (str): 종속변수에 대한 컬럼명
        hue (str): 명목형 변수에 대한 컬럼명
        pairs (list, optional): 비교할 그룹의 목록. 명목형 변수에 포함된 값 중에서 비교 대상을 [("A","B")] 형태로 선정한다.
        test (str, optional): 검정방법. Defaults to "t-test_ind".
            - t-test_ind(독립,등분산), t-test_welch(독립,이분산)
            - t-test_paired(대응,등분산), Mann-Whitney(대응,이분산), Mann-Whitney-gt, Mann-Whitney-ls
            - Levene(분산분석), Wilcoxon, Kruskal
        text_format (str, optional): 출력형식(full, simple, star). Defaults to "star".
        loc (str, optional): 출력위치(inside, outside). Defaults to "outside".
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 4).
        dpi (int, optional): 그래프의 해상도. Defaults to 150.
    """
    plt.figure(figsize=figsize, dpi=dpi)
    ax = sb.boxplot(data=data, x=hue, y=target)

    annotator = Annotator(ax, data=data, x=hue, y=target, pairs=pairs)
    annotator.configure(test=test, text_format=text_format, loc=loc)
    annotator.apply_and_annotate()

    sb.despine()
    plt.show()
    plt.close()

