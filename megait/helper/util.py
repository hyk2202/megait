import numpy as np
from tabulate import tabulate
from pandas import DataFrame, read_excel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def my_pretty_table(data: DataFrame) -> None:
    print(tabulate(data, headers='keys', tablefmt='psql',showindex=True, numalign="right"))

def my_read_excel(path: str, index_col: str=None, info: bool = True) -> DataFrame:
    """엑셀 파일을 데이터 프레임으로 로드하고 정보를 출력한다
    
    Args:
        path (str): 엑셀 파일의 경로 (혹은 URL)
        index_col (str, optional) : 인덱스 필드의 이름. Defaults to None.
        info (bool, optional) : True일 경우 정부 정보 출력. Defaults to True.

    Returns:
        DataFrame : 데이터프레임 객체
    """
    if index_col:
        data : DataFrame = read_excel(path, index_col = index_col)
    else:
        data : DataFrame = read_excel(path)
    
    if info:
        print("데이터프레임 크기: 행 수: {0}, 열 수: {1}".format(data.shape[0], data.shape[1]),end = '\n')

        print("\n데이터프레임 상위 5개 행",end = '\n\n')
        my_pretty_table(data.head())

        print("\n데이터프레임 하위 5개 행",end = '\n\n')
        my_pretty_table(data.tail())

        print("\n기술통계",end = '\n\n')
        desc = data.describe().T
        desc['nan'] = data.isnull().sum()
        my_pretty_table(data.describe())

    return data

def my_standard_scaler(data: DataFrame, yname: str = None) -> DataFrame:
    """데이터프레임의 연속형 변수에 대해 표준화를 수행한다.

    Args:
        data (DataFrame): 데이터프레임 객체
        yname (str, optional): 종속변수의 컬럼명. Defaults to None.
        
    Returns:
        DataFrame: 표준화된 데이터프레임
    """
    
    df = data.copy()
    
    if yname:
        y = df[yname]
        df = df.drop(yname, axis=1)
        
    category_fields = []
    for f in df.columns:
        if df[f].dtypes not in ['int', 'int32', 'int64', 'float', 'float32', 'float64']:
            category_fields.append(f)
    
    cate = df[category_fields]
    df = df.drop(category_fields, axis=1)
    
    scaler = StandardScaler()
    std_df = DataFrame(scaler.fit_transform(df), index=data.index, columns=df.columns)
    
    if category_fields:
        std_df[category_fields] = cate
        
    std_df[yname] = y
    
    return std_df

def my_train_test_split(data : DataFrame, yname : str = 'y', test_size : float = 0.3, random_state : int = 123, scalling : bool = False) -> tuple:
   '''데이터프레임을 학습용 데이터와 테스트용 데이터로 나눈다.

    Args:
        data (DataFrame): 데이터프레임 객체
        yname (str, optional): 종속변수의 컬럼명. Defaults to 'y'.
        test_size (float, optional): 검증 데이터의 비율(0~1). Defaults to 0.3.
        random_state (int, optional): 난수 시드. Defaults to 123.
        scalling (bool, optional): True일 경우 표준화를 수행한다. Defaults to False.

    Returns:
        tuple: x_train, x_test, y_train, y_test

   '''
      
   x = data.drop(yname,axis=1)
   y = data[yname]
   x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size, random_state=random_state)
   
   if scalling:
       scaler = StandardScaler()
       x_train = DataFrame(scaler.fit_transform(x_train), index=x_train.index, columns=x_train.columns)
       x_test = DataFrame(scaler.transform(x_test), index=x_test.index, columns=x_test.columns)

   return (x_train, x_test, y_train, y_test)

def set_category(data : DataFrame, *args : str) -> DataFrame:
    """카테고리 데이터를 설정한다.

    Args:
        data (DataFrame): 데이터프레임 객체
        *args (str): 컬럼명 목록

    Returns:
        DataFrame: 카테고리 설정된 데이터프레임
    """
    df = data.copy()

    for k in args:
        df[k] = df[k].astype('category')

    return df

def my_unmelt(data: DataFrame, id_vars: str ='class', value_vars: str='values') -> DataFrame:
    """두 개의 컬럼으로 구성된 데이터프레임에서 하나는 명목형, 나머지는 연속형일 경우
    명목형 변수의 값에 따라 고유한 변수를 갖는 데이터프레임으로 변환한다.
    
    Args:
        data (DataFrame): 데이터프레임
        id_vars (str, optional): 명목형 변수의 컬럼명. Defaults to 'class'.
        value_vars (str, optional): 연속형 변수의 컬럼명. Defaults to 'values'.

    Returns:
        DataFrame: 변환된 데이터프레임
    """
    result = data.groupby(id_vars)[value_vars].apply(list)
    mydict = {}

    for i in result.index:
        mydict[i] = result[i]
        
    return DataFrame(mydict)

def replace_missing_value(data: DataFrame, strategy: str = 'mean', fill_value : any = None) -> DataFrame:
    """결측치를 대체하여 데이터프레임을 재구성한다.

    Args:
        data (DataFrame): 데이터프레임
        strategy (["median", "mean", "most_frequent", "constant"], optional): 대체방법. Defaults to 'mean'.
        fill_value (str or numerical value): 상수로 대체할 경우 지정할 값.Defaults to 'None'

    Returns:
        DataFrame: _description_
    """
    # 결측치 처리 규칙 생성
    imr = SimpleImputer(missing_values=np.nan, strategy=strategy, fill_value=fill_value)
    
    # 결측치 처리 규칙 적용 --> 2차원 배열로 반환됨
    df_imr = imr.fit_transform(data.values)
    
    # 2차원 배열을 데이터프레임으로 변환 후 리턴
    return DataFrame(df_imr, index=data.index, columns=data.columns)
    
def get_outlier_table(data: DataFrame, *fields: str) -> DataFrame:
    """데이터프레임의 사분위수와 결측치 경계값을 구한다.
    함수 호출 전 상자그림을 통해 결측치가 확인된 필드에 대해서만 처리하는 것이 좋다.

    Args:
        data (DataFrame): 데이터프레임
        *fields (str): 컬럼명 목록

    Returns:
        DataFrame: IQ
    """
    if not fields:
        fields = data.columns

    result = []
    for f in fields:
        # 숫자 타입이 아니라면 건너뜀
        if data[f].dtypes not in ['int', 'int32', 'int64', 'float', 'float32', 'float64']:
            continue
        
        # 사분위수
        q1 = data[f].quantile(q=0.25)
        q2 = data[f].quantile(q=0.5)
        q3 = data[f].quantile(q=0.75)
        
        # 결측치 경계
        iqr = q3 - q1
        down = q1 - 1.5 * iqr
        up = q3 + 1.5 * iqr
        
        iq = {
            'FIELD': f,
            'Q1': q1,
            'Q2': q2,
            'Q3': q3,
            'IQR': iqr,
            'UP': up,
            'DOWN': down
        }
        
        result.append(iq)
        
    return DataFrame(result).set_index('FIELD')

def replace_outliner(data: DataFrame, *fields: str) -> DataFrame:
    """이상치 경계값을 넘어가는 데이터를 경계값으로 대체한다.

    Args:
        data (DataFrame): 데이터프레임
        *fields (str): 컬럼명 목록

    Returns:
        DataFrame: 이상치가 경계값으로 대체된 데이터 프레임
    """
    
    # 이상치 경계값을 구한다.
    outliner_table = get_outlier_table(data, *fields)
    
    # 원본 데이터 프레임 복사
    df = data.copy()
    
    # 이상치가 발견된 필드에 대해서만 처리
    for f in outliner_table.index:
        df.loc[df[f] < outliner_table.loc[f, 'DOWN'], f] = outliner_table.loc[f, 'DOWN']
        df.loc[df[f] > outliner_table.loc[f, 'UP'], f] = outliner_table.loc[f, 'UP']
        
    return df

def replace_outliner_to_nan(data: DataFrame, *fields: str) -> DataFrame:
    """이상치를 결측치로 대체한다.

    Args:
        data (DataFrame): 데이터프레임
        *fields (str): 컬럼명 목록

    Returns:
        DataFrame: 이상치가 결측치로 대체된 데이터프레임
    """
    
    # 이상치 경계값을 구한다.
    outliner_table = get_outlier_table(data, *fields)
    
    # 원본 데이터 프레임 복사
    df = data.copy()
    
    # 이상치가 발견된 필드에 대해서만 처리
    for f in outliner_table.index:
        df.loc[df[f] < outliner_table.loc[f, 'DOWN'], f] = np.nan
        df.loc[df[f] > outliner_table.loc[f, 'UP'], f] = np.nan
        
    return df

def replace_outliner_to_mean(data: DataFrame, *fields: str) -> DataFrame:
    """이상치를 평균값으로 대체한다.

    Args:
        data (DataFrame): 데이터프레임
        *fields (str): 컬럼명 목록

    Returns:
        DataFrame: 이상치가 평균값으로 대체된 데이터프레임
    """
    
    df = replace_outliner_to_nan(data, *fields)
    
    if not fields:
        fields = data.columns

    caterogy_type = []
    for f in fields:
        # 카테고리 타입만 골라냄
        if data[f].dtypes not in ['int', 'int32', 'int64', 'float', 'float32', 'float64']:
            caterogy_type.append(f)
            
    # 카테고리 타입을 제외한 필드에 대해서만 처리
    df2 = df.drop(caterogy_type, axis=1)
    df3 = replace_missing_value(df2, 'mean')
    
    # 원래의 데이터프레임(df)에다가 결측치를 평균값으로 대체한 데이터프레임(df3)을 덮어씌움
    df[df3.columns] = df3
    return df
