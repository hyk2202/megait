from tabulate import tabulate    
from pandas import DataFrame, read_excel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



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