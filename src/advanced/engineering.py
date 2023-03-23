import pandas as pd
import numpy as np
import warnings
import os
import math
import pickle
import src.advanced.setting as setting

from sklearn.preprocessing import LabelEncoder

np.random.seed(2023)
warnings.filterwarnings(action='ignore')
config = setting.read_config()

features = []
transformers = {}

def execute_label_encoding(data:pd.DataFrame, feat_name:str):
    data[feat_name] = data[feat_name].astype(str)
    if feat_name in transformers: # test: 기존 인코더가 있는 경우 사용
        data[feat_name] = transformers[feat_name].transform(data[feat_name])
    else:
        encoder = LabelEncoder().fit(data[feat_name])
        data[feat_name] = encoder.transform(data[feat_name])
        transformers[feat_name] = encoder
    features.append(feat_name)

def execute_onehot_encoding(data: pd.DataFrame, feat_name:str,
                            use_val:dict):
    for real_val, new_val in use_val.items():
        new_feat = f'{feat_name}_{new_val}'
        data[new_feat] = data[feat_name].map(lambda x: 1 if x == real_val
                                             else 0).astype(np.int8)
        features.append(new_feat)


def date_to_int(str_date: pd.Series) -> pd.Series:
    y, m, d = [int(x) for x in str_date.strip().split('-')]
    int_date = (y - 2015) * 12 + m
    return int_date



def preprocess_custom(data:pd.DataFrame) -> (pd.DataFrame):
    """
    1. numeric : 결측치 처리 및 단위, 분포 변환
        age, renta, cod_prov
        ind_actividad_cliente, indrel_1mes
    2. categorical : encoding
        2-1. LabelEncoder : unique값이 많은 변수에 적용
            canal_entrada(162), pais_residencia(118)
        2-2. OneHotEncoder: 이진 변수 생성-> 범주형 데이터 표현력 높임/ unique 개수-1개 생성
            indresi(2), indext(2), conyuemp(2), sexo(2), ind_empleado(5),
            ind_nuevo(2), segmento(3), indfall(2), tiprel_1mes(5), indrel(2)
    3. datetime : 날짜 관련 파생 변수 생성
        3-1. fecha_dato : 기록 날짜
            - 월, 년도 추출
            - 정수 변환
        3-2 fecha_alta : 첫 가입일
            - 월, 년도 추출
            - 가입 기간

    :param data:
    :return:
    """
    features = []

    ## 1. numeric
    # age : 결측치 처리
    data['age'] = data['age'].fillna(0.0).astype(np.int8)
    features.append('age')
    # antiguedad : 결측치 처리
    data['antiguedad'] = data['antiguedad'].\
        map(lambda x: 0.0 if x < 0 or np.isnan(x) else x).astype(np.int8)
    features.append('antiguedad')
    # renta_log : 분포 변환
    data['renta'] = data['renta'].fillna(1.0)
    data['renta_log'] = data['renta'].map(math.log)
    features.append('renta_log')
    # renta_rank : 소득 밀도 순위
    unique_val, unique_cnt = np.unique(data['renta_log'].map(lambda x: round(x, 2)),
                                       return_counts=True)
    val_rank = sorted(zip(unique_val, unique_cnt), key=lambda x: x[1], reverse=True)
    rank_map = {d[0]: rank+1 for d, rank in zip(val_rank, range(len(val_rank)))}
    data['renta_rank'] = data['renta_log'].map(lambda x: round(x, 2)).\
        map(lambda x: rank_map[x])
    features.append('renta_rank')
    # cod_prov : 우편번호
    data['cod_prov'] = data['cod_prov'].fillna(0.0).astype(np.int8)
    features.append('cod_prov')
    # ind_actividad_cliente : 활성지표 - na값 별도 추가
    data['ind_actividad_cliente'] = data['ind_actividad_cliente'].map(
        lambda x: 0.0 if np.isnan(x) else x+1.0).astype(np.int8)
    features.append('ind_actividad_cliente')
    # indrel_1mes : 월초 고객 상태
    data['indrel_1mes'] = data['indrel_1mes'].fillna(0.0)
    data['indrel_1mes'] = data['indrel_1mes'].map(
        lambda x: 5.0 if x == 'P' else float(x)).astype(np.int8)
    features.append('indrel_1mes')


    ## 2. categorical
    # LabelEncoding
    execute_label_encoding(data, 'canal_entrada')           # 고객 유입 채널
    execute_label_encoding(data, 'pais_residencia')         # 국적
    # OnehotEncoding
    execute_onehot_encoding(data, 'indresi', {'S': 's'})    # 거주국가 여부 : S, N
    execute_onehot_encoding(data, 'indext', {'S': 's'})     # 외국인 여부: S, N
    execute_onehot_encoding(data, 'conyuemp', {'S': 's'})   # 배우자지표: S, N
    execute_onehot_encoding(data, 'sexo', {'V': 'v'})       # 성별: V, H
    execute_onehot_encoding(data, 'ind_empleado',
                            {'N': 'n', 'B': 'b', 'F': 'f', 'A': 'a'})
                                                            # 고용상태: N, B, F, A, S
    execute_onehot_encoding(data, 'ind_nuevo', {'1.0': 'new'}) # 신규고객: 0.0, 1.0
    execute_onehot_encoding(data, 'segmento',
                            {'01 - TOP': 'top',
                             '02 - PARTICULARES': 'part'})
                            # 고객분류: 02 - PARTICULARES, 03 - UNIVERSITARIO, 01 - TOP
    execute_onehot_encoding(data, 'indfall', {'S': 's'})# 사망여부: S, N
    execute_onehot_encoding(data, 'tiprel_1mes',
                            {'I': 'i', 'A': 'a', 'P': 'p', 'R': 'r'})
                            # 월초고객타입: I,A, P, R, N
    execute_onehot_encoding(data, 'indrel', {1: '1'}) # 고객 등급: 1, 99

    ## 3. datetime
    # fecha_dato
    data['fecha_dato_year'] = data['fecha_dato'].map(
        lambda x: str(x).split('-')[0]).astype(np.int16)
    features.append('fecha_dato_year')
    data['fecha_dato_month'] = data['fecha_dato'].map(
        lambda x: str(x).split('-')[1]).astype(np.int8)
    features.append('fecha_dato_month')
    data['fecha_dato_int'] = data['fecha_dato'].map(date_to_int).astype(np.int8)
    # fecha_alta
    data['fecha_alta_year'] = data['fecha_alta'].map(
        lambda x: 0.0 if type(x) == float else str(x).split('-')[0]).astype(np.int16)
    features.append('fecha_alta_year')
    data['fecha_alta_month'] = data['fecha_alta'].map(
        lambda x: 0.0 if type(x) == float else str(x).split('-')[1]).astype(
        np.int8)
    features.append('fecha_alta_month')
    # duration
    data['apply_duration'] = \
        ((data['fecha_dato_year']*12) + data['fecha_dato_month']) - \
        ((data['fecha_alta_year'] * 12) + data['fecha_alta_month'])
    features.append('apply_duration')

    return data


def make_previous_data(data: pd.DataFrame, step: int, feats: list)\
        -> (pd.DataFrame, list):
    prev_df = data[['ncodpers', 'fecha_dato_int']]
    prev_df['fecha_dato_int'] = prev_df['fecha_dato_int'] + 1

    prev_feats = [f'{x}_prev_{step}' for x in feats]
    for prod, prev_prod in zip(feats, prev_feats):
        prev_df[prev_prod] = data[prod]
    return prev_df, prev_feats



def preprocess_product(data: pd.DataFrame):
    """
    1. 전처리
        - 결측값 0 대체
    2. lag-n 변수 생성
    3. lag-n summary 변수 생성


    :param data:
    :return:
    """
    n_lag: int = 5
    ## 1
    prods = [col for col in data.columns if 'ult1' in col]
    data[prods] = data[prods].fillna(0.0).astype(np.int8)

    ## 2.make lag features
    prev_dfs = []
    prev_feats_list = []
    for step in range(1, n_lag+1):
        prev_df, prev_feats = make_previous_data(data, step, prods)
        prev_dfs.append(prev_df)
        prev_feats_list.extend(prev_feats)
    use_prev_feats = [x for x in prev_feats_list if ('1' in x) or ('2' in x)]
    features.extend(use_prev_feats)

    for i, prev_df in enumerate(prev_dfs):
        data = data.merge(prev_df, on=['ncodpers', 'fecha_dato_int'],
                          how='inner')

    ## 3. summary features
    std_range = [(1, 3), (1, 5), (2, 5)]
    minmax_range = [(2, 3), (2, 5)]
    for prod in prods:
        for start, end in std_range:
            range_feats = [f'{prod}_prev_{i}' for i in range(start, end+1)]
            std_feat = f'{prod}_std_{start}_{end}'
            data[std_feat] = np.nanstd(data[range_feats].values, axis=1)
            features.append(std_feat)

        for start, end in minmax_range:
            range_feats = [f'{prod}_prev_{i}' for i in range(start, end+1)]
            min_feat = f'{prod}_min_{start}_{end}'
            data[min_feat] = np.nanmin(data[range_feats].values, axis=1)
            features.append(min_feat)
            max_feat = f'{prod}_max_{start}_{end}'
            data[max_feat] = np.nanmax(data[range_feats].values, axis=1)
            features.append(max_feat)

    final_columns = ["ncodpers", "fecha_dato", 'fecha_dato_int'] + prods + features
    final_data = data[final_columns]

    return final_data






if __name__ == '__main__':
    total_df = pd.read_csv(os.path.join(config['path']['data'], 'total.csv'))

    df = preprocess_custom(total_df)
    df = preprocess_product(df)

    with open(os.path.join(config['path']['model'], 'features.pkl'), 'wb') as f:
        pickle.dump(features, f)
    df.to_csv(os.path.join(config['path']['model'], 'engineered_df.csv'), index=False)