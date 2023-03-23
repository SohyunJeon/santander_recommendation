import pandas as pd
import numpy as np
import warnings
import os
import configparser
import src.advanced.setting as setting

np.random.seed(2023)
warnings.filterwarnings(action='ignore')
config = setting.read_config()

if __name__ == '__main__':
    train = pd.read_csv(os.path.join(config['path']['data'], 'train_ver2.csv'))
    test = pd.read_csv(os.path.join(config['path']['data'], 'test_ver2.csv'))

    finance_feats = [x for x in train.columns if 'ult1' in x]
    test[finance_feats] = 0

    total_df = pd.concat([train, test])

    # 단일 값 변수 제거
    not_use_cols = ['tipodom']
    total_df = total_df.drop(not_use_cols, axis=1)

    # 불규칙적인 space 제거
    total_df = total_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    total_df.to_csv(os.path.join(config['path']['data'], 'total.csv'), index=False)