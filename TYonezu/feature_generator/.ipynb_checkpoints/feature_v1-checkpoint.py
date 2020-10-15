import pandas as pd
import glob
import os
import numpy as np
import gc

from .function import *
from sklearn.preprocessing import LabelEncoder

def label_encode(df, cols):
    for col in cols:
        le = LabelEncoder()
        tmp = df[col].fillna("NaN")
        df[col] = pd.Series(le.fit_transform(tmp), index=tmp.index)

    return df


class FeaturesMaker_v1(object):

    def __init__(self,target_col):
        self.name = "features_ver1"
        self.feature_exp = "simple features which "

        self.target_col = target_col
        self.necessary_col =  ["sig_id",'cp_type',"data_part"] + [target_col]

    def make_feature(self,df):

        # check existstance of necessary columns
        if check_columns(self.necessary_col,df.columns):

            # label encoding
            cols = ['cp_type']
            df = label_encode(df, cols=cols)


            # split train and test
            df = df.set_index(["sig_id"],drop=True)

            features = [c for c in df.columns if "g" in c]
            features = features + [c for c in df.columns if "c" in c]
            features = features + ["cp_type"]

            print("-- ",self.name," --")
            print("dim:",len(features))
            print("N:",len(df))
            print("-----------------")

            return {sub[0]:(sub[1][features],sub[1][self.target_col]) for sub in df.groupby(by="data_part")}

        else:
            return False
