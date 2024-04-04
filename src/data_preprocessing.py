import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_data(RAV):
    emotion = []
    gender = []
    path = []
    dir_list = os.listdir(RAV)
    for i in dir_list:
        fname = os.listdir(RAV + i)
        for f in fname:
            part = f.split('.')[0].split('-')
            emotion.append(int(part[2]))
            temp = int(part[6])
            if temp % 2 == 0:
                temp = "female"
            else:
                temp = "male"
            gender.append(temp)
            path.append(RAV + i + '/' + f)
    RAV_df = pd.DataFrame(emotion)
    RAV_df = RAV_df.replace({1: 'neutral', 2: 'neutral', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fear', 7: 'disgust', 8: 'surprise'})
    RAV_df = pd.concat([pd.DataFrame(gender), RAV_df], axis=1)
    RAV_df.columns = ['gender', 'emotion']
    RAV_df['labels'] = RAV_df.gender + '_' + RAV_df.emotion
    RAV_df['source'] = 'RAVDESS'
    RAV_df = pd.concat([RAV_df, pd.DataFrame(path, columns=['path'])], axis=1)
    RAV_df = RAV_df.drop(['gender'], axis=1)
    return RAV_df
