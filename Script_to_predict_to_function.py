import cv2

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import re

import pymorphy2



import varname

import advertools as adv



import matplotlib.pyplot as plt #for visualization

import seaborn as sns #for visualization

import datetime

import ast 

import dostoevsky

import os

from sklearn.ensemble import GradientBoostingRegressor

from stop_words import get_stop_words

import string



# pd.set_option('display.max_rows', None)



pd.set_option('display.max_colwidth', None)

pd.set_option('display.max_columns', None)



from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

from sklearn import linear_model #линейные модели

from sklearn import metrics #метрики

from sklearn import preprocessing #предобработка

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LinearRegression

from numpy import mean

from numpy import absolute

from numpy import sqrt

import pandas as pd

from sklearn.manifold import TSNE

import swifter

# Визуализация

import plotly.express as px # для визуализации данных

import matplotlib.pyplot as plt # для отображения рукописных цифр

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import RobustScaler

import math



from swifter import set_defaults

set_defaults(

    npartitions=None,

    dask_threshold=1,

    scheduler="processes",

    progress_bar=False,

    progress_bar_desc=None,

    allow_dask_on_strings=False,

    force_parallel=False,

)



from dostoevsky.tokenization import RegexTokenizer

from dostoevsky.models import FastTextSocialNetworkModel



from PIL import Image



import timeit

import datetime

import time



from pymystem3 import Mystem



import joblib

def predict_of_forwards_views_rsh():

    def transform_info():

        data_prep = pd.read_csv('global_data_prep.csv')

        msg_aio = pd.read_csv('msg.csv')

        data_prep_zero = pd.DataFrame(data=np.zeros((1,data_prep.shape[1])), columns=data_prep.columns)

        data_prep_zero.loc[0,'date'] = msg_aio.loc[3,'1']

        data_prep_zero['date'] = data_prep_zero['date'].apply(lambda secs: datetime.datetime.fromtimestamp(int(secs)).strftime('%Y-%m-%d %H:%M:%S'))

        data_prep_zero.loc[0,'message'] = msg_aio.loc[8,'1']

        data_prep_zero.loc[0,'media'] = msg_aio.loc[7,'1']

        with open('test.txt', 'r') as output:
            
            a = output.readline()
            
        data_prep_zero.loc[0,'file_name_in_dir'] = a

        data_prep_zero['entities'] = '[]'
        
        return data_prep_zero

    data_prep = transform_info()

    head = pd.read_csv('table.csv')

    data_zero = pd.DataFrame(data=np.zeros((data_prep.shape[0],head.shape[1])), columns=head.columns)

    data_prep.reset_index(drop=True, inplace=True)

    # SENTIMENTS ==============================================================================================

    tokenizer = RegexTokenizer()

    model = FastTextSocialNetworkModel(tokenizer=tokenizer)

    def sentiment_points(message):
        
        try:
            result = model.predict([message], k=5)
            
            list_of_results = [result[0]['positive'], result[0]['negative'], result[0]['neutral'], result[0]['skip'], result[0]['speech']]
        
        except:
            
            list_of_results = [0,0,0,0,0]
            
        return list_of_results

    sents = data_prep['message'].apply(sentiment_points)

    sent_tnls = ['positive', 'negative', 'neutral', 'skip', 'speech']

    data_zero[sent_tnls] = pd.DataFrame(data=list(sents), columns=sent_tnls)

    # DATES ============================================================================================================================

    data_zero[['second','minute','hour','day_of_week','day_of_year','month']] = pd.concat([
        
                        pd.DataFrame(np.array(pd.to_datetime(data_prep['date']).dt.second), columns=['second']),
                        
                        pd.DataFrame(np.array(pd.to_datetime(data_prep['date']).dt.minute), columns=['minute']),
                        
                        pd.DataFrame(np.array(pd.to_datetime(data_prep['date']).dt.hour), columns=['hour']),
                        
                        pd.DataFrame(np.array(pd.to_datetime(data_prep['date']).dt.day_of_week), columns=['day_of_week']),
                        
                        pd.DataFrame(np.array(pd.to_datetime(data_prep['date']).dt.day_of_year), columns=['day_of_year']),
                        
                        pd.DataFrame(np.array(pd.to_datetime(data_prep['date']).dt.month), columns=['month'])
                        
                        ], axis = 1)


    # EMOJIS ============================================================================================================================

    def update_message(message):
        
        if pd.isna(message):
            
            return ''
        
        return ' '.join(message.split('\n\n'))

    data_prep['msg_up'] = data_prep['message'].apply(update_message)

    emoji_cols = []

    for col in data_zero.columns:
        
        if 'in_msg' in col:
            
            emoji_cols.append(col)
            
    for col in emoji_cols:
        
        emoji = col.replace('in_msg','')
        
        data_zero[col] = data_prep['message'].swifter.apply(lambda msg: len(re.findall(emoji, str(msg))) if emoji in str(msg) else 0)
        

    # COLORS ==========================================================================================================================

    def mask_segmentation_main_color_update(img):
        
        picture = cv2.imread(f'pictest\\{img}')
        
        picture_rgb = cv2.cvtColor(picture, cv2.COLOR_BGR2RGB)
        
        picture_hsv = cv2.cvtColor(picture_rgb, cv2.COLOR_RGB2HSV)
        
        low_up_dict = {'blue':[(180/2, 0.15*255, 0.1*255),(255/2, 1*255, 1*255)],
                    
                    'blue-green':[(150/2, 0.15*255, 0.1*255),(180/2, 1*255, 1*255)],
                    
                    'purple':[(255/2, 0.5*255, 0.1*255),(310/2, 1*255, 1*255)],
                    
                    'light-purple':[(255/2, 0.15*255, 0.1*255),(310/2, 0.5*255, 1*255)],
                    
                    'yellow':[(45/2, 0.15*255, 0.1*255),(64/2, 1*255, 1*255)],
                    
                    'orange':[(11/2, 0.15*255, 0.75*255),(45/2, 1*255, 1*255)],
                    
                    'pink_1':[(0/2, 0*255, 0.1*255),(11/2, 0.7*255, 1*255)],
                    
                    'pink_2':[(351/2, 0*255, 0.1*255),(360/2, 0.7*255, 1*255)],
                    
                    'pink_3':[(310/2, 0.15*255, 0.1*255),(351/2, 1*255, 1*255)],
                    
                    'brown':[(11/2, 0.15*255, 0.1*255),(45/2, 1*255, 0.75*255)],
                    
                    'green':[(64/2, 0.15*255, 0.1*255),(150/2, 1*255, 1*255)],
                    
                    'black':[(0, 0*255, 0*255),(360/2,1*255,0.1*255)],
                    
                    'white':[(0, 0*255, 0.65*255),(360/2,0.15*255,1*255)],
                    
                    'grey':[(0, 0*255, 0.1*255),(360/2,0.15*255,0.65*255)]}
        
        fraction_color_dict = {}
        
        mask_pic = {}
        
        total_pixels = picture_hsv.shape[0] * picture_hsv.shape[1]
        
        
        for key in low_up_dict.keys():
            
            mask_pic[key] = cv2.inRange(picture_hsv, low_up_dict[key][0], low_up_dict[key][1])
            
            fraction_color_dict[key] = np.sum(mask_pic[key]) / 255 / total_pixels
        
        return fraction_color_dict


    def mp4_to_jpeg_in_filename(filename):
        
        if '.mp4' in str(filename):
            
            filename = f'{filename}.jpg'
            
        return filename

    data_prep['file_name_in_dir'] = data_prep['file_name_in_dir'].swifter.apply(mp4_to_jpeg_in_filename)

    print('====================================================================================================')

    def colors_segmenation(filename):
        
        try:
            
            colors_dict = mask_segmentation_main_color_update(filename)
            
        except:
            
            colors_dict = dict.fromkeys(['blue', 'blue-green', 'purple', 'light-purple', 'yellow', 'orange',
                                        'pink_1', 'pink_2', 'pink_3', 'brown', 'green',
                                        'black', 'white', 'grey'], 0)
            
        return list(colors_dict.values())

    clr_cols = ['blue', 'blue-green', 'purple', 'light-purple', 'yellow', 'orange',
                                        'pink_1', 'pink_2', 'pink_3', 'brown', 'green',
                                        'black', 'white', 'grey']

    colors_sgm = data_prep['file_name_in_dir'].apply(colors_segmenation)

    data_zero[clr_cols] = pd.DataFrame(data=list(colors_sgm), columns=clr_cols)

    def color_count(path):
        
        try:
            
            src = cv2.imread(f'pictest\\{path}')
            
            hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
            
            return pd.DataFrame(data=hsv.reshape(-1, hsv.shape[2])).drop_duplicates().shape[0]
            
        except:
            
            return 0
        
    cnt_clr_fnc = np.vectorize(color_count)

    cnt_clr = cnt_clr_fnc(data_prep['file_name_in_dir'].values)

    data_prep['color_cnt'] = cnt_clr

    data_zero['list_of_count_colors'] = data_prep['color_cnt']

    def color_hsv_mean(filename):
        
        try:
        
            src = cv2.imread(f'pictest\\{filename}')
            
            hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
            
            return list(pd.DataFrame(data=hsv.reshape(-1, hsv.shape[2])).mean())
    
        except:
            
            return [0,0,0]
        
    clr_means = data_prep['file_name_in_dir'].apply(color_hsv_mean)

    data_zero[['hue_hsv_mean','saturation_hsv_mean','value_hsv_mean']] = pd.DataFrame(data=list(clr_means), columns=['hue_hsv_mean','saturation_hsv_mean','value_hsv_mean'])


    # MEDIA PROPERTY =================================================================================================================

    def string_media_catch(str_media):
        
        liat_of_video_property = re.findall(r"'duration': \d+, 'width': \d+, 'height': \d+", str_media)[0]
        
        video_dict = ast.literal_eval("{" + liat_of_video_property + "}")
    
        return list(video_dict.values())

    data_prep['list_video'] = data_prep['media'].apply(string_media_catch)

    data_zero['duration'] = data_prep['list_video'].apply(lambda x: x[0])

    data_zero['w'] = data_prep['list_video'].apply(lambda x: x[1])

    data_zero['h'] = data_prep['list_video'].apply(lambda x: x[2])

    data_prep.drop(['list_video'], axis = 1, inplace=True)

    data_zero['weight_to_height'] = data_zero['w'] / data_zero['h']

    #  Messages ======================================================================================================================

    def remove_emojis(data):
        
        emoj = re.compile("["
            
            u"\U0001F600-\U0001F64F"  # emoticons
            
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            
            u"\U00002500-\U00002BEF"  # chinese char
            
            u"\U00002702-\U000027B0"
            
            u"\U00002702-\U000027B0"
            
            u"\U000024C2-\U0001F251"
            
            u"\U0001f926-\U0001f937"
            
            u"\U00010000-\U0010ffff"
            
            u"\u2640-\u2642" 
            
            u"\u2600-\u2B55"
            
            u"\u200d"
            
            u"\u23cf"
            
            u"\u23e9"
            
            u"\u231a"
            
            u"\ufe0f"  # dingbats
            
            u"\u3030"
            
                    "]+", re.UNICODE)
        
        return re.sub(emoj, '', data)

    def update_message(message):
        
        if pd.isna(message):
        
            return ''
        
        return ' '.join(message.split('\n\n'))

    msg_up = data_prep['message'].apply(update_message)

    drop_emoji = msg_up.apply(remove_emojis)

    lms_msg = drop_emoji.apply(lambda msg_text: re.sub('[^А-Яа-яA-Za-z0-9 -]+', '', msg_text).replace('   ',' ').replace('  ',' ').lower())

    # Lemmatize

    text_lem = '||'.join(list(lms_msg))

    m = Mystem()

    lemmas = m.lemmatize(text_lem)

    lemms = ''.join(lemmas).strip().split('||')

    data_prep['lemms'] = pd.DataFrame(data=list(lemms), columns=['lemms'])

    lemms_df = pd.DataFrame(data=list(lemms), columns=['lemms'])

    data_zero['re_lemmas_count'] = data_prep['lemms'].swifter.apply(lambda txt: 0 if len(txt) == 0 else len(str(txt).split(' ')))

    data_zero['re_lemmas_count']

    max_str = 301 

    wrds_cnt_df = pd.read_excel('wrds_cnt_df.xlsx')

    wrds_cnt_lst = list(wrds_cnt_df['words'])

    lst_lm = []

    for lemmas in data_prep['lemms']:
        
        lemmas = str(lemmas).encode('utf-8').decode('utf-8')
        
        lem_lst = lemmas.split(' ')
        
        print(lem_lst)
        
        zeros = list(np.zeros(max_str))
            
        if lem_lst[0] == '':
            
            lst_lm.append(zeros) 
            
            continue
            
        for index in range(len(lem_lst)):
                
            if lem_lst[index] in wrds_cnt_lst:
                
                zeros[index] = wrds_cnt_lst.index(lem_lst[index]) + 1
            
            else:
                
                zeros[index] = 0
            
        lst_lm.append(zeros)
        

    cols_wrds = []

    for index in range(max_str):

        cols_wrds.append(f'word_number_{index + 1}')
        
    data_zero[cols_wrds] = pd.DataFrame(data = list(lst_lm), columns=cols_wrds).astype(int) / (wrds_cnt_df.shape[0])

    # Entities ===================================================================================================================

    def entities_get(entities):
        
        ents_list = []
        
        ents = re.findall(r"'_': '\w+', 'offset': \d+, 'length': \d+", str(entities))
        
        for ent in ents:
            
            ents_list.append(ast.literal_eval("{" + ent + "}"))
            
        return ents_list

    data_prep['ent_dicts'] = data_prep['entities'].apply(entities_get) 

    msg_ent_type = {'MessageEntityBold',
    
    'MessageEntityCode',
    
    'MessageEntityCustomEmoji',
    
    'MessageEntityEmail',
    
    'MessageEntityHashtag',
    
    'MessageEntityItalic',
    
    'MessageEntityMention',
    
    'MessageEntitySpoiler',
    
    'MessageEntityStrike',
    
    'MessageEntityTextUrl',
    
    'MessageEntityUnderline',
    
    'MessageEntityUrl'}

    def get_msg_ents(dickts):
        
        msg_ent_type = {}
        
        for dickt in dickts:
            
            if dickt['_'] not in msg_ent_type:
                
                msg_ent_type[dickt['_']] = 1
                
            else:
                
                msg_ent_type[dickt['_']] += 1
                
        return msg_ent_type

    data_prep['types_ent'] = data_prep['ent_dicts'].swifter.apply(get_msg_ents)

    for type_ent in msg_ent_type:
        
        data_zero[type_ent] = data_prep['types_ent'].swifter.apply(lambda dickt: dickt[type_ent] if type_ent in dickt else 0)
        

    ## Additional =================================================================================================================

    ## Day of month and days between day of message and today =====================================================================

    data_prep['date'] = pd.to_datetime(data_prep['date'])

    today = datetime.datetime.now()

    today.replace(tzinfo=datetime.timezone.utc)

    pd.to_datetime(today).replace(tzinfo=None) 

    data_prep['dtdays'] = (today.replace(tzinfo=None) - data_prep.date).dt.days

    data_prep['day_of_month'] = pd.to_datetime(data_prep['date']).dt.day

    data_zero['dtdays'] = data_prep['dtdays']

    data_zero['day_of_month'] = data_prep['day_of_month']

    ## HSV color model - std, min, max, median of Hue, Saturation, Value  ==================================================================

    def color_hsv_std(filename):
        
        try:
        
            src = cv2.imread(f'pictest\\{filename}')
            
            hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
            
            return list(pd.DataFrame(data=hsv.reshape(-1, hsv.shape[2])).std())
    
        except:
            
            return [0,0,0]
        

    clr_stds = data_prep['file_name_in_dir'].apply(color_hsv_std)

    df_std_hsv = pd.DataFrame(data=list(clr_stds), columns=['h_std','s_std','v_std'])

    data_zero_2 = pd.concat([data_zero, df_std_hsv], axis=1)

    def color_hsv_median_min_max(filename):
        
        try:
        
            src = cv2.imread(f'pictest\\{filename}')
            
            hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
            
            testing = pd.DataFrame(data=hsv.reshape(-1, hsv.shape[2]))
            
            return list(testing.median()) + list(testing.min()) + list(testing.max())
    
        except:
            
            return [0,0,0,0,0,0,0,0,0]
        

    clr_median_min_max = data_prep['file_name_in_dir'].apply(color_hsv_median_min_max)

    df_hsv_diff = pd.DataFrame(data=list(clr_median_min_max), columns=['h_median','s_median','v_median','h_min','s_min','v_min','h_max','s_max','v_max'])

    data_zero_3 = pd.concat([data_zero_2, df_hsv_diff], axis=1)

    ## Additional days ===================================================================================================================

    data_zero_3['morinig'] = data_zero_3['hour'].swifter.apply(lambda hour: 1 if hour in [6,7,8,9,10,11] else 0)

    data_zero_3['day'] = data_zero_3['hour'].swifter.apply(lambda hour: 1 if hour in [12,13,14,15,16,17] else 0)

    data_zero_3['evening'] = data_zero_3['hour'].swifter.apply(lambda hour: 1 if hour in [18,19,20,21,22,23] else 0)

    data_zero_3['night'] = data_zero_3['hour'].swifter.apply(lambda hour: 1 if hour in [0,1,2,3,4,5] else 0)

    data_zero_3['summer'] = data_zero_3['month'].swifter.apply(lambda month: 1 if month in [6,7,8] else 0)

    data_zero_3['autumn'] = data_zero_3['month'].swifter.apply(lambda month: 1 if month in [9,10,11] else 0)

    data_zero_3['winter'] = data_zero_3['month'].swifter.apply(lambda month: 1 if month in [12,1,2] else 0)

    data_zero_3['spring'] = data_zero_3['month'].swifter.apply(lambda month: 1 if month in [3,4,5] else 0)

 ## RGB model mean and std ==========================================================================================================

    def color_rgb_mean_std(filename):
        
        try:
                        
            src = cv2.imread(f'pictest\\{filename}')
                            
            return [src[:,:,0].mean(), src[:,:,1].mean(), src[:,:,2].mean(), src[:,:,0].std(), src[:,:,1].std(), src[:,:,2].std()]
    
        except:
            
            return [0,0,0,0,0,0]
        

    rgb_means_std = data_prep['file_name_in_dir'].apply(color_rgb_mean_std)

    df_for_add = pd.DataFrame(data=list(rgb_means_std), columns=['r_mean', 'g_mean', 'b_mean','r_std', 'g_std', 'b_std'])

    data_zero_3 = pd.concat([data_zero_3, df_for_add], axis=1)

    print(data_zero_3.shape)

    ## Load model and predict ==============================================================================================================

    import pickle

    with open('csr.pkl', 'rb') as pkl_file:
        
        regressor_from_file = pickle.load(pkl_file)
        
    base = 1000

    y_pred = base ** regressor_from_file.predict(data_zero_3.drop(['forwards_to_views','dtdays'], axis=1))

    print(f'The ratio of reposts to views is predicted at the level of {y_pred[0]}')
    
    return y_pred[0]