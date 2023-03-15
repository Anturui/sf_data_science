# from pandas._config.config import Display
from telethon import TelegramClient, sync
import re 
import ast
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import string
from datetime import datetime
import os
import re

# api_id = 25673476
# api_hash = '5b9d46fbf4ea78db18fd54a588a31fd7'

api_id = 26313760
api_hash = '8b8f5a9619f9bb388b1915055f3e5379'

client = TelegramClient('session_name', api_id, api_hash)
client.start()


from telethon.tl.functions.messages import (GetHistoryRequest)
from telethon.tl.types import (
PeerChannel
)
import numpy as np #for matrix calculations
import pandas as pd #for data analysis and preprocessing
import time

import traceback

offset_id = 0
limit = 100
all_messages = []
total_messages = 0
total_count_limit = 0

i = 0


message_str = []

list_of_file_name_in_directory = []

message_dict = []



# df_for_add_pars = pd.read_csv('WB analisys\\for_add_parse.csv')
# list_for_add = list(df_for_add_pars['list_chan_to_add_parsing'])[6:9]

list_for_add =  ['https://t.me/MesExpertBot']

# import sys
# def callback(current, total):
#     print('Downloaded', current, 'out of', total,
#           'bytes: {:.2%}'.format(current / total))
#     if current > 300000:
#         sys.exit('fuck off')
import logging

log = logging.getLogger(__name__)

i = 0 

print()

index_like = 0

for channel_username in list_for_add:

    chl_nm = channel_username.replace('https://t.me/','@')
    
    message_dict = []
    
    message_str = []
    
    list_of_file_name_in_directory = []
    
    try:
        
        for msg in client.iter_messages(channel_username, limit = None):

            
            
            i += 1
        
            if not msg.forwards:
                
                continue
            
            try:
                
                if msg.to_dict()['forwards'] and msg.to_dict()['views'] and msg.to_dict()['media']:

                    string_media = str(msg.to_dict()['media'])

                    if 'video/mp4' in string_media:
                        
                        index_like += 1
                        
                        print(i)
                        
                        liat_of_video_property = re.findall(r"'duration': \d+, 'w': \d+, 'h': \d+", string_media)[0]
                        
                        video_dict = ast.literal_eval("{" + liat_of_video_property + "}")
                        
                                                
                        
                        if video_dict['h'] < 10000:
                            
                            print(video_dict)
                            
                            try: 
                                
                                time.sleep(0.3 + float(np.random.rand(1)))     
                                
                                client.download_media(msg, file = f'pictg\\picture_{msg.id}_{i}_{index_like}_{chl_nm}_of_message.jpg',thumb = -1)
                                
                                list_of_file_name_in_directory.append(f'picture_{msg.id}_{i}_{index_like}_{chl_nm}_of_message.jpg')
                                
                            except: 
                                
                                list_of_file_name_in_directory.append('none')
                                
                                print(traceback.format_exc())
                            
                            message_dict.append(msg.to_dict())
                            
                            message_str.append(msg)
                            
                            print(msg.id,list_of_file_name_in_directory[-1],i,index_like)
                        
                
            
            except:
                
                print(traceback.format_exc())
                
                log.exception('Это сообщение об ошибке:')
                
                print('*******************************************************************')

            
            break
            
        try:
            
            data_dict = pd.DataFrame()
            
            for dictionary in message_dict:
                
                if data_dict.shape[0] == 0: 
                    
                    data_dict = pd.DataFrame(data=[[*dictionary.values()]],columns=[*dictionary.keys()])
                    
                else:
                    
                    data_sub_dict = pd.DataFrame(data=[[*dictionary.values()]],columns=[*dictionary.keys()])
                    
                    data_dict = pd.concat([data_dict,data_sub_dict])
            
            string_to_file_abstracts_date = str(datetime.now().time().hour) + '_' + str(datetime.now().time().minute) + '_' + str(datetime.now().time().second)

            data_dict.to_csv(f'data_few_channels_add_{string_to_file_abstracts_date}_{chl_nm}.csv',index=False)
            
            try:   
                
                data_dict['file_name_in_dir'] = list_of_file_name_in_directory
                
                string_to_file_abstracts_date = str(datetime.now().time().hour) + '_' + str(datetime.now().time().minute)

                data_dict.to_csv(f'data_few_channels_add_{string_to_file_abstracts_date}_{chl_nm}.csv',index=False)
                
            except:

                try:

                    data_dict =  pd.DataFrame(message_dict)

                    data_dict['file_name_in_dir'] = list_of_file_name_in_directory

                    data_dict.to_csv(f'data_few_channels_add_{string_to_file_abstracts_date}_{chl_nm}.csv',index=False)
                    
                    print(traceback.format_exc())
                    
                    log.exception('Это сообщение об ошибке:')
                    
                    print('no_data')
                
                except: 

                    print(traceback.format_exc())
                        
                    log.exception('Это сообщение об ошибке:')
                        
                    print('no_data') 
                
        except:


            
            print(traceback.format_exc())
            
            log.exception('Это сообщение об ошибке:')
           
            print('fuck off')
    
            
    
    except:
        
        print(traceback.format_exc())
        
        print('off')
        
    
     





# data_dict.to_csv(f'data_few_channels_add_reserve.csv',index=False)

# channel_list = [ 'https://t.me/NatureTravelVacationPictures',
# 'https://t.me/Planet_Earth',
# 'https://t.me/animals_101',
# 'https://t.me/livingocean',
# 'https://t.me/internationalgeographic',
# 'https://t.me/beautifulll_place',
# 'https://t.me/Gods_Nature 700',
# 'https://t.me/wanderlustguide',
# 'https://t.me/travelpicturesworldwide',
# 'https://t.me/c/1361584729/2306'
# ]