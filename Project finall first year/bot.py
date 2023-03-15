from aiogram import Bot, types

from aiogram.dispatcher import Dispatcher 

from aiogram.utils import executor 

import pandas as pd 

import numpy as np


import datetime

from Script_to_predict_to_function import predict_of_forwards_views_rsh


# from telethon import TelegramClient, sync

# api_id = 25673476

# api_hash = '5b9d46fbf4ea78db18fd54a588a31fd7'

# # api_id = 26313760
# # api_hash = '8b8f5a9619f9bb388b1915055f3e5379'

# client = TelegramClient('session_name', api_id, api_hash)

# client.start()

# from telethon.tl.functions.messages import (GetHistoryRequest)

# from telethon.tl.types import (PeerChannel)



# 'https://t.me/MesExpertBot'


TOKEN = "ваш токен от бота здесь"

bot = Bot(token='5821291460:AAFgBRXYlV4Hx46Q7tnkSQCtLP73jcFPi74')

dp = Dispatcher(bot)

@dp.message_handler(commands=['start', 'help'])

async def send_welcome(msg: types.Message):
    
    await msg.reply_to_message(f'Я бот. Приятно познакомиться, {msg.from_user.first_name}')
    
    
@dp.message_handler(content_types=['text'])

async def get_text_messages(msg: types.Message):
    
    if msg.text.lower() == 'привет':
       
       await msg.answer('Здравствуйте!')
       
    if msg.text.lower() != 'привет':
       
       await msg.answer('Здравствуйте!')
    
    await msg.answer("""Целью данного проекта является проведение оценки постов в категории "Позновательное".
                     Для этого было проанализировано порядка 40000 постов, рассмотрено 1345 признаков каждого из них.
                     На данной базе разработана модель предсказания отношения количества репостов к просмотрам каждого поста.
                     Как это работает: вы скидываете боту пост и он дает оценку Вашего поста, выставляя очки.
                     Чем выше очков у постов Вашего канала, тем Выше ER и ER24 можно будет получить""")
    
    await msg.answer("""Если данный бот принес Вам определенную пользу, то поддержите, пожалуйста, проект
                     Для этого напишите, пожалуйста, @rurualexei - администратор бота """)
       

@dp.message_handler(content_types=['voice', 'video', 'text', 'document', 'animation'])

async def downloader(message:types.Message):
    
    pd.DataFrame(message).to_csv('msg.csv')
    
    if message.voice:
      
        file_id = message.voice.file_id
        
        file = await bot.get_file(file_id)
        
        file_path = file.file_path
        
        await bot.download_file(file_path, "pictest/123.ogg")
        
        await message.reply('это голосовое сообщение')
    
    elif message.video:
        
        pd.DataFrame(message.video).to_csv('video.csv', index=False)
      
        file_id = message.video.thumb.file_id
        
        file = await bot.get_file(file_id)
        
        file_path = file.file_path
        
        print(file_path)
        
        save_date = 123
        
        string = f"123_{save_date}_1.jpeg"
        
        await bot.download_file(file_path, f"pictest/123_{save_date}_1.jpeg")
        
        with open('test.txt', 'w') as output:
    
            output.write(string)
        
        predict = predict_of_forwards_views_rsh()
        
        await message.reply(f'это видео и и его прогнозное отношение репостов к просмотрам равно {predict} или {np.round(predict * 100000)} очков')
    
    elif message.text:
        
        print(message.text)
        
        await message.answer('Лосони тунца и не пизди!')
    
    elif message.document:
    
        file_id = message.document.thumb.file_id
                
        file = await bot.get_file(file_id)
        
        file_path = file.file_path
        
        save_date = 'gif'
        
        string = f"123_{save_date}_1.jpeg"
        
        await bot.download_file(file_path, f"pictest/123_{save_date}_1.jpeg")
        
        predict = predict_of_forwards_views_rsh()
        
        await message.reply(f'это gif и его прогнозное отношение репостов к просмотрам равно {predict} или {np.round(predict * 100000)} очков')
        

           

if __name__ == '__main__':
    
   executor.start_polling(dp)
                               