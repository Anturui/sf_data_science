<div align="center"> <h1 align="center"> PROJECT-4. Перенос стиля на мобильном устройстве. </h1> </div>
 
<div align="center"> <h3 align="center"> by Aleksey Kolychev </h3> </div>

<div align="center"> <h3 align="center">Задача</h3> </div>

<div align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Обучить модель, оптимизировать её и импортировать веса в приложение. </div>

<div align="center"> <h3 align="center"> ЭТАПЫ ВЫПОЛНЕНИЯ ПРОЕКТА:</h3> </div>

1. ИЗУЧИТЬ ИНСТРУКЦИЮ, ПОДГОТОВИТЬ ДАННЫЕ И ЗАПУСТИТЬ ОБУЧЕНИЕ
2.	ПЕРЕВЕСТИ МОДЕЛЬ В TF LITE 
3.	ВЫБРАТЬ МОБИЛЬНУЮ ОПЕРАЦИОННУЮ СИСТЕМУ
4.	СОЗДАТЬ ПРИЛОЖЕНИЕ
5.	ОФОРМИТЬ И ОТПРАВИТЬ РЕШЕНИЕ НА ПРОВЕРКУ

<div align="center"> <h2 align="center"> 1.	ИЗУЧИТЬ ИНСТРУКЦИЮ, ПОДГОТОВИТЬ ДАННЫЕ И ЗАПУСТИТЬ ОБУЧЕНИЕ </h2> </div>

<div align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Обучение и инфиренс выполняются в файле "Обучение и инфиренс моделей style transfer.ipynb"</div>


<div align="center"> <h2 align="center"> 2. ПЕРЕВЕСТИ МОДЕЛЬ В TF LITE  </h2> </div>

<div align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Перевод модели в формат TF Lite производится в файле колаб "Преобразование модели в формат TF Lite project_4.ipynb" </div>

<div align="center"> <h2 align="center"> 3.	ВЫБРАТЬ МОБИЛЬНУЮ ОПЕРАЦИОННУЮ СИСТЕМУ  </h2> </div>

<div align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Выбрана операционная система Android</div>


<div align="center"> <h2 align="center"> 4. СОЗДАТЬ ПРИЛОЖЕНИЕ  </h2> </div>

<div align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Приложение было создано в Android Studio</div>


<div align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Приложение работает следующим образом. Производится фотографирование, выбирается стиль и применяется, нажатием кнопки "RUN" </div>

<div align="center"> <h2 align="center"> Скриншоты и видео </h2> </div>

![photo_2023-05-09_19-05-06](https://github.com/Anturui/sf_data_science/assets/106611550/64fa6816-d11a-48b4-a394-6dd11686f16c)

![photo_2023-05-09_19-05-03](https://github.com/Anturui/sf_data_science/assets/106611550/a47d3ccb-425b-4754-99dd-cdc0d5a53f42)

![photo_2023-05-09_19-05-01](https://github.com/Anturui/sf_data_science/assets/106611550/12fc617a-84ed-4b60-83f9-9a0a57f7cd15)



<div align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Была предпринята попытка кластеризирвоать объекты датасета, однако, ни к чему это не привело (Файл Clusters). DBScan указал на необходимость создании 10 тыс. кластеров для наибольшего коэффициента Силуэта. </div>


<div align="center"> <h2 align="center"> Описание датасетов </h2> </div>

***global_data_prep*** - исходный датасет с данными по постам, собранными при помощи библиотеки telethone
#### [Исходный датасет](https://drive.google.com/file/d/1SUDo5XR1wiGO2JvqsaIr8ZixXQW9QtCx/view?usp=share_link)

***grand_data_zero_3*** - обработанный датасет с новыми признаками, полученными из исходного датасета
#### [Обработанный датасет](https://drive.google.com/file/d/1bOAwLDo-ZP2XVJTnVWz6VphYcC8sdpu5/view?usp=sharing)

<div align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Датасеты добавляете в директорию с проектом и работаете далее, если есть что дополнить=)</div>

<div align="center"> <h2 align="center"> Описание файлов </h2> </div>

***csr.pkl*** - модель, предсказывающее репостовость (отношение репостов к просмотрам) 

***table.csv*** - файл, в который идет запись новых признаков из исходного датасета

***test.txt*** - файл, в который записывается название файла изображения, для его цветового анализа

***msg.csv*** - файл, который записывается ботом aiogram с данными по сообщению (текст, медиа, дата и др.)

***Script_to_predict_to_function.py*** - скрипт, который преобразует входящее (для анализа) сообщение в формат, необходимый для прогноза, является частью бота

***wrds_cnt_df.csv*** - список всех слов, которые были в текстах 39000+ постов



<div align="center"> <h2 align="center"> ВЫВОДЫ </h2> </div>

1.	Была проделана большая работа по анализу постов в ТГ и созданию модели ML для предсказания репостовости поста.
2.	Производился анализ текста, изображений, эмодзи и оформления. 
3.	Получилась точность MAE – 0.02 и MAPE – 36 % на тесте, что для таргета, отличающегося на несколько порядков хороший показатель
4.	Фундаментально существуют характеристики поста, которые влияют на отношение к нему человека, как биологического вида, например, золотое сечение. 
5.	Получен большой опыт в DS в части сбора, обработки датафреймов, подготовки к ML, обучение и продакшн. 
6.	Я собой доволен.


PS: Работу бота можно попробовать. Обратитесь ко мне, пожалуйста, я запущу его, чтобы опробовать 
