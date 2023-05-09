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

<div align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Пример инфиренса при различных значения степени переноса стиля</div>

![Образец инфиренса при разных уровнях 1](https://github.com/Anturui/sf_data_science/assets/106611550/8c0a3565-b56a-4d4c-8bd3-643469b9fb2a)

![Образец инфиренса при разных уровнях 2](https://github.com/Anturui/sf_data_science/assets/106611550/fa5857cb-c531-47f5-ac31-fd7e98c88649)


<div align="center"> <h2 align="center"> 2. ПЕРЕВЕСТИ МОДЕЛЬ В TF LITE  </h2> </div>

<div align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Перевод модели в формат TF Lite производится в файле колаб "Преобразование модели в формат TF Lite project_4.ipynb" </div>

<div align="center"> <h2 align="center"> 3.	ВЫБРАТЬ МОБИЛЬНУЮ ОПЕРАЦИОННУЮ СИСТЕМУ  </h2> </div>

<div align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Выбрана операционная система Android</div>


<div align="center"> <h2 align="center"> 4. СОЗДАТЬ ПРИЛОЖЕНИЕ  </h2> </div>

<div align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Приложение было создано в Android Studio</div>

#### [Приложение в формате apk, можно скачать здесь](https://drive.google.com/file/d/1SUDo5XR1wiGO2JvqsaIr8ZixXQW9QtCx/view?usp=share_link)

<div align="justify"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Приложение работает следующим образом. Производится фотографирование, выбирается стиль и применяется, нажатием кнопки "RUN" </div>

<div align="center"> <h2 align="center"> Скриншоты и видео </h2> </div>

![photo_2023-05-09_19-05-06](https://github.com/Anturui/sf_data_science/assets/106611550/64fa6816-d11a-48b4-a394-6dd11686f16c)

![photo_2023-05-09_19-05-03](https://github.com/Anturui/sf_data_science/assets/106611550/a47d3ccb-425b-4754-99dd-cdc0d5a53f42)

![photo_2023-05-09_19-05-01](https://github.com/Anturui/sf_data_science/assets/106611550/12fc617a-84ed-4b60-83f9-9a0a57f7cd15)


https://github.com/Anturui/sf_data_science/assets/106611550/7b3bac69-a1a4-4895-8534-fc2ca1ff1d17


<div align="center"> <h2 align="center"> ВЫВОДЫ </h2> </div>

1.	Была обучена модель передачи стиля, произведены тестовые инфиренсы.
2.	Модель переведена в формат TF Lite. 
3.	Создано приложение с быстрым инфиренсом. Работа приложения продемонстрирована видео и скриншотами
4.	Все требования проекта были выполнены полностью
