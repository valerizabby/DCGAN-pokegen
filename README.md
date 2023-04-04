# DCGAN-pokegen

Решение задачи генерации новых покемонов с помощью модели из класса архитектур DCGAN. 

## Data preprocessing
[Датасет с Kaggle](https://www.kaggle.com/datasets/hlrhegemony/pokemon-image-dataset).
Имеем 2500 чистых картинок покемонов. Это не очень много, поэтому саугментируем данные и понадеемся, что это улучшит перфоманс модели. 
- Поиграемся с цветами с помощью `color_jitter`
- Сделеаем `elastic_transform`

Так же обрежем все фото до $64 \times 64$ пикселей и отнормализуем в отрезок $[-1, 1]$ и так же [понадеемся](https://datascience.stackexchange.com/questions/54296/should-input-images-be-normalized-to-1-to-1-or-0-to-1), что это улучшит перфоманс по сравнению с $[0,1]$.

Итак, мы увеличили датасет в 3 раза.

## Model

Класс архитектур DCGAN основан на наборе принципов, таких как: 
- Замена pooling слоев на пошаговые свертки
- Использование пакетной нормализации в генераторе и дискриминаторе
- `ReLU` на всех слоях генератора, кроме выходного (там  `tanh`)
- `LeakyReLU` на всех слоях дискриминатора 

В этой работе используются следующие архитектуры: 

Diskriminator            |  Generator
:-------------------------------------:|:-------------------------------------:
![disk](https://github.com/valerizabby/DCGAN-pokegen/blob/main/pictures/discriminator-model100.pth.png)  |  ![gen](https://github.com/valerizabby/DCGAN-pokegen/blob/main/pictures/generator-model100.pth.png)

## Training

В обучении используется `BCE Loss` и `Adam Optimizer` с параметрами `betas=(0.5, 0.9)` (причем GAN-ы из-за специфики обучения крайне чувствительны к гиперпараметрам). 


