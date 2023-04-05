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

В обучении используется `BCE Loss` и `Adam Optimizer` с параметрами `betas = (0.5, 0.9)` (GAN-ы из-за специфики обучения крайне чувствительны к гиперпараметрам). Обучим модель два раза на $50$ и $100$ эпох соответственно. Гифки с процессом обучения: 


50 epochs            |  100 epochs
:-------------------------------------:|:-------------------------------------:
![50](https://github.com/valerizabby/DCGAN-pokegen/blob/main/pictures/gif50.gif)  |  ![100](https://github.com/valerizabby/DCGAN-pokegen/blob/main/pictures/gif100.gif)
`learning rate = 0.00275` | `learning rate = 0.0028`

## Tensor Board
Процесс обучения (лоссы и скоры) сохраняются в Tensor Board, результаты можно посмотреть по ссылкам для [50 эпох](https://tensorboard.dev/experiment/ltajKpFnRnG9xoclg8v5zw/#scalars) и [100 эпох](https://tensorboard.dev/experiment/SBqmdE1BT2uRkD8T3mrFKw/#scalars).

## Pretrained models
Обученные модели можно скачать по [ссылке](https://drive.google.com/drive/u/0/folders/1N3PIDPy60qruqCeyqI-4teDcdi533rxf). В конце ноутбука есть функция `download_file_from_google_drive(id, destination)`, которая импортирует предобученные модели.

## Summary 
DCGAN довольно старый класс архитектур (2016 г), в нем используются устаревшие идеи, поэтому получить крутой перфоманс не получится. Среди таких идей, например, использование пошаговых сверток и pooling слоев. Так как в этой задаче используются фотографии в низком разрешении ($64 \times 64$ пикселей) эти слои приводят к потере "мелких" признаков и ухудшению признаковых представлений картинок ([источник](https://arxiv.org/abs/2208.03641#)).

## References
- DCGAN [статья](https://arxiv.org/pdf/1511.06434.pdf), [туториал](https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/5f81194dd43910d586578638f83205a3/dcgan_faces_tutorial.ipynb#scrollTo=PEDpuNs6NZ4x)
- [TensorBoard](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html)
- [Tips to imporve GAN's performance](https://machinelearningmastery.com/how-to-train-stable-generative-adversarial-networks/)
- [Useful functions from PokeDex](https://jovian.ml/jkleiber8/course-project-pokegan/v/3?utm_source=embed)
- [Общая информация про GAN](https://developers.google.com/machine-learning/gan/loss?hl=ru)
- [Netron](https://netron.app/)
- [Функции для сохранения модели с Google Drive](https://forums.fast.ai/t/unpicklingerror-invalid-load-key/68035)
