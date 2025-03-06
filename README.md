# 🔥 PyTorch Lightning ⚡️: Sign MNIST Dataset Example  🤗 

[![python](https://img.shields.io/badge/Python-3.9-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
![PyTorch Lightning](https://img.shields.io/badge/pytorch-lightning-blue.svg?logo=PyTorch%20Lightning)
[![pytorch](https://img.shields.io/badge/PyTorch-1.6.0-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)


Вы когда-нибудь считали, сколько строк кода надо сваять, чтобы обучить нейронку?)

А я вот посчитала, когда поступила на курс [🤖 Машинное обучение с помощью ClearML и Pytorch Lightning ⚡](https://stepik.org/course/214389?auth=login)

80 строк кода занимает только трейн-луп :) а еще собрать надо загрузчика и тд, и тп. 

Как сейчас помню свой курсач по "Введению в глубокое обучение" в [НИУ ВШЭ](https://www.hse.ru/ma/mds/news/909801290.html) - у меня было два тренировочных цикла - мы тюнили и конволюшку, и берта. Я там еще и ошибку допустила, пока цикл делала: считала трейн лосс каждый батч, а вал-лосс - каждую эпоху, упс, но прокатил курсач на 10/10. Видимо я была очень убедительна, но ошибки надо исправлять, ибо прод и учеба - очень разные темы. А еще нельзя останавливаться на достигнутом любительском уровне, нужно переходить на качественный код 🫡

Пришлось немного попотеть, но было интересно - в результате родился компактный, легко настраиваемый скрипт (за разом забыла и про argparse, когда посоветовали click)

Из явных преимуществ для меня:
- запуск с консоли
- хуки
- модульность
- структурированность, уменьшение вероятности технических ошибок
- легко экспериментировать, изменяя конфиг
- продакшн стайл, дорого-богато (заходите в питонячий скрипт, зацените) :)

Вообщем, датасет - американский язык жестов, почитать о нём можно тут: [Sign MNIST Dataset](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)

![alphabet.png](pics/alphabet.png)

А вот и претендент на то, чтобы классно обучаться с обвязкой Pytorch Lightning :) колбаса-конволюшечка:

![pavloops_myconvnet_graph.png](pics/pavloops_myconvnet_graph.png)

Для её обучения уже всё готово, помните как в рекламе? "Просто добавь воды", но мы добавим в виртуальное окружение либы из файлика requirements.txt:
```bash
pip install -r /path/to/requirements.txt
pip install lightning[extra]
```

Если есть непреодолимое желание подкрутить гиперы - ставим свои значения в конфигах скрипта (я уверена, что в будущем их будет пылесосить клермл, но не будем сильно спойлерить), но там уже все норм.

## Запускаем :) 

### (код сделан под OS Linux, у меня винда для игрушечек, разделяю и властвую)

```bash
python pavloops_solution2.py --fast_dev_run
```

Опа, а это что за флаг такой "fast_dev_run"? Его можно не звать, но... Это то, что является хорошей практикой, запустить сеточку для проверки, всё ли живое и готовое к обучению, в копилку, спасибо :)

Кусь:

```bash
[2025-Mar-06 14:21:01] INFO: Тестовый прогон успешно пройден
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
[2025-Mar-06 14:21:02] INFO: Train file already downloaded.
[2025-Mar-06 14:21:02] INFO: Test file already downloaded.
[2025-Mar-06 14:21:04] INFO: Train is loaded to RAM.
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

  | Name      | Type             | Params | Mode
-------------------------------------------------------
0 | block1    | Sequential       | 96     | train
1 | block2    | Sequential       | 1.2 K  | train
2 | lin1      | Linear           | 78.5 K | train
3 | act1      | LeakyReLU        | 0      | train
4 | drop1     | Dropout          | 0      | train
5 | lin2      | Linear           | 2.5 K  | train
6 | criterion | CrossEntropyLoss | 0      | train
-------------------------------------------------------
82.3 K    Trainable params
0         Non-trainable params
82.3 K    Total params
0.329     Total estimated model params size (MB)
15        Modules in train mode
0         Modules in eval mode
Epoch 19: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [00:04<00:00, 47.43it/s, v_num=0, train/loss=0.237]`Trainer.fit` stopped: `max_epochs=20` reached.
Epoch 19: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [00:04<00:00, 47.26it/s, v_num=0, train/loss=0.237]
Seed set to 2025
[2025-Mar-06 14:22:32] INFO: Test is loaded to RAM.
Fact: 21, Prediction: 21
[2025-Mar-06 14:22:33] INFO: Process finished successfully.
```

## Это чистый вин 🎯🏆

Вот такого покажут (по идее, в лайтнинге же имплеменирована даже функция "зазерни всё, что можно"):

![test_picture.png](pics/test_picture.png)

## Так, подождите

В скрипте 370 строк, алло, где упрощение, где однострочник, что за развод? По опыту: 370 строк в промышленной разработке - это мало, особенно, если это качественно обвязано и хорошо масштабируемо, правда-правда.

## Что дальше?

Коллбэки, метрики не забыла из второй части второй лабы?) Забыла, иди делай :)
