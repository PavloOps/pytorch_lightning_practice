# [🤖 ClearML & PyTorch Lightning Course's Solutions (Stepik)](https://stepik.org/course/214389?auth=login)

## [Lab 1: Logging](https://github.com/PavloOps/pytorch_lightning_practice/tree/main/lab_1_logging)

<img src="lab_1_logging/log_mlops.png" width="800" />

## [Lab 2-3: Trainer & Sign MNIST Dataset Example  🤗](https://github.com/PavloOps/pytorch_lightning_practice/tree/main/lab_2_3) 

## (👑🥇 merch winner 🥳🎉)
[![python](https://img.shields.io/badge/Python-3.12-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
![PyTorch Lightning](https://img.shields.io/badge/pytorch-lightning-blue.svg?logo=PyTorch%20Lightning)
[![pytorch](https://img.shields.io/badge/PyTorch-2.5.0-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
## Motivation
Вы когда-нибудь считали, сколько строк кода надо сваять, чтобы обучить нейронку?)

А я вот посчитала, когда поступила на курс: 80 строк кода занимает только трейн-луп :) а еще собрать надо загрузчика и тд, и тп. 

Как сейчас помню свой курсач по "Введению в глубокое обучение" в [НИУ ВШЭ](https://www.hse.ru/ma/mds/news/909801290.html) - у меня было два тренировочных цикла - мы тюнили и конволюшку, и берта. Это был JN по размеру сопоставимый с "Войной и миром" Льва Толстого. Запутаться, допустить ошибку там можно было очень легко, мы выживали как могли. Считаю, что нельзя останавливаться на достигнутом любительском уровне, нужно переходить на качественный код 🫡

Из явных преимуществ для меня:
- запуск с консоли
- хуки
- модульность
- структурированность, уменьшение вероятности технических ошибок
- легко экспериментировать, изменяя конфиг
- продакшн стайл, дорого-богато (заходите в код, посмотрите)

Пришлось немного попотеть, но было интересно - в результате родился такой проект :)

## Files' tree
📁 pytorch_lightning_practice/<br>
├─📁 dataset/ (не загружалась на гит, но загрузится во время выполнения кода)<br> 
│ ├─📄 sign_mnist_test.csv<br>
│ └─📄 sign_mnist_train.csv<br>
├─📁 lightning_logs/<br>
│ └─📁 MyConvNet/<br>
│   └─📁 version_0/<br>
│     ├─📄 hparams.yaml<br>
│     └─📄 metrics.csv<br>
├─📁 pics/<br>
│ ├─📄 alphabet.png<br>
│ ├─📄 pavloops_myconvnet_graph.png<br>
│ ├─📄 test_picture.png<br>
│ └─📄 training_plot.png<br>
├─📁 saved_models/<br>
│ ├─📄 epoch=15-step=2752.ckpt<br>
│ ├─📄 epoch=17-step=3096.ckpt<br>
│ └─📄 epoch=19-step=3440.ckpt<br>
├─📁 src/<br>
│ ├─📁 tests/<br>
│ │ └─📄 custom_metrics_test.py<br>
│ ├─📄 convolutional_network.py<br>
│ ├─📄 custom_metric.py<br>
│ ├─📄 network_trainer.py<br>
│ └─📄 sign_data_module.py<br>
├─📁 terminal_logs/<br>
│ ├─📄 first_run_terminal_log.txt<br>
│ └─📄 second_run_terminal_log.txt<br>
├─📄 .gitignore<br>
├─📄 README.md<br>
├─📄 config.py<br>
├─📄 pavloops_solution2.py<br>
├─📄 requirements.txt<br>
└─📄 test_picture.png<br>

## Dataset
Датасет - американский язык жестов, почитать о нём можно тут: [Sign MNIST Dataset](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)

![alphabet.png](lab_2_3_trainer/pics/alphabet.png)


## Model
А вот и претендент на то, чтобы классно обучаться с обвязкой Pytorch Lightning :) простая сеточка - колбаса-конволюшечка:

![pavloops_myconvnet_graph.png](lab_2_3_trainer/pics/pavloops_myconvnet_graph.png)

## How to Run (Linux OS)

> На винде у меня игрушечки, я не делала под неё адаптацию :)

1. Склонируй репо
```bash
git clone https://github.com/PavloOps/pytorch_lightning_practice.git
```

2. Сделай виртуальное окружение, поставь либы из файлика requirements.txt:
```bash
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt && pip install lightning[extra]
```

3. Покрути гиперы в конфигах по желанию (см. файлик config.py)

4. Запускаем обучение

```bash
python pavloops_solution2.py --fast_dev_run
```

> Опа, а это что за флаг такой "fast_dev_run"? Его можно не звать, но... Это то, что является хорошей практикой, запустить сеточку для проверки, всё ли живое и готовое к обучению, в копилку, спасибо :)

## How does it work?

```python
    # Решение начинается с того, что с CLI будет собран флаг fast_dev_run (по умолчанию он True)

cfg = CFG()  # импортируем конфигурацию из файлика config.py

visualize_network(MyConvNet(cfg), "pavloops_myconvnet_graph")  # создаем и сохраняем граф нейросети

run_experiment(cfg, need_dev_run=fast_dev_run)  # запускаем наш эксперимент

make_one_picture_inference(config=cfg, dir_path="lab_2_3_trainer/saved_models",
                           wanted_index=12)  # делаем инференс по желаемому индексу из тестового датасета

simple_visualize_metrics(
  "lightning_logs/MyConvNet/version_0/metrics.csv")  # смотрим графики обучения (пока тут, потом будет в ClearML)
```

Хотелось бы отдельно немного рассказать о функции run_experiment:
- прежде всего, используется собственная функция PyTorch Lightning "зазерни всё" для воспроизводимости экспериментта
- потом используется специально созданный загрузчик, который наследуется от LightningDataModule. Вот тут мы используем все богатства хуков:
  - prepare_data() позволяет нам загрузить датасет по ссылкам и сохранить на жетский диск (это уже философия лайтнинга). Более того, сделала функции _calculate_sha256 и _file_is_available, чтобы проверить, а был ли уже загружен файл, чтобы не загружать его снова и был ли он загружен полноценно. Всё загрузится, распакуется, подготовится :)
  - setup() как раз займется тем, что заберет файлы с жесткого диска и отправит в RAM: предварительно подготовит их - для трейна сделает аугментацию, например, а также подготовит валидационную часть (обратите внимание на то, что указывается параметр stratify в функции train_test_split, чтобы мы сохранили распределение классов на валидации)
  - также учитывается философия стейджинга
  - в teardown будет работать сборщик мусора
  - кстати да, класс, который датасет делает надо реализовать, иначе загрузчику нечем будет датасет клепать
- сеточка сделана уже не на чистом торче, а с помощью LightningModule. Тут тоже используются хуки training_step() и тд :)
- логгироваться будут три метрики: две уже реализованные в torchmetrics (FBetaScore, AUROC) и самописный False Discovery Rate (FDR). Под него написаны тесты для самопроверки, чтобы всё считалось как задумано
- в трейнер добавлены два коллбэка (ранний останов и сохранение весов модельки), CSV-логгер (чтобы потом нарисовать графики обучения - это костылёк, потом графики будут в ClearML)
- для инференса дополнительно сделала простенькая функция, которая будет забирать последнюю лучшую модельку


## Results  🎯🏆

1. Как выглядит первый запуск с загрузкой файлов с гита и обучением (это была первая часть лабы): [first_run_terminal_log.txt](terminal_logs%2Ffirst_run_terminal_log.txt)
2. Как выглядит запуск обучения, когда файлы уже загружены, добавлены метрики, коллбэки (это была вторая часть лабы): [second_run_terminal_log.txt](terminal_logs%2Fsecond_run_terminal_log.txt)
3. Графики с процесса обучения


<img src="lab_2_3_trainer/pics/training_plot.png" width="600" />

4. А вот и картиночка с инференса:

<img src="lab_2_3_trainer/pics/test_picture.png" width="200" />

## Final Notes

В проекте дофига каких-то файлов, ООП, конфиги-папки, алло, где упрощение, где однострочник, что за развод? По опыту: в промышленной разработке - это очень компактно, особенно, если это качественно обвязано и хорошо масштабируемо, правда-правда.

Мне понравилось, харды прокачались)

🌟🌟🌟Если вам понравилось моё решение, и оно было вам полезно - сделайте тык в звездочку, вам не сложно, а мне будет приятно ❤️❤️❤️

## [Lab 4: GAN & PyTorch Lighting  🤗](lab_4_gan/pavloops_solution4.py) 

Клонируй с гита:

```bash
git clone https://github.com/PavloOps/pytorch_lightning_practice.git && cd lab_4_gan
```

Запуск с терминала (в этот раз делала под Windows для разнообразия):

```bash
py pavloops_solution4.py -F -E 12 -D 2
```

Аналогичная команда с полными именами параметров:

```bash
py pavloops_solution4.py --fast_dev_run --epoch 12 --debug_samples_epoch 2
```

Результат в юайке ClearML:

https://app.clear.ml/projects/a7f426d19f9f493980c13330e8aa07b6/experiments/afc67674c8b34197b32644a0ff02324a/output/execution

Мои логи:

```
$ py pavloops_solution4.py -F -E 12 -D 2
Enter CLEARML_WEB_HOST:
Enter CLEARML_API_HOST:
Enter CLEARML_FILES_HOST:
Enter CLEARML_API_ACCESS_KEY:
Enter CLEARML_API_SECRET_KEY:
All environment variables are set.
ClearML Task: created new task id=afc67674c8b34197b32644a0ff02324a
ClearML results page: https://app.clear.ml/projects/a7f426d19f9f493980c13330e8aa07b6/experiments/afc67674c8b34197b32644a0ff02324a/output/log
torch version:  2.2.2+cu121
cuda version:  12.1
gpu is available:  True
device name:  NVIDIA GeForce RTX 3080 Ti Laptop GPU
Seed set to 2025
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
Running in `fast_dev_run` mode: will run the requested loop using 1 batch(es). Logging and checkpointing is suppressed.
[2025-Nov-03 14:41:53] INFO: Processed dataset not found — downloading raw MNIST...
[2025-Nov-03 14:42:23] INFO: Processed MNIST saved to ../data\dataset.pt
[2025-Nov-03 14:42:29] INFO: Train and validation datasets are loaded in RAM.
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

  | Name             | Type          | Params | Mode
-----------------------------------------------------------
0 | generator        | Generator     | 1.8 M  | train
1 | discriminator    | Discriminator | 138 K  | train
2 | criterion        | BCELoss       | 0      | train
3 | resize_transform | Resize        | 0      | train
-----------------------------------------------------------
1.9 M     Trainable params
0         Non-trainable params
1.9 M     Total params
7.729     Total estimated model params size (MB)
23        Modules in train mode
0         Modules in eval mode
`Trainer.fit` stopped: `max_steps=1` reached.
Debug run has been finished.
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
[2025-Nov-03 14:43:08] INFO: Processed dataset already exists — skipping download.
2025-11-03 14:43:14,263 - clearml.model - WARNING - Connecting multiple input models with the same name: `dataset`. This might result in the wrong model being used when executing remotely
[2025-Nov-03 14:43:19] INFO: Train and validation datasets are loaded in RAM.
C:\Users\olgal\PycharmProjects\pytorch_lightning_practice\.venv\Lib\site-packages\lightning\pytorch\callbacks\model_checkpoint.py:654: UserWarning:

Checkpoint directory C:\Users\olgal\PycharmProjects\pytorch_lightning_practice\lab_4_gan exists and is not empty.

LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

  | Name             | Type          | Params | Mode
-----------------------------------------------------------
0 | generator        | Generator     | 1.8 M  | train
1 | discriminator    | Discriminator | 138 K  | train
2 | criterion        | BCELoss       | 0      | train
3 | resize_transform | Resize        | 0      | train
-----------------------------------------------------------
1.9 M     Trainable params
0         Non-trainable params
1.9 M     Total params
7.729     Total estimated model params size (MB)
23        Modules in train mode
0         Modules in eval mode
Epoch 1: 100%|███████████████████████████████████████████████████████████████████████████████████| 750/750 [00:08<00:00, 89.48it/s, v_num=0, train/loss_discriminator=0.457, train/loss_generator=1.120]Metric val/loss_generator improved. New best score: 14.148████████████████████████████████████████████████████████████████████████████████████████████████████████████| 187/187 [00:00<00:00, 222.88it/s] 
Epoch 11: 100%|█████████████████████████| 750/750 [00:13<00:00, 57.55it/s, v_num=0, train/loss_discriminator=0.169, train/loss_generator=2.520, val/loss_discriminator=0.0986, val/loss_generator=21.80]Monitored metric val/loss_generator did not improve in the last 5 records. Best score: 14.148. Signaling Trainer to stop.██████████████████████████████████████████████| 187/187 [00:01<00:00, 94.11it/s]
Epoch 11: 100%|█████████████████████████| 750/750 [00:15<00:00, 49.78it/s, v_num=0, train/loss_discriminator=0.169, train/loss_generator=2.520, val/loss_discriminator=0.0951, val/loss_generator=22.50]`Trainer.fit` stopped: `max_epochs=12` reached.
Epoch 11: 100%|█████████████████████████| 750/750 [00:15<00:00, 49.78it/s, v_num=0, train/loss_discriminator=0.169, train/loss_generator=2.520, val/loss_discriminator=0.0951, val/loss_generator=22.50] 

olgal@DESKTOP-CH6JPSI MINGW64 ~/PycharmProjects/pytorch_lightning_practice/lab_4_gan (main)
```

Прогресс генератора можно посмотреть в ClearML: debug samples -> Generated Samples, --all, но можно и тут :)

![img.png](lab_4_gan/generated_samples/img.png)<br>
![img_1.png](lab_4_gan/generated_samples/img_1.png)<br>
![img_2.png](lab_4_gan/generated_samples/img_2.png)<br>
![img_3.png](lab_4_gan/generated_samples/img_3.png)<br>
![img_4.png](lab_4_gan/generated_samples/img_4.png)<br>
![img_5.png](lab_4_gan/generated_samples/img_5.png)<br>
![img_6.png](lab_4_gan/generated_samples/img_6.png)
