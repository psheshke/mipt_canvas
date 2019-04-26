
# coding: utf-8

# <p style="align: center;"><img align=center src="https://s8.hostingkartinok.com/uploads/images/2018/08/308b49fcfbc619d629fe4604bceb67ac.jpg" style="height:90px;" width=500/></p>
# 
# <h3 style="text-align: center;"><b>Физтех-Школа Прикладной математики и информатики (ФПМИ) МФТИ</b></h3>

# ---

# В этом ноутбке мы научимся писать свои свёрточные нейросети на фреймворке PyTorch, и протестируем их работу на датасетах MNIST и CIFAR10. 
# 
# **ВНИМАНИЕ:** Рассматривается ***задача классификации изображений***.
# 
# (Подразумевается, что читатель уже знаком с многослойной нейроннной сетью).  
# 
# ***Свёрточная нейросеть (Convolutional Neural Network, CNN)*** - это многослойная нейросеть, имеющая в своей архитектуре помимо *полносвязных слоёв* (а иногда их может и не быть) ещё и **свёрточные слои (Conv Layers)** и **pooling-слои (Pool Layers)**.  
# 
# Собственно, название такое эти сети получили потому, что в основе их работы лежит операция **свёртки**. 
# 
# 
# Сразу же стоит сказать, что свёрточные нейросети **были придуманы прежде всего для задач, связанных с картинками**, следовательно, на вход они тоже "ожидают" картинку.
# 
# Расмотрим их устройство более подробно:
# 
# * Вот так выглядит неглубокая свёрточная нейросеть, имеющая такую архитектуру:  
# `Input -> Conv 5x5 -> Pool 2x2 -> Conv 5x5 -> Pool 2x2 -> FC -> Output`
# 
# <img src="https://camo.githubusercontent.com/269e3903f62eb2c4d13ac4c9ab979510010f8968/68747470733a2f2f7261772e6769746875622e636f6d2f746176677265656e2f6c616e647573655f636c617373696669636174696f6e2f6d61737465722f66696c652f636e6e2e706e673f7261773d74727565" width=800, height=600>
# 
# Свёрточные нейросети (обыкновенные, есть и намного более продвинутые) почти всегда строятся по следующему правилу:  
# 
# `INPUT -> [[CONV -> RELU]*N -> POOL?]*M -> [FC -> RELU]*K -> FC`  
# 
# то есть:  
# 
# 1). ***Входной слой*** (batch картинок `HxWxC`)  
# 
# 2). $M$ блоков (M $\ge$ 0) из свёрток и pooling-ов, причём именно в том порядке, как в формуле выше. Все эти $M$ блоков вместе называют ***feature extractor*** свёрточной нейросети, потому что эта часть сети отвечает непосредственно за формирование новых, более сложных признаков, поверх тех, которые подаются (то есть, по аналогии с MLP, мы опять же переходим к новому признаковому пространству, однако здесь оно строится сложнее, чтем в обычных многослойных сетях, поскольку используется операция свёртки)  
# 
# 3). $K$ штук FullyConnected-слоёв (с активациями). Эту часть из $K$ FC-слоёв называют ***classificator***, поскольку эти слои отвечают непосредственно за предсказание нужно класса (сейчас рассматривается задача классификации изображений).

# 
# <h3 style="text-align: center;"><b>Свёрточная нейросеть на PyTorch</b></h3>
# 
# Ешё раз напомним про основные компоненты нейросети:
# 
# - непосредственно, сама **архитектура** нейросети (сюда входят типы функций активации у каждого нейрона);
# - начальная **инициализация** весов каждого слоя;
# - метод **оптимизации** нейросети (сюда ещё входит метод изменения `learning_rate`);
# - размер **батчей** (`batch_size`);
# - количетсво итераций обучения (`num_epochs`);
# - **функция потерь** (`loss`);  
# - тип **регуляризации** нейросети (для каждого слоя можно свой);  
# 
# То, что связано с ***данными и задачей***:  
# - само **качество** выборки (непротиворечивость, чистота, корректность постановки задачи);  
# - **размер** выборки;  
# 
# Так как мы сейчас рассматриваем **архитектуру CNN**, то, помимо этих компонент, в свёрточной нейросети можно настроить следующие вещи:  
# 
# - (в каждом ConvLayer) **размер фильтров (окна свёртки)** (`kernel_size`)
# - (в каждом ConvLayer) **количество фильтров** (`out_channels`)  
# - (в каждом ConvLayer) размер **шага окна свёртки (stride)** (`stride`)  
# - (в каждом ConvLayer) **тип padding'а** (`padding`)  
# 
# 
# - (в каждом PoolLayer) **размер окна pooling'a** (`kernel_size`)  
# - (в каждом PoolLayer) **шаг окна pooling'а** (`stride`)  
# - (в каждом PoolLayer) **тип pooling'а** (`pool_type`)  
# - (в каждом PoolLayer) **тип padding'а** (`padding`)

# Какими их берут обычно -- будет показано в примере ниже. По крайней мере, можете стартовать с этих настроек, чтобы понять, какое качество "из коробки" будет у простой модели.

# Посмотрим, как работает CNN на MNIST'е и на CIFAR'е:

# <img src="http://present5.com/presentation/20143288_415358496/image-8.jpg" width=500, height=400>

# **MNIST:** это набор из 70k картинок рукописных цифр от 0 до 9, написанных людьми, 60k из которых являются тренировочной выборкой (`train` dataset)), и ещё 10k выделены для тестирования модели (`test` dataset).

# In[1]:


import torch
import torchvision
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt  # для отрисовки картиночек
get_ipython().run_line_magic('matplotlib', 'inline')


# Скачаем и загрузим в `loader`'ы:

# **Обратите внимание на аргумент `batch_size`:** именно он будет отвечать за размер батча, который будет подаваться при оптимизации нейросети

# In[2]:


transform = transforms.Compose(
    [transforms.ToTensor()])

trainset = torchvision.datasets.MNIST(root='./data', train=True, 
                                      download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                     download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = tuple(str(i) for i in range(10))


# Сами данные лежат в полях `trainloader.dataset.train_data` и `testloader.dataset.test_data`:

# In[3]:


trainloader.dataset.train_data.shape


# In[4]:


testloader.dataset.test_data.shape


# Выведем первую картинку:

# In[5]:


trainloader.dataset.train_data[0]


# Посмотрим, как она выглядит:

# In[6]:


# преобразовать тензор в np.array
numpy_img = trainloader.dataset.train_data[0].numpy()


# In[7]:


numpy_img.shape


# In[8]:


plt.imshow(numpy_img);


# In[9]:


plt.imshow(numpy_img, cmap='gray');


# Отрисовка заданной цифры:

# In[13]:


# случайный индекс от 0 до размера тренировочной выборки
i = np.random.randint(low=0, high=60000)

plt.imshow(trainloader.dataset.train_data[i].numpy(), cmap='gray');


# Как итерироваться по данным с помощью `loader'`а? Очень просто:

# In[14]:


for data in trainloader:
    print(data)
    break


# То есть мы имеем дело с кусочками данных размера batch_size (в данном случае = 4), причём в каждом батче есть как объекты, так и ответы на них (то есть и $X$, и $y$).

# Теперь вернёмся к тому, что в PyTorch есть две "парадигмы" построения нейросетей -- `Functional` и `Seuquential`. Со второй мы уже хорошенько разобрались в предыдущих ноутбуках по нейросетям, теперь мы испольузем именно `Functional` парадигму, потому что при построении свёрточных сетей это намного удобнее:

# In[2]:


import torch.nn as nn
import torch.nn.functional as F  # Functional


# In[3]:


# ЗАМЕТЬТЕ: КЛАСС НАСЛЕДУЕТСЯ ОТ nn.Module
class SimpleConvNet(nn.Module):
    def __init__(self):
        # вызов конструктора предка
        super(SimpleConvNet, self).__init__()
        # необходмо заранее знать, сколько каналов у картинки (сейчас = 1),
        # которую будем подавать в сеть, больше ничего
        # про входящие картинки знать не нужно
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(4 * 4 * 16, 120)  # !!!
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.shape)
        x = x.view(-1, 4 * 4 * 16)  # !!!
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# **Важное примечание:** Вы можете заметить, что в строчках с `#!!!` есть не очень понятный сходу 4 `*` 4 `*` 16. Это -- размерность картинки перед FC-слоями (H x W x C), тут её приходиться высчитывать вручную (в Keras, например, `.Flatten()` всё делает за Вас). Однако есть один *лайфхак* -- просто сделайте в `forward()` `print(x.shape)` (закомментированная строка). Вы увидите размер `(batch_size, C, H, W)` -- нужно перемножить все, кроме первого (batch_size), это и будет первая размерность `Linear()`, и именно в C * H * W нужно "развернуть" x перед подачей в `Linear()`.  
# 
# То есть нужно будет запустить цикл с обучением первый раз с `print()` и сделать после него `break`, посчитать размер, вписать его в нужные места и стереть `print()` и `break`.

# Код обучения слоя:

# In[4]:


from tqdm import tqdm_notebook


# In[18]:


# объявляем сеть
net = SimpleConvNet()

# выбираем функцию потерь
loss_fn = torch.nn.CrossEntropyLoss()

# выбираем алгоритм оптимизации и learning_rate
learning_rate = 1e-4
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# итерируемся
for epoch in tqdm_notebook(range(3)):

    running_loss = 0.0
    for i, batch in enumerate(tqdm_notebook(trainloader)):
        # так получаем текущий батч
        X_batch, y_batch = batch
        
        # обнуляем веса
        optimizer.zero_grad()

        # forward + backward + optimize
        y_pred = net(X_batch)
        loss = loss_fn(y_pred, y_batch)
        loss.backward()
        optimizer.step()

        # выведем текущий loss
        running_loss += loss.item()
        # выведем качество каждые 2000 батчей
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Обучение закончено')


# Протестируем на всём тестовом датасете, используя метрику accuracy_score:

# In[19]:


class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

with torch.no_grad():
    for data in testloader:
        images, labels = data
        y_pred = net(images)
        _, predicted = torch.max(y_pred, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))


# Два свёрточных слоя побили многослойную нейросеть. Не магия ли?

# ---

# ### Задача 1

# Протестируйте эту нейросеть на отдельных картинках из тестового датасета: напишите функцию, которая принимает индекс картинки в тестовом датасете, отрисовывает её, потом запускает на ней модель (нейросеть) и выводит результат предсказания.

# In[169]:


# Ваш код здесь

i = np.random.randint(low=0, high=10000)

def show_pred(i, testloader):

    image, label = testloader.dataset[i]

    plt.imshow(image.view(28,28).numpy(), cmap='gray');

    y_pred = net(image.view(1,1,28,28))
    _, predicted = torch.max(y_pred, 1)

    print('predicted - %s, fact - %s'%(predicted.numpy()[0], label))


# In[182]:


show_pred(7, testloader)


# ---

# <h3 style="text-align: center;"><b>CIFAR10</b></h3>

# <img src="https://raw.githubusercontent.com/soumith/ex/gh-pages/assets/cifar10.png" width=500, height=400>

# **CIFAR10:** это набор из 60k картинок 32х32х3, 50k которых составляют обучающую выборку, и оставшиеся 10k - тестовую. Классов в этом датасете 10: `'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'`.
# 
# Скачаем и загрузим в `loader`'ы:
# 
# **Обратите внимание на аргумент `batch_size`:** именн он будет отвечать за размер батча, который будет подаваться при оптимизации нейросети

# In[8]:


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# In[9]:


# случайный индекс от 0 до размера тренировочной выборки
i = np.random.randint(low=0, high=50000)

plt.imshow(trainloader.dataset.train_data[i], cmap='gray');


# То есть мы имеем дело с кусочками данных размера batch_size (в данном случае = 4), причём в каждом батче есть как объекты, так и ответы на них (то есть и $X$, и $y$).
# 
# Данные готовы, мы даже на них посмотрели. **Однако учтите** - при подаче в нейросеть мы будем разворачивать картинку 32х32х3 в строку 1х(32`*`32`*`3) = 1х3072, то есть мы считаем пиксели (значения интенсивности в пикселях) за признаки нашего объекта (картинки).  
# 
# К делу:

# ### Задача 2

# Напишите свою свёрточную нейросеть для предсказания на CIFAR10.

# In[ ]:


# ЗАМЕТЬТЕ: КЛАСС НАСЛЕДУЕТСЯ ОТ nn.Module
class MyConvNet(nn.Module):
    def __init__(self):
        # вызов конструктора предка
        super(MyConvNet, self).__init__()
        # Ваш код здесь
        pass

    def forward(self, x):
        # Ваш код здесь
        pass


# Обучим:

# In[ ]:


from tqdm import tqdm_notebook


# In[ ]:


# пример взят из официального туториала: 
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

net = MyConvNet()

loss_fn = torch.nn.CrossEntropyLoss()

learning_rate = 1e-4
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# итерируемся
for epoch in tqdm_notebook(range(3)):

    running_loss = 0.0
    for i, batch in enumerate(tqdm_notebook(trainloader)):
        # так получаем текущий батч
        X_batch, y_batch = batch
        
        # РАЗВОРАЧИВАЕМ КАРТИНКУ В СТРОКУ
        X_batch = X_batch.view(4, -1)

        # обнуляем веса
        optimizer.zero_grad()

        # forward + backward + optimize
        y_pred = net(X_batch)
        loss = loss_fn(y_pred, y_batch)
        loss.backward()
        optimizer.step()

        # выведем текущий loss
        running_loss += loss.item()
        # выводем качество каждые 2000 батчей
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Обучение закончено')


# Посмотрим на accuracy на тестовом датасете:

# In[ ]:


class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

with torch.no_grad():
    for data in testloader:
        images, labels = data
        y_pred = net(images.view(4, -1))
        _, predicted = torch.max(y_pred, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))


# Как думаете, этого достаточно?

# ### Задача 3  
# 
# Улучшите свёрточную нейросеть: поэкспериментируйте с архитектурой (количество слоёв, порядок слоёв), с гиперпараметрами слоёв (размеры kernel_size, размеры pooling'а, количество kernel'ов в свёрточном слое) и с гиперпараметрами, указанными в "Компоненты нейросети" (см. памятку выше).

# In[ ]:


# Ваш код здесь


# (Ожидаемый результат -- скорее всего, сходу Вам не удастся выжать из Вашей сетки больше, чем ~70% accuracy (в среднем по всем классам). Если это что-то в этом районе - Вы хорошо постарались).

# <h3 style="text-align: center;"><b>Полезные ссылки</b></h3>

# 1). *Примеры написания нейросетей на PyTorch (офийиальные туториалы) (на английском): https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#examples  
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html*

# 2). ***Один из самых подробных и полных курсов по deep learning на данный момент - это курс Стэнфордского Университета (он вообще сейчас один из лидеров в области ИИ, его выпускники работают в Google, Facebook, Amazon, Microsoft, в стартапах в Кремниевой долине):  http://cs231n.github.io/***  

# 3). Практически исчерпывающая информация по основам свёрточных нейросетей (из cs231n) (на английском):  
# 
# http://cs231n.github.io/convolutional-networks/  
# http://cs231n.github.io/understanding-cnn/  
# http://cs231n.github.io/transfer-learning/

# 4). Видео о Computer Vision от Andrej Karpathy: https://www.youtube.com/watch?v=u6aEYuemt0M
