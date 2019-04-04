
# coding: utf-8

# <img src="https://s8.hostingkartinok.com/uploads/images/2018/08/308b49fcfbc619d629fe4604bceb67ac.jpg" width=500,>
# <h3 style="text-align: center;"><b>Физтех-Школа Прикладной математики и информатики (ФПМИ) МФТИ</b></h3>

# ---

# <h2 style="text-align: center;"><b>Домашнее задание: соревнование на Kaggle по распознаванию одежды</b></h2>

# ---

# Всем привет!  
# 
# Надеемся, что вам показались интересными и понятными лекция и семинар по многослйным нейросетям и PyTorch. Если же Вы ещё не успели ими насладиться -- просьба посмотреть видео на нашем канале и просмотреть ноутбуки с семинара, в этом ноутбуке эти знания будут использоваться на практике.

# <h2 style="text-align: center;"><b>FashionMNIST</b></h2>

# <img src="https://emiliendupont.github.io/imgs/mnist-chicken/mnist-and-fashion-examples.png">

# Выше изображены примеры того, с чем мы будем работать -- чёрно-белые изображения одежды. Слева более классический датасет -- MNIST, он же датасет рукописных цифр. Мы решили, что вам будет интереснее всё же рнаучить машину распознавать одежду (спойлер: с рукописными цифрами такой подход это тоже будут работать ;)

# <h3 style="text-align: center;"><b>Ссылка на соревнование: https://www.kaggle.com/c/dlschool-fashionmnist3. Вам нужно скачать оттуда всё из раздела `Data`, далее мы будем работать с этим - обучаться на train и предсказывать на test.</b></h3> 

# <h4 style="text-align: center;"><b>Оргиниальный датасет: https://www.kaggle.com/zalando-research/fashionmnist</b></h4> 

# После скачивания (скачанный архив распакуйте в одну папку с этим ноутбуком) и регистрации на Kaggle Вам нужно вступить в соревнование (по ссылке выше) и прочитать его описание.
# 
# <h3 style="text-align: center;"><b>Пожалуйста, укажите в соревновании свой ник == вашему нику на Canvas, иначе мы не сможем потом поставить вам баллы</b></h3>

# Платформа **Kaggle** -- основная платформа для соревнований в Data Science, так что привыкайте ;)

# <h2 style="text-align: center;"><b>Данные</b></h2>

# Мы будем работать с картинками одежды (чёрно-белыми, то есть цветовых каналов не 3, а 1). По входной картинке нужно предсказать тип одежды. Давайте посмотрим на то, что за датасет мы скачали:

# In[71]:


import pandas as pd
import os


# In[85]:


TRAIN_PATH = './fashionmnist/fashion-mnist_train.csv'
TEST_PATH = './fashionmnist/fashion-mnist_test.csv'


# In[86]:


train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)


# In[87]:


test_df


# Выведем начало таблиц:

# In[88]:


train_df.head()


# In[89]:


test_df.head()


# Выведем размеры обучающей и тестовой выборок:

# In[7]:


train_df.shape


# In[8]:


test_df.shape


# Что значат эти размеры и числа внутри DataFrame'ов? Всё просто -- **каждая строчка соответствует одной картинке**, а **столбцы -- это значения в пикселях этой кратинки**. **Первый столбец в train_df говорит о типе (классе) одежды (от 0 до 9)**.  
# 
# Однако перед тем, как двигаться дальше, краткая информация о представлении изображений в компьютере:

# <h2 style="text-align: center;"><b>Изображения</b></h2>

# <p align=center><img src="https://openclipart.org/image/2400px/svg_to_png/136057/1304647802.png" width=300 height=300></p>

# Как и вся информация, изображения представляются в компьютере числами. Стандартное цветовое пространство, с помощью которого декодируют и отрисовывают изображение -- это RGB (Red, Green и Blue). Каждая комбинация трёх чисел от 0 до 255 задаёт какой-то цвет. Например, (255,255,255) задаёт белый цвет, (255,0,0) -- красный. Также происходит и при загрузке картинок в Python, давайте посмотрим напрмиере:

# * Загрузим произвольную цветную картинку с помощью matplotlib:

# In[9]:


import numpy as np
import matplotlib.pyplot as plt


# In[10]:


image_png = plt.imread('./fpmi_logo.png')


# In[11]:


image_png


# * Посмотрим на тип загруженного объекта:

# In[12]:


type(image_png)


# Интересно, картинка стала `numpy.array`. А какая его форма и что внутри?

# In[13]:


image_png.shape


# In[14]:


image_png[0].shape


# In[15]:


image_png[0]


# In[16]:


image_png.dtype


# То есть это какая матрица, а точнее тензор (потому что есть третья размерность), у которого 573 строки, 1579 столбцов и 4 канала. Можно представлять это себе как 4 наложенных друг на друга матрицы, каждая из которых отвечает за один цвет -- R, G и B.  Внутри всех этих матриц лежат числа типа float32, то есть вещественные. Тут стоит сказать, что это просто тонкости загрузки в matplotlib -- на самом деле это матрицы из целых числе от 0 до 255 (включительно).
# 
# Стоп, но ведь каналов 4, а не 3? Да, четвёртый канал в данном случае -- это альфа-канал, у .png картинок он обычно присутствует. Давайте попробуем загрузить .jpg картинку:

# In[17]:


image_jpg = plt.imread('./dlschool_logo.jpg')


# In[18]:


image_jpg.shape


# Отлично, теперь 3 канала и нам совсем не страшно -- это три матрицы 400 на 400, каждая из которых отвечает за один цвет. Давайте отрисуем две загруженные картинки с помощью matplotlib:

# In[19]:


plt.imshow(image_png);


# In[20]:


plt.imshow(image_jpg);


# В данном случае **пиксель** -- это кортеж (упорядоченная последовательность чисел), состоящий из трёх чисел (как в примере раньше, например, (255,0,0) -- полностью красный пиксель). 

# Итак, картинки -- это матрицы, состоящие из чисел, которые характеризуют насышенность данного пикселя определённым цветом цветом.  
# 
# Аналогично и с **чёрно-белыми изображениями** -- это просто матрица с одним каналом (то есть пксель -- это просто число), например, 28 на 28, каждое число которой от 0 до 255 характеризует яркость пикселя (насыщенность белым). 
# Например, 255 -- это полностью белый пиксель, 0 -- полностью чёрный. Пора посмотреть, с чем мы будем работать в соревновании.

# <h2 style="text-align: center;"><b>Данные (2)</b></h2>

# Вернёмся к данным:

# In[21]:


train_df


# In[22]:


test_df


# Самый первый столбец -- **label**. Подробнее:

# каждая картинка иметт класс от 0 до 9, расшифровка меток класса:  
# 
# |class_id|class_name|
# |----|----|
# |0| T-shirt/top|
# |1| Trouser|
# |2| Pullover|
# |3| Dress|
# |4| Coat|
# |5| Sandal|
# |6| Shirt|
# |7| Sneaker|
# |8| Bag|
# |9| Ankle boot| 

# In[90]:


train_df.values[0]


# Видно, что это картинка типа Pullover (класс 2).

# **Примечание:** у тестового датасета нужно удалить столбец label (по понятным причинам) -- вам нужно будет его предсказать и отправить эти предсказания в Kaggle.

# Итак, мы имеем 60000 картинок, у каждой известна метка класса (то есть что это за одежда).  
# Отделим `X` (признаковое описание объектов) и `y` (метки классов):

# In[91]:


X_train = train_df.values[:, 1:]
y_train = train_df.values[:, 0]

X_test = test_df.values  # [:, 1:]  # удаляем столбец 'label'


# In[25]:


print(X_train.shape, y_train.shape)


# In[26]:


print(X_test.shape)


# Но почему пиксели так странно представлены? На самом деле 784 пикселя -- это 28 * 28, то есть это "развёрнутая в строку" чёрно-белая картинка 28 на 28 пикселей.
# 
# Давайте убедимся в этом, отрисовав несколько (можете менять индекс и смотрть на отрисовку):

# In[32]:


plt.imshow(X_train[0].reshape(28, 28), cmap='gray');


# Не слишком похоже на пулловер, правда? :)  
#     
# Просто если мы будем использовать изображения большего разрешения, нам понадобятся бОльшие вычислительные мощности, поэтому пока что будем довольствоваться такими размерами.

# Отлично, мы убедились в том, что имеем 60k картинок с метками для обучения, картинки "развёрнуты" в строку. Зачем разворачивать в строку? Потому что каждый пиксель в данном случае -- это один признак, то есть всего 784 признака, и уже их мы будем взвешивать нашей нейросетью, то есть у одного нейрона на входном слое будет 784 веса (+ Bias,  то есть 785 весов), на каждый пиксель по весу, и дальше уже будут второй слой, третий слой и так далее..

# Время тренировать нейросети!

# <h2 style="text-align: center;"><b>Нейросеть на PyTorch</b></h2>

# Надеемся, что вы уже прорешали семинар, там довольно подробно всё описано. На всякий случай ещё раз напомним, из чего состоит процесс обучения нейросети:

# - непосредственно, сама **архитектура** нейросети (сюда входят, например, типы функций активации у каждого нейрона);
# - начальная **инициализация** весов каждого слоя;
# - метод **оптимизации** нейросети (сюда ещё входит метод изменения `learning_rate`);
# - размер **батчей** (`batch_size`);
# - количество **итераций обучения** (`num_epochs`);
# - **функция потерь** (`loss`);  
# - тип **регуляризации** нейросети (для каждого слоя можно свой);  
# 
# То, что связано с ***данными и задачей***:  
# - само **качество** выборки (непротиворечивость, чистота, корректность постановки задачи);  
# - **размер** выборки;  

# In[33]:


import matplotlib.pyplot as plt
import numpy as np

import torch


# Проверим версию PyTorch:

# In[34]:


torch.__version__


# Сначала обернём данные в тензоры пайторча (может занять некоторое время):

# In[92]:


X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train.astype(np.int64))


# Проверим:

# In[93]:


print(X_train_tensor.shape, y_train_tensor.shape)


# In[94]:


y_train_tensor.unique()


# На лекции обсуждалось, что нельзя просто запихнуть в LogLoss (основная функция потерь для задачи классификации, [как мы помним](https://drive.google.com/open?id=15wdyreZufKDxNQ55v4cl4Em2rtj7Q45B)) метки классов, предлагаем вам самим ещё раз подумать, почему. На всякий случай -- [ноутбук с более подробной информацией о функциях потерь](https://drive.google.com/open?id=1j6WpzeJQV1kS1Os4VJ0Avf68OkXVBo6W).

# Так вот, нам надо преобразовать метки классов из целых чисел в OneHot-кодированные метки (если вам не знакомо это слово, [посмотрите первую половину этого видео](https://www.youtube.com/watch?v=ufkDhrngcr0)):

# In[95]:


length = y_train_tensor.shape[0]
num_classes = 10  # количество классов, в нашем случае 10 типов одежды

# закодированные OneHot-ом метки классов
y_onehot = torch.FloatTensor(length, num_classes)

y_onehot.zero_()
y_onehot.scatter_(1, y_train_tensor.view(-1, 1), 1)

print(y_train_tensor)
print(y_onehot)


# Видим, что наши метки перешли в вид "единица там, где номер класса, а остальные нули".

# Напишем код, очень похожий на код с семинара: возьмём два слоя -- входной и один скрытый (выходной обычно не считают, но он тоже есть):

# In[96]:


# N - размер батча (batch_size, нужно для метода оптимизации)
# D_in - размерность входа (количество признаков у объекта)
# H - размерность скрытых слоёв; 
# D_out - размерность выходного слоя (суть - количество классов)
D_in, H, D_out = 784, 100, 10

# определим нейросеть:
net = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
    torch.nn.Softmax()
)


# Обратите внимание:  
# 
# `D_in` -- это входная размерность (784 признака -- пикселя)  
# `D_out` -- выходная размерность (10 классов -- типов одежды), то есть 10 нейронов на выходном слое  
# `H` -- количество нейронов в скрытом слое  

# Осталось выбрать Loss (функцию потерь) и метод оптимизации, с помощью которого мы будем считать градиенты и обновлять с помощью них обновлять веса.  
# 
# Loss мы выберем CrossEntropy, то есть кросс-энтропию, этот лосс почти всегда используется в задаче многоклассовой классификации (см. лекцию и ноутбук [loss_functions.ipynb](https://drive.google.com/open?id=1j6WpzeJQV1kS1Os4VJ0Avf68OkXVBo6W), там всё подробно объясняется), а метод оптимизации выберем обычный SGD (Stochastic Gradient Descent, стохастический градиентный спуск, см. лекцию про нейрон).

# ---

# Однако перед тем, как перейти к коду обучения нейросети, есть одна тонкость -- **батчи**, а точнее **мини-батчи**.
# 
# **Мини-батчи** -- это небольшие (обычно размера 16, 32 или 64) "куски" выборки, то есть мини-батч размера 64 -- это 64 объекта из датасета. Обычно мини-батч называют просто батч (batch).

# Так вот: методы оптимиазции по типу стохастического градиентного спуска часто считаются не под одному объекту (в этом случае оптимизация будет очень нестабильная, "шумная"), а по нескольким -- по батчу. То есть в обычном градиентном спуске будет сумма по всей выборке, в стохастическом (чистом варианте) -- по одному объекту, а "между ними" -- мини-батч SGD, то есть подсчёт градиентов на небольшом кусочке данных.

# Одна **итерация (iteration)** алгоритма оптимизации -- это проход по одному батчу.
# Одна **эпоха (epoch)** алгоритма оптимизации -- это проход по всей выборке. 
# 
# То есть, например, если выборка размера 60000, а батч размера 64, то одна эпоха занимает 60000 / 64 = 937,5 = 938 итераций.

# ---

# Напишем функцию, генерирующую батчи:

# In[97]:


def generate_batches(X, y, batch_size=64):
    for i in range(0, X.shape[0], batch_size):
        X_batch, y_batch = X[i:i+batch_size], y[i:i+batch_size]
        yield X_batch, y_batch


# Код обучения нейросети (обязателньо убедитесь, что понимаете, что делает каждая строчка -- это необходимо для ваших дальнейших экспериментов):

# In[98]:


BATCH_SIZE = 64
NUM_EPOCHS = 10

loss_fn = torch.nn.CrossEntropyLoss(size_average=False)

learning_rate = 1e-4
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

for epoch_num  in range(NUM_EPOCHS):
    iter_num = 0
    running_loss = 0.0
    for X_batch, y_batch in generate_batches(X_train_tensor, y_train_tensor, BATCH_SIZE):
        # forward (подсчёт ответа с текущими весами)
        y_pred = net(X_batch)

        # вычисляем loss'ы
        loss = loss_fn(y_pred, y_batch)
        
        running_loss += loss.item()
        
        # выводем качество каждые 2000 батчей
            
        if iter_num % 100 == 99:
            print('[{}, {}] current loss: {}'.format(epoch_num, iter_num + 1, running_loss / 2000))
            running_loss = 0.0
            
        # зануляем градиенты
        optimizer.zero_grad()

        # backward (подсчёт новых градиентов)
        loss.backward()

        # обновляем веса
        optimizer.step()
        
        iter_num += 1


# Отлично, мы получили обученную нейросеть. Давайте измерим качество на обучающей выбоорке:

# In[42]:


class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
           'Sandal', 'Shirt', 'Sneaker','Bag', 'Ankle boot']

with torch.no_grad():
    for X_batch, y_batch in generate_batches(X_train_tensor, y_train_tensor, BATCH_SIZE):
        y_pred = net(X_batch)
        _, predicted = torch.max(y_pred, 1)
        c = (predicted == y_batch).squeeze()
        for i in range(len(y_pred)):
            label = y_batch[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))


# Уже сейчас видно, что сеть далеко не идеально -- она предсказывает только 7 классов, а про некоторые просто "забывает".

# Теперь предскажем на тестовой и сохраним предсказания в файл. Это ни что иное, как baseline, который вам надо побить, чтобы получить хоть какие-то ненулевые баллы за это ДЗ.

# In[43]:


y_test_pred = net(torch.FloatTensor(X_test))


# In[44]:


y_test_pred.shape


# In[45]:


y_test_pred[:5]


# Преобразуем OneHot'ы в числовые метки:

# In[46]:


_, predicted = torch.max(y_test_pred, 1)

predicted


# Сохраним в датафрейм:

# In[47]:


answer_df = pd.DataFrame(data=predicted.numpy(), columns=['Category'])
answer_df.head()


# In[48]:


answer_df['Id'] = answer_df.index


# In[49]:


answer_df.head()


# In[50]:


answer_df.tail()


# Отлично, созраним в файл и отправим:

# In[51]:


answer_df.to_csv('./baseline.csv', index=False)


# В точности этот файл и есть **baseline.csv**, который вы видите на лидерборде и который вам нужно побить.

# <h2 style="text-align: center;"><b>Задание</b></h2>

# Добейтесь как можно лучшего качества в соревновании!  
# 
# Используйте знания, полученные на занятиях и те, которые вы найдёте в интернете. Если у вас получится, можете использовать и свёрточные нейросети, а не только полносвязные. Вам нужно как минимум побить baseline.

# *Рекомендация*: попробуйте поменять количество итераций, количество нейронов, количество слоёв, гиперпараметры сети (learning_rate, метод оптимизации вместо SGD можно взять другой)

# In[132]:


X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train.astype(np.int64))

length = y_train_tensor.shape[0]
num_classes = 10  # количество классов, в нашем случае 10 типов одежды

# закодированные OneHot-ом метки классов
y_onehot = torch.FloatTensor(length, num_classes)

y_onehot.zero_()
y_onehot.scatter_(1, y_train_tensor.view(-1, 1), 1)



# N - размер батча (batch_size, нужно для метода оптимизации)
# D_in - размерность входа (количество признаков у объекта)
# H - размерность скрытых слоёв; 
# D_out - размерность выходного слоя (суть - количество классов)
D_in, H, D_out = 784, 100, 10

# определим нейросеть:
net = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
    torch.nn.Softmax()
)


BATCH_SIZE = 64
NUM_EPOCHS = 200

loss_fn = torch.nn.CrossEntropyLoss(size_average=False)

learning_rate = 1e-4
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

for epoch_num  in range(NUM_EPOCHS):
    iter_num = 0
    running_loss = 0.0
    for X_batch, y_batch in generate_batches(X_train_tensor, y_train_tensor, BATCH_SIZE):
        # forward (подсчёт ответа с текущими весами)
        y_pred = net(X_batch)

        # вычисляем loss'ы
        loss = loss_fn(y_pred, y_batch)
        
        running_loss += loss.item()
        
        # выводем качество каждые 2000 батчей
            
        if iter_num % NUM_EPOCHS*10 == 99:
            print('[{}, {}] current loss: {}'.format(epoch_num, iter_num + 1, running_loss / 2000))
            running_loss = 0.0
            
        # зануляем градиенты
        optimizer.zero_grad()

        # backward (подсчёт новых градиентов)
        loss.backward()

        # обновляем веса
        optimizer.step()
        
        iter_num += 1


# In[133]:


class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
           'Sandal', 'Shirt', 'Sneaker','Bag', 'Ankle boot']

with torch.no_grad():
    for X_batch, y_batch in generate_batches(X_train_tensor, y_train_tensor, BATCH_SIZE):
        y_pred = net(X_batch)
        _, predicted = torch.max(y_pred, 1)
        c = (predicted == y_batch).squeeze()
        for i in range(len(y_pred)):
            label = y_batch[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))


# In[134]:


y_test_pred = net(torch.FloatTensor(X_test))

_, predicted = torch.max(y_test_pred, 1)

predicted

answer_df = pd.DataFrame(data=predicted.numpy(), columns=['Category'])
answer_df.head()

answer_df['Id'] = answer_df.index

name = input("try name - ")

try:
    os.stat("submits")
except:
    os.mkdir("submits")  

answer_df.to_csv('./submits/%s.csv' %(name), index=False)


# <h3 style="text-align: center;"><b>Полезные ссылки</b></h3>

# 1). *Примеры написания нейросетей на PyTorch (офийиальные туториалы) (на английском): https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#examples  
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html*
# 
# 2). ***Один из самых подробных и полных курсов по deep learning на данный момент - это курс Стэнфордского Университета (он вообще сейчас один из лидеров в области ИИ, его выпускники работают в Google, Facebook, Amazon, Microsoft, в стартапах в Кремниевой долине):  http://cs231n.github.io/***  
# 
# 3). Практически исчерпывающая информация по основам нейросетей (из cs231n) (на английском):  
# 
# http://cs231n.github.io/neural-networks-1/,  
# http://cs231n.github.io/neural-networks-2/,  
# http://cs231n.github.io/neural-networks-3/,  
# http://cs231n.github.io/neural-networks-case-study/#linear
# 
# 4). *Хорошие статьи по основам нейросетей (на английском):  http://neuralnetworksanddeeplearning.com/chap1.html*
# 
# 5). *Наглядная демонстрация того, как обучаются нейросети:  https://cs.stanford.edu/people/karpathy/convnetjs/*

# 6). *Подробнее про backprop -- статья на Medium: https://medium.com/autonomous-agents/backpropagation-how-neural-networks-learn-complex-behaviors-9572ac161670*

# 7). *Статья из интернет по Backprop: http://page.mi.fu-berlin.de/rojas/neural/chapter/K7.pdf*
