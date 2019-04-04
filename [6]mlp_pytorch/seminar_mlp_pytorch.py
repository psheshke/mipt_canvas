
# coding: utf-8

# <img src="https://s8.hostingkartinok.com/uploads/images/2018/08/308b49fcfbc619d629fe4604bceb67ac.jpg" width=500,>
# <h3 style="text-align: center;"><b>Физтех-Школа Прикладной математики и информатики (ФПМИ) МФТИ</b></h3>

# ---

# <h2 style="text-align: center;"><b>Многослойная сеть на PyTorch</b></h3>

# В этом ноутбке мы научимся писать свои нейросети на фреймворке PyTorch, конкретно - рассмотрим, как написать многослойную полносвязную сеть (Fully-Connected, FC), и сравним их качество на датасете картинок MNIST.

# <h3 style="text-align: center;"><b>Компоненты нейросети</b></h3>

# Здесь самое время напомнить о том, какие вещи играют принципиальную роль в построении любой ***нейронной сети*** (все их мы задаём *руками*, самостоятельно):  
# 
# - непосредственно, сама **архитектура** нейросети (сюда входят типы функций активации у каждого нейрона);
# - начальная **инициализация** весов каждого слоя;
# - метод **оптимизации** нейросети (сюда ещё входит метод изменения `learning_rate`);
# - размер **батчей** (`batch_size`);
# - количество итераций обучения (`num_epochs`);
# - **функция потерь** (`loss`);  
# - тип **регуляризации** нейросети (для каждого слоя можно свой);  
# 
# То, что связано с ***данными и задачей***:  
# - само **качество** выборки (непротиворечивость, чистота, корректность постановки задачи);  
# - **размер** выборки;  

# <h3 style="text-align: center;"><b>Многослойная нейронная сеть</b></h3>

# Как можно понять из названия, многослойная нейросеть состоит из нескольких **слоёв**. Каждый слой состоит из **нейронов**. Ранее мы уже писали свой нейрон на NumPy, вот из таких нейронов и состоит ***MLP (Multi-Layer Perceptron)***. Ещё такую многослойную нейросеть, у которой каждый нейрон на предыдущем уровне соединён с нейроном на следующем уровне, называют ***Fully-Connected-сетью*** (или ***Dense-сетью***).  
# 
# Расмотрим их устройство более подробно:

# * Вот так выглядит двухслойная нейросеть (первый слой - input layer - не считается, потому что это, по сути, не слой):

# <img src="http://cs231n.github.io/assets/nn1/neural_net.jpeg" width=300, height=200>

# * Так выглядит трёхслойная нейросеть:

# <img src="http://cs231n.github.io/assets/nn1/neural_net2.jpeg" width=400>

# .. и так далее для большего случая слоёв.

# **Обратите внимание:** связи есть у нейронов со слоя $L_{i-1}$  и нейронов $L_{i}$, но между нейронами в одном слое связей **нет**.

# **Входной слой** -- это данные (матрица $(n, m)$).

# Слои, которые не являются входными или выходными, называются **скрытыми слоями (hidden layers)**.

# При решении ***задачи регрессии*** на **выходном слое** обычно один нейрон, который возвращает предсказанные числа (для каждого объекта по числу).  
# 
# В случае ***задачи классификации*** на **выходном слое** обычно один нейрон, если задача бинарной классификации, и $K$ нейронов, если задача $K$-класовой классификации.

# #### Forward pass в MLP

# Каждый слой многослойной нейросети - это матрица весов, строки которой - это нейроны (одна строка - один нейрон), а столбцы - веса каждого нейрона (то есть одна строка - это веса одного нейрона).

# Допустим, мы решаем задачу $K$-классовой классификации (на последнем слое $K$ нейронов). Рассмотрим, как в таком случае выглядит `forward_pass` нейросети:

# * Вход: $$X =
# \left(
# \begin{matrix} 
# x_{11} & ... & x_{1M} \\
# ... & \ddots  & ...\\
# x_{N1} & ... & x_{NM} 
# \end{matrix}
# \right)
# $$
# 
# -- матрица $(N, M)$

# * Структура сети - много слоёв, в слоях много нейронов. Первый слой (после входного) выглядит так:

# $$ W^1 =
# \left(
# \begin{matrix} 
# w_{11} & ... & w_{1L_1} \\
# ... & \ddots  & ...\\
# w_{M1} & ... & w_{ML_1} 
# \end{matrix}
# \right)
# $$
# 
# -- матрица $(M, L_1)$

# То есть это в точности $L_1$ нейронов, каждый имеет свои собственные веса, их $M$ штук.

# Мы помним, что нейрон - это линейное преобразование и потом нелинейная функция активации от этого преобразования. Однако в многослойных нейростеях часто отделяют `Linear` часть и `Activation`, то есть слоем считаем набор весов нейронов, а следующий слой всегда функция активации (у всех нейронов из слоя она одна и та же, обычно фреймворки не позволяют задавать конкретному нейрону в слое отличную от других нейронов в этом слое функцию активации, однако это легко сделать, объявив слой из одного нейрона).

# * Другие слои выглядит точно так же, как первый слой. Например, у второй слой будет такой:

# $$ W^2 =
# \left(
# \begin{matrix} 
# w_{11} & ... & w_{1L_2} \\
# ... & \ddots  & ...\\
# w_{L_11} & ... & w_{L_1L_2} 
# \end{matrix}
# \right)
# $$
# 
# -- матрица $(L_1, L_2)$

# То есть это в точности $L_2$ нейронов, каждый имеет свои собственные веса, их $L_1$ штук.

# * Выходной слой:  
# 
# Пусть в нейросети до выходного слоя идут $t$ слоёв. Тогда выходной слой имеет форму:

# $$ W^{out} =
# \left(
# \begin{matrix} 
# w_{11} & ... & w_{1K} \\
# ... & \ddots  & ...\\
# w_{L_t1} & ... & w_{L_tK} 
# \end{matrix}
# \right)
# $$
# 
# -- матрица $(L_t, K)$, где $L_t$ - количество нейронов в $t$-ом слое, а $K$ -- количество классов.

# В итоге ***для `forward_pass` нам нужно просто последовтельно перемножить матрицы друг за другом, применяя после каждого умножения соответсвующую функцию активации***.

# *Примечание*: можно думать об умножении на очередную матрицу весов как на переход в **новое признаковое пространство**. Действительно, когда подаём матрицу $X$ и умножаем на матрицу первого слоя, мы получаем матрицу размера $(N, L_1)$, то есть как будто $L_1$ "новых" признаков (построенных как линейная комбинация старых до применения функции активации, и уже как нелинейная комбинация после активации).

# **Backward pass в MLP**

# Обучается с помощью метода "Error Backpropagation" - ["Обратное распространение ошибки"](https://ru.wikipedia.org/wiki/%D0%9C%D0%B5%D1%82%D0%BE%D0%B4_%D0%BE%D0%B1%D1%80%D0%B0%D1%82%D0%BD%D0%BE%D0%B3%D0%BE_%D1%80%D0%B0%D1%81%D0%BF%D1%80%D0%BE%D1%81%D1%82%D1%80%D0%B0%D0%BD%D0%B5%D0%BD%D0%B8%D1%8F_%D0%BE%D1%88%D0%B8%D0%B1%D0%BA%D0%B8), принцип распространения очень похож на то, как мы обучали один нейрон - это градиентный спуск, но по "всей нейросети" сразу.  

# Backpropagation работает корректно благодаря ***chain rule*** (=правилу взятия производной сложной функции):  
# 
# Если $f(x) = f(g(x))$, то:  
# 
# $$\frac{\partial{f}}{\partial{x}} = \frac{\partial{f}}{\partial{g}} \frac{\partial{g}}{\partial{x}}$$

# Более подробно про backpropagation можно прочитать здесь (на английском):  https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

# <h3 style="text-align: center;"><b>Многослойная нейросеть на PyTorch</b></h3>

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

# Не будем медлить - бахнем 100 нейронов в двуслойную нейросеть (датасет - снова "Игрушка дьявола"):

# In[1]:


import matplotlib.pyplot as plt
import numpy as np

import torch


# Генерация и отрисовка датасета:

# In[2]:


# код для генерации взят из Стэнфордсокго курса:
# http://cs231n.github.io/neural-networks-case-study/#linear

N = 100
D = 2
K = 3
X = np.zeros((N * K, D))
y = np.zeros(N * K, dtype='uint8')

for j in range(K):
    ix = range(N * j,N * (j + 1))
    r = np.linspace(0.0, 1, N)
    t = np.linspace(j * 4, (j + 1) * 4,N) + np.random.randn(N) * 0.2 # theta
    X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
    y[ix] = j


# Не забываем оборачивать данные (без этого градиенты не посчитать):

# In[12]:


X = torch.autograd.Variable(torch.FloatTensor(X))
y = torch.autograd.Variable(torch.LongTensor(y.astype(np.int64)))


# In[13]:


print(X.data.shape, y.data.shape)


# Сама ячейка с сеткой и обучением:

# In[14]:


# пример взят из официального туториала: 
# https://pytorch.org/tutorials/beginner/pytorch_with_examples.html

# N - размер батча (batch_size, нужно для метода оптимизации); 
# D_in - размерность входа (количество признаков у объекта);
# H - размерность скрытых слоёв; 
# D_out - размерность выходного слоя (суть - количество классов)
N, D_in, H, D_out = 64, 2, 100, 3

# Use the nn package to define our model and loss function.
two_layer_net = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)

loss_fn = torch.nn.CrossEntropyLoss(size_average=False)

learning_rate = 1e-4
optimizer = torch.optim.SGD(two_layer_net.parameters(), lr=learning_rate)
for t in range(500):
    # forward
    y_pred = two_layer_net(X)

    # loss
    loss = loss_fn(y_pred, y)
    print('{} {}'.format(t, loss.data))

    # ЗАНУЛЯЕМ!
    optimizer.zero_grad()

    # backward
    loss.backward()

    # ОБНОВЛЯЕМ! 
    optimizer.step()


# **Обратите внимание:** несмотря на то, что это задача 3-х классовой классификации и столбец $y$ нужно по-хорошему кодировать OneHotEncoding'ом, мы подали просто столбец из 0, 1 и 2 и всё отработало. Вывод - PyTorch сам делает OneHot в таком случае.

# Проверим, насколько хороша наша сеть из 100 нейронов:

# In[15]:


# Обратно в Numpy для отрисовки
X = X.data.numpy()
y = y.data.numpy()


# In[16]:


# Отрисовочная магия, снова взято из:
# http://cs231n.github.io/neural-networks-case-study/#linear

h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
grid_tensor = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])

Z = two_layer_net(torch.autograd.Variable(grid_tensor))
Z = Z.data.numpy()
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 8))

plt.contourf(xx, yy, Z, cmap=plt.cm.rainbow, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.rainbow)

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

plt.title('Игрушка дьявола', fontsize=15)
plt.xlabel('$x$', fontsize=14)
plt.ylabel('$y$', fontsize=14)
plt.show();


# Качество не очень. Как же так, ведь мы использовали 100 нейронов? Разве их мало?

# ---

# ### Задача 2

# Улучшите сеть (помните про вещи, которые можно менять (см. Компоненты нейросети)). Экспериментируйте, в этом **вся суть deep learning** (и в том, чтобы рано или поздно сетка научилась экспериментировать за Вас :)

# In[22]:


# Ваш код здесь (можно (и нужно) создавать больше ячеек)

X = torch.autograd.Variable(torch.FloatTensor(X))
y = torch.autograd.Variable(torch.LongTensor(y.astype(np.int64)))


# пример взят из официального туториала: 
# https://pytorch.org/tutorials/beginner/pytorch_with_examples.html

# N - размер батча (batch_size, нужно для метода оптимизации); 
# D_in - размерность входа (количество признаков у объекта);
# H - размерность скрытых слоёв; 
# D_out - размерность выходного слоя (суть - количество классов)
N, D_in, H, D_out = 64, 2, 100, 3

# Use the nn package to define our model and loss function.
two_layer_net = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)

loss_fn = torch.nn.CrossEntropyLoss(size_average=False)

learning_rate = 1e-4
optimizer = torch.optim.Adam(two_layer_net.parameters(), lr=learning_rate)
for t in range(5000):
    # forward
    y_pred = two_layer_net(X)

    # loss
    loss = loss_fn(y_pred, y)
    print('{} {}'.format(t, loss.data))

    # ЗАНУЛЯЕМ!
    optimizer.zero_grad()

    # backward
    loss.backward()

    # ОБНОВЛЯЕМ! 
    optimizer.step()


# In[23]:


# Обратно в Numpy для отрисовки
X = X.data.numpy()
y = y.data.numpy()

# Отрисовочная магия, снова взято из:
# http://cs231n.github.io/neural-networks-case-study/#linear

h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
grid_tensor = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])

Z = two_layer_net(torch.autograd.Variable(grid_tensor))
Z = Z.data.numpy()
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 8))

plt.contourf(xx, yy, Z, cmap=plt.cm.rainbow, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.rainbow)

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

plt.title('Игрушка дьявола', fontsize=15)
plt.xlabel('$x$', fontsize=14)
plt.ylabel('$y$', fontsize=14)
plt.show();


# ---

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
