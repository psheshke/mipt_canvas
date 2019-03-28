
# coding: utf-8

# <p style="align: center;"><img align=center src="https://s8.hostingkartinok.com/uploads/images/2018/08/308b49fcfbc619d629fe4604bceb67ac.jpg"  width=500 height=400></p>
# 
# <h3 style="text-align: center;"><b>Физтех-Школа Прикладной математики и информатики (ФПМИ) МФТИ</b></h3>

# ---

# <h3 style="text-align: center;"><b>Элементы теории оптимизации. Производные и частные производные.</b></h3>

# <p style="text-align: center;">(На основе https://github.com/romasoletskyi/Machine-Learning-Course)</p>

# <h3 style="text-align: center;"><b>Приращение линейной функции</b></h3>

# Давайте рассмотрим линейную функцию $y=kx+b$ и построим график: <br>  
# 
# ![source: Wikipedia](https://upload.wikimedia.org/wikipedia/commons/c/c1/Wiki_slope_in_2d.svg) <br>  
# 
# Введём понятие **приращения** функции в точке $(x, y)$ как отношение вертикального изменения (измненеия функции по вертикали) $\Delta y$ к горизонтальному изменению $\Delta x$ и вычислим приращение для линейной функции:  
# 
# $$приращение ("slope")=\frac{\Delta y}{\Delta x}=\frac{y_2-y_1}{x_2-x_1}=\frac{kx_2+b-kx_1-b}{x_2-x_1}=k\frac{x_2-x_1}{x_2-x_1}=k$$  
# 
# Видим, что приращение в точке у прямой не зависит от $x$ и $\Delta x$.

# <h3 style="text-align: center;"><b>Приращение произвольной функции</b></h3>

# Но что, если функция не линейная, а произвольная $f(x)$?  
# В таком случае просто нарисуем **касательную ** в точке, в которой ищем приращение, и будем смотреть уже на приращение касательной. Так как касательная - это прямая, мы уже знаем, какое у неё приращение (см. выше).
# ![source: Wikipedia](https://upload.wikimedia.org/wikipedia/commons/d/d2/Tangent-calculus.svg)

# Имея граик функции мы, конечно, можем нарисовать касательную в точке. Но часто функции заданы аналитически, и хочется уметь сразу быстро получать формулу для приращения функциии в точке. Тут на помощь приходит **производная**.  Давайте посмотрим на определение производной его с нашим понятием приращения:  
# 
# $$f'(x) = \lim_{\Delta x \to 0}\frac{\Delta y}{\Delta x} = \lim_{\Delta x \to 0}\frac{f(x + \Delta x) - f(x)}{\Delta x}$$  
# 
# То есть по сути, значение производной функции в точке - это и есть приращение функции, если мы стремим длину отрезка $\Delta x$ к нулю.

# Посомтрим на интерактивное демо, демонстрирующее стремление $\Delta x$ к нулю (*в Google Colab работать не будет!*):

# In[1]:


from __future__ import print_function

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

import matplotlib.pyplot as plt
import numpy as np


# In[2]:


# pip install ipywidgets


# In[3]:


@interact(lg_z=(-0.5,4.0,0.1))
def f(lg_z=1.0):
    z = 10 ** lg_z
    x_min = 1.5 - 6/z
    x_max = 1.5 + 6/z
    l_min = 1.5 - 4/z
    l_max = 1.5 + 4/z
    xstep = (x_max - x_min)/100
    lstep = (l_max - l_min)/100
    
    x = np.arange(x_min, x_max, xstep)
    
    plt.plot(x, np.sin(x), '-b')     
    
    plt.plot((l_min,l_max), (np.sin(l_min), np.sin(l_max)), '-r')
    plt.plot((l_min,l_max), (np.sin(l_min), np.sin(l_min)), '-r')
    plt.plot((l_max,l_max), (np.sin(l_min), np.sin(l_max)), '-r')
    
    yax = plt.ylim()    
    
    plt.text(l_max + 0.1/z, (np.sin(l_min) + np.sin(l_max)) / 2, "$\Delta y$")
    plt.text((l_min + l_max)/2, np.sin(l_min) - (yax[1]-yax[0]) / 20, "$\Delta x$")
    
    plt.show()
    
    print('slope =', (np.sin(l_max) - np.sin(l_min)) / (l_max - l_min))


# Видим, что при уменьшении отрезка $\Delta x$, значение приращения стабилизируется (перестаёт изменяться). Это число и есть приращение функции в точке, равное проиводной функции в точке. Производную функции $f(x)$ в точке x обознают как $f'(x)$ или как $\frac{d}{dx}(f(x))$.  

# <h3 style="text-align: center;"><b>Пример вычисления проиводной</b></h3>

# Возьмём производную по определению:

# 1. $f(x)=x$  
# 
# $$\frac{\Delta y}{\Delta x}=\frac{x+\Delta x-x}{\Delta x}=1\Rightarrow \mathbf{\frac{d}{dx}(x)=1}$$  
# 
# 2. $f(x)=x^2$  
# 
# $$\frac{\Delta y}{\Delta x}=\frac{(x+\Delta x)^2-x^2}{\Delta x}=\frac{x^2+2x\Delta x+\Delta x^2-x^2}{\Delta x}=2x+\Delta x\rightarrow 2x (\Delta x\rightarrow 0)\Rightarrow \mathbf{\frac{d}{dx}(x^2)=2x}$$  
#     
# 3. В общем случае для степенной функции $f(x)=x^n$ формула будет такой:  
# 
# $$\mathbf{\frac{d}{dx}(x^n)=nx^{n-1}}$$  

# <h3 style="text-align: center;"><b>Правила вычисления проиводной</b></h3>

# Выпишет правила *дифференцирования*:  
# 
# 1). Если $f(x)$ - константа, то её производная (приращение) 0:  
# 
# $$(C)' = 0$$
# 
# 2). Производная суммы функций - это сумма производных:  
# 
# $$(f(x) + g(x))' = f'(x) + g'(x)$$
# 
# 3). Производная разности - разность производных:  
# 
# $$(f(x) - g(x))' = f'(x) - g'(x)$$
# 
# 4). Производная произведения функций:  
# 
# $$(f(x)g(x))' = f'(x)g(x) + f(x)g'(x)$$
# 
# 5). Производная частного:  
# 
# $$\left(\frac{f(x)}{g(x)}\right)'=\frac{f'(x)g(x)-g'(x)f(x)}{g^2(x)}$$
# 
# 6). Производная сложной функции ("правило цепочки", "chain rule"):  
# 
# $$(f(g(x)))'=f'(g(x))g'(x)$$
# 
# Можно записать ещё так:  
# 
# $$\frac{d}{dx}(f(g(x)))=\frac{df}{dg}\frac{dg}{dx}$$

# **Примеры**:

# * Вычислим производную функции $$f(x) = \frac{x^2}{cos(x)} + 100$$:  
# 
# $$f'(x) = \left(\frac{x^2}{cos(x)}+100\right)' = \left(\frac{x^2}{cos(x)}\right)' + (100)' = \frac{(2x)\cos(x) - x^2(-\sin(x))}{cos^2(x)}$$

# * Вычислим производную функции $$f(x) = tg(x)$$:  
# 
# $$f'(x) = \left(tg(x)\right)' = \left(\frac{\sin(x)}{\cos(x)}\right)' = \frac{\cos(x)\cos(x) - \sin(x)(-\sin(x))}{cos^2(x)} = \frac{1}{cos^2(x)}$$

# <h3 style="text-align: center;"><b>Частные производные</b></h3>

# Когда мы имеем функци многих переменных, её уже сложнее представить себе в виде рисунка (в случае более 3-х переменных это действительно не всем дано). ОДнако формальные правила взятия производной у таких функций созраняются. Они в точности совпадают с тоеми, которые рассмотрены выше для функции одной переменной.  
# 
# Итак, правило взятия частной производной функции мнгих переменных:  
# 1). Пусть $f(\overline{x}) = f(x_1, x_2, .., x_n)$ - функция многих переменных;  
# 2). Частная проиводная по $x_i$ это функции - это производная по x_i, считая все остальные переменные **константами**. 
# 
# Более математично:  
# 
# Частная производная функции $f(x_1,x_2,...,x_n)$ по $x_i$ равна  
# 
# $$\frac{\partial f(x_1,x_2,...,x_n)}{\partial x_i}=\frac{df_{x_1,...,x_{i-1},x_{i+1},...x_n}(x_i)}{dx_i}$$  
# 
# где $f_{x_1,...,x_{i-1},x_{i+1},...x_n}(x_i)$ означает, что переменные $x_1,...,x_{i-1},x_{i+1},...x_n$ - это фиксированные значения, и с ними нужно обращаться как с константами.

# **Примеры**:   

# * Найдём частные производные функции $f(x, y) = -x^7 + (y - 2)^2 + 140$ по $x$ и по $y$:  
# 
# $$f_x'(x, y) = \frac{\partial{f(x, y)}}{\partial{x}} = -7x^6$$  
# $$f_y'(x, y) = \frac{\partial{f(x, y)}}{\partial{y}} = 2(y - 2)$$

# * Найдём частные производные функции $f(x, y, z) = \sin(x)\cos(y)tg(z)$ по $x$, по $y$ и по $z$:  
# 
# $$f_x'(x, y) = \frac{\partial{f(x, y)}}{\partial{x}} = \cos(x)\cos(y)tg(z)$$  
# $$f_y'(x, y) = \frac{\partial{f(x, y)}}{\partial{y}} = \sin(x)(-\sin(y))tg(z)$$
# $$f_z'(x, y) = \frac{\partial{f(x, y)}}{\partial{y}} = \frac{\sin(x)\cos(y)}{\cos^2{z}}$$

# <h3 style="text-align: center;"><b>Градиентный спуск</b></h3>

# **Градиентом** функции $f(\overline{x})$, где $\overline{x} \in \mathbb{R^n}$, то есть $\overline{x} = (x_1, x_2, .., x_n)$, называется вектор из частных производных функции $f(\overline{x})$:  
# 
# $$grad(f) = \nabla f(\overline{x}) = \left(\frac{\partial{f(\overline{x})}}{\partial{x_1}}, \frac{\partial{f(\overline{x})}}{\partial{x_2}}, .., \frac{\partial{f(\overline{x})}}{\partial{x_n}}\right)$$

# Есть функция $f(x)$. Хотим найти аргумент, при котором она даёт минимум.
# 
# Алгоритм градиентного спуска:  
# 1. $x^0$ - начальное значение (обычно берётся просто из разумных соображений или случайное);  
# 2. $x^i = x^{i-1} - \alpha \nabla f(x^{i-1})$, где $\nabla f(x^{i-1})$ - это градиент функции $f$, в который подставлено значение $x^{i-1}$;
# 3. Выполнять пункт 2, пока не выполнится условие остановки: $||x^{i} - x^{i-1}|| < eps$, где $||x^{i} - x^{i-1}|| = \sqrt{(x_1^i - x_1^{i-1})^2 + .. + (x_n^i - x_n^{i-1})^2}$.  

# **Примеры:**

# * *Пример 1*: Посчитаем формулу градиентного спуска для функции $f(x) = 10x^2$:   

# $x^i = x^{i-1} - \alpha \nabla f(x^{i-1}) = x^{i-1} - \alpha f'(x^{i-1}) = x^{i-1} - \alpha (20x^{i-1})$

# Имея эту формулу, напишем код градиентного спуска для функции $f(x) = 10x^2$:

# In[30]:


import numpy as np
from tqdm import tqdm

def f(x):
    return 10 * x**2

def gradient_descent(alpha=0.01, eps=0.001):
    x_pred = 100  # начальная инициализация
    x = 50  # начальная инициализация
    for _ in tqdm(range(100000)):
        print("Step:",_,"\tX =", round(x,5),"\tf(X) =", round(f(x), 5))  # смотрим, на каком мы шаге
        if np.sum((x - x_pred)**2) < eps**2:  # условие остановки
            break
        x_pred = x
        x = x_pred - 20 * alpha * x_pred  # по формуле выше
    return x


# In[31]:


x_min = gradient_descent(0.01, 0.01)


# In[32]:


x_min


# In[33]:


f(x_min)


# * *Пример 2*: Посчитаем формулу градиентного спуска для функции $f(x, y) = 10x^2 + y^2$:   

# $$\left(\begin{matrix} x^i \\ y^i \end{matrix}\right) = \left(\begin{matrix} x^{i-1} \\ y^{i-1} \end{matrix}\right) - \alpha \nabla f(x^{i-1}, y^{i-1}) = \left(\begin{matrix} x^{i-1} \\ y^{i-1} \end{matrix}\right) - \alpha \left(\begin{matrix} \frac{\partial{f(x^{i-1}, y^{i-1})}}{\partial{x}} \\ \frac{\partial{f(x^{i-1}, y^{i-1})}}{\partial{y}} \end{matrix}\right) = x^{i-1} - \alpha \left(\begin{matrix} 20x^{i-1} \\ 2y^{i-1} \end{matrix}\right)$$

# Осталось написать код, выполняющий градиентный спуск, пока не выполнится условие остановки, для функции $f(x, y) = 10x^2 + y^2$:

# In[38]:


import numpy as np
from tqdm import tqdm

def f(x):
    return 10 * x[0]**2 + x[1]**2

def gradient_descent(alpha=0.01, eps=0.001):
    x_prev = np.array([100, 100])  # начальная инициализация
    x = np.array([50, 50])  # начальная инициализация
    for _ in tqdm(range(100000)):
        print("Step:",_,"\tX =", np.round(x,3),"\tf(X) =", np.round(f(x), 5))  # смотрим, на каком мы шаге
        if np.sum((x - x_prev)**2) < eps**2:  # условие остановки
            break
        x_prev = x
        x = x_prev - alpha * np.array(20 * x_prev[0], 2 * x_prev[1])  # по формуле выше
    return x


# In[39]:


x_min = gradient_descent()


# In[40]:


x_min


# In[41]:


f(x_min)


# <h3 style="text-align: center;"><b>Домашнее задание</b></h3>

# 1). (только для тех, кто раньше брал производные) Вычислите производную функции $f(x)=\frac{1}{x}$ по определению и сравните с производной степенной функции в общем случае;  
# 2). Найдите производную функции $Cf(x)$, где С - число;  
# 3). Найдите производные функций:  
# 
# $$f(x)=x^3+3\sqrt{x}-e^x$$
# 
# $$f(x)=\frac{x^2-1}{x^2+1}$$
# 
# $$\sigma(x)=\frac{1}{1+e^{-x}}$$
# 
# $$L(y, \hat{y}) = (y-\hat{y})^2$$  
# 
# 4). Напишите формулу и код для градиентного спуска для функции:  
# $$f(w, x) = \frac{1}{1 + e^{-wx}}$$  
# 
# То есть по аналогии с примером 2 вычислите частные производные по $w$ и по $x$ и запишите формулу векторно (см. пример 2)
# 
# В задаче 3 производную нужно брать по $\hat{y}$.

# <h3 style="text-align: center;"><b>Полезные ссылки</b></h3>

# 0). Прикольный сайт с рисунками путём задания кривых уравнениями и функциями:  
# 
# https://www.desmos.com/calculator/jwshvscdzb
# 
# ***Производные:***
# 
# 1). Про то, как брать частные производные:  
# 
# http://www.mathprofi.ru/chastnye_proizvodnye_primery.html
# 
# 2). Сайт на английском, но там много видеоуроков и задач по производным:  
# 
# https://www.khanacademy.org/math/differential-calculus/derivative-intro-dc
# 
# 3). Задачи на частные производные:  
# 
# http://ru.solverbook.com/primery-reshenij/primery-resheniya-chastnyx-proizvodnyx/  
# 
# 4). Ещё задачи на частные проивзодные:  
# 
# https://xn--24-6kcaa2awqnc8dd.xn--p1ai/chastnye-proizvodnye-funkcii.html  
# 
# 5). Производные по матрицам:  
# 
# http://nabatchikov.com/blog/view/matrix_der  
# 
# ***Градиентны спуск:***
# 
# 6). [Основная статья по градиентному спуску](http://www.machinelearning.ru/wiki/index.php?title=%D0%9C%D0%B5%D1%82%D0%BE%D0%B4_%D0%B3%D1%80%D0%B0%D0%B4%D0%B8%D0%B5%D0%BD%D1%82%D0%BD%D0%BE%D0%B3%D0%BE_%D1%81%D0%BF%D1%83%D1%81%D0%BA%D0%B0)
# 
# 7). Статья на Хабре про градиетный спуск для нейросетей:  
# 
# https://habr.com/post/307312/  
# 
# ***Методы оптимизации в нейронных сетях:***
# 
# 8). Сайт с анимациями того, как сходятся алгоритмы градиентного спуска:
# www.denizyuret.com/2015/03/alec-radfords-animations-for.html
# 
# 9). Статья на Хабре про метопты (град. спуск) в нейронках:
# https://habr.com/post/318970/
# 
# 10). Ещё сайт (англ.) про метопты (град. спуск) в нейронках (очень подробно):
# http://ruder.io/optimizing-gradient-descent/
