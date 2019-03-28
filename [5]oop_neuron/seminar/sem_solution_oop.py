
# coding: utf-8

# <img src="https://s8.hostingkartinok.com/uploads/images/2018/08/308b49fcfbc619d629fe4604bceb67ac.jpg" width=500, height=450>
# <h3 style="text-align: center;"><b>Физтех-Школа Прикладной математики и информатики (ФПМИ) МФТИ</b></h3>

# ---

# <h2 style="text-align: center;"><b>Объектно-ориентированное программирование: решение</b></h2>

# ---

# Представьте себе, что вы проектируете автомобиль. Вы знаете, что автомобиль должен содержать двигатель, подвеску, две передних фары, 4 колеса, и т.д. Вы описываете все запчасти, из которых состоит ваш автомобиль, а также то, каким образом эти запчасти взаимодействуют между собой. Кроме того, вы описываете, что должен сделать пользователь, чтобы машина затормозила, или включился дальний свет фар. Результатом вашей работы будет некоторый эскиз. Вы только что разработали то, что в ООП называется класс.

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image
from IPython.core.display import HTML 
Image(url= "https://cs5.pikabu.ru/post_img/big/2015/12/14/6/1450080142173639353.jpg")


# <b>Классы</b> играют роль фабрик экземпляров. Их атрибуты обеспечивают поведение – данные и функции – то есть наследуются всеми экземплярами, созданными от них (например, функция, вычисляющая стоимость поездки, исходя
# из расхода топлива).

# <b>Экземпляры</b> представляют конкретные элементы программы. Их атрибуты хранят данные, которые могут отличаться в  конкретных объектах (например, количество колес).

# Команда интерпретации и запуска — python start.py

# In[2]:


class Start:
    def say_hello (self):
        print('Hello, world!')
Start().say_hello() 


# Создадим class Car: (object можно не указывать)

# In[3]:


class Car(object):
    def __init__(self, model):
        self.model = model

my_car = Car("BMW")
my_car


# In[4]:


my_car.model


# Попробуйте создать свой сlass Car, где при создании объекта будет указываться ещё и passengers, color, speed.

# In[5]:


#your code here:
class Car:
    def __init__(self, model, passengers, color, speed):
        self.model = model
        self.passengers = passengers
        self.color = color
        self.speed = speed


# Теперь создайте несколько машин с разными параметрами - атрибутами экземпляров:

# In[6]:


#your code here:
bmw = Car("BMW", 4, "red", 5)
ferrari = Car("Ferrari", 2, "black", 10)
ford = Car("Ford", 6, "blue", 6)


# Теперь попробуйте покрасить любую машины в любимый ваш цвет:

# In[7]:


#your code here:
bmw.color = 'yellow'
bmw.color


# Добавим к class Car метод для работы с экземплярами данного класса - атрибут класса 

# In[10]:


class Car:
    def __init__(self, model, passengers, color, speed):
        self.model = model
        self.passengers = passengers
        self.color = color
        self.speed = speed
    def set_cost(self, cost):
        self.cost = cost
    


# In[11]:


Car.set_cost


# Установим цену на BMW:

# In[12]:


bmw = Car("BMW", 4, "red", 5)
bmw.set_cost(3 * 10 ** 6)
bmw.cost


# Добавьте к class Car метод, который вычисляет стоимость машины через несколько лет:

# In[13]:


#class Car:
    #...
    #def new_cost(..., year):
        #k = year ** 2 * (3 * 10 ** 4)    коэффициент уменьшение цены
        #...
#your code hear
class Car:
    def __init__(self, model, passengers, color, speed):
        self.model = model
        self.passengers = passengers
        self.color = color
        self.speed = speed
    def set_cost(self, cost):
        self.cost = cost
    def new_cost(self, year):
        k = year ** 2 * (3 * 10 ** 4)
        self.cost = self.cost - k
        return self.cost


# Вычислите эту стоимость в течение 5 лет с периодом 1 год и нарисуте график:

# In[14]:


#your code hear
bmw = Car("BMW", 4, "red", 5)
bmw.set_cost(3 * 10 ** 6)
years = range(0, 6)
costs = [bmw.new_cost(year) for year in years]
plt.plot(years, costs)
plt.show()

