
# coding: utf-8

# <p style="align: center;"><img src="https://static.tildacdn.com/tild6636-3531-4239-b465-376364646465/Deep_Learning_School.png", width=300, height=300></p>
# 
# <h3 style="text-align: center;"><b>Физтех-Школа Прикладной математики и информатики (ФПМИ) МФТИ</b></h3>

# <h2 style="text-align: center;"><b>Библиотека <a href="http://pandas.pydata.org/">pandas</a>: домашнее задание</b></h2>

# Перед этим заданием необходимо просмотреть и прорешать соответствующий семинар. Он полезный, правда!

# In[ ]:


#!pip install pandas


# In[1]:


import pandas as pd
import numpy as np


# <img src="http://www.keepcalmstudio.com/_gallery/1500/kcs_dd487f3f.png" width=300 height=300>

# ### Даю DATA: https://www.kaggle.com/mehdidag/black-friday

# In[2]:


data = pd.read_csv('./black-friday/BlackFriday.csv')


# Посмотрим, что же такое переменная `data`:

# In[3]:


data


# In[4]:


type(data)


# ---

# #### Вопрос

# О чём данные? (Hint: https://www.kaggle.com/mehdidag/black-friday/home)

# ---

# #### Основное задание (в тесте надо будет вставлять ответы для этих пунктов):

# *Примечание:* не бойтесь гуглить и заглядывать в "Полезные ссылки" для того, чтобы выполнить какие-то задания. Возможно, на семинаре не было какого-то нужного метода, но он находится в поисковике за 2 минуты.

# **0).** Сколько всего возрастных категорий?  

# In[12]:


# Ваш код здесь
len(data.Age.unique())


# **1).** Сколько строк с мужчинами из категории города A? (речь не об уникальных ID мужчин, а о количестве строк)  

# In[17]:


# Ваш код здесь
data[(data["City_Category"] == "A")& (data["Gender"] == "M")].shape[0]


# **2).** Сколько женщин от 46 до 50, потративших (столбец Purchase) больше 20000 (условных единиц, в данном случае)?   (речь не об уникальных ID, а о количестве строк)  

# In[22]:


# Ваш код здесь
data[(data["Gender"] == "F") & (data["Age"] == "46-50") & (data["Purchase"] > 20000)].shape[0]


# **3).** Сколько NaN'ов в столбце Product_Category_3?  

# In[27]:


# Ваш код здесь
data[data["Product_Category_3"].isna() == True].shape[0]


# **4).** Какую долю (вещественное число от 0 до 1, округлить до 4-го знака) от всех покупателей составляют ВМЕСТЕ мужчины от 26 до 35 лет и женщины старше 36 лет (то есть нужно учесть несколько возрастных категорий)? (речь не об уникальных ID, а о количестве таких строк)  

# In[53]:


# Ваш код здесь

x1 = data[((data["Gender"] == "M") & (data["Age"] == "26-35")) | ((data["Gender"] == "F") & (data["Age"].isin(["36-45", "46-50", "51-55", "55+"])))].shape[0]

x2 = data.shape[0]

round(x1 / x2, 4)


# ---

# Больше про pandas можно найти по этом полезным ссылкам:

# * Официальные туториалы: http://pandas.pydata.org/pandas-docs/stable/tutorials.html

# * Статья на Хабре от [OpenDataScience сообщества](http://ods.ai/)**:** https://habr.com/company/ods/blog/322626/

# * Подробный гайд: https://media.readthedocs.org/pdf/pandasguide/latest/pandasguide.pdf

# Главное в работе с новыми библиотеками -- не бояться тыкать в разные функции, смотреть типы возвращаемых объектов и активно пользоваться Яндексом, а ещё лучше понимать всё из docstring'а (`Shift+Tab` при нахождении курсора внутри скобок функции).
