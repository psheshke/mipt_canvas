
# coding: utf-8

# <img src="https://s8.hostingkartinok.com/uploads/images/2018/08/308b49fcfbc619d629fe4604bceb67ac.jpg" width=500, height=450>
# <h3 style="text-align: center;"><b>Физтех-Школа Прикладной математики и информатики (ФПМИ) МФТИ</b></h3>

# ---

# На основе [курса по Машинному Обучению ФИВТ МФТИ](https://github.com/ml-mipt/ml-mipt) и [Открытого курса по Машинному Обучению](https://habr.com/ru/company/ods/blog/322626/).

# ---

# <h2 style="text-align: center;"><b>k Nearest Neighbor(KNN)</b></h2>

# Метод ближайших соседей (k Nearest Neighbors, или kNN) — очень популярный метод классификации, также иногда используемый в задачах регрессии. Это один из самых понятных подходов к классификации. На уровне интуиции суть метода такова: посмотри на соседей, какие преобладают, таков и ты. Формально основой метода является гипотеза компактности: если метрика расстояния между примерами введена достаточно удачно, то схожие примеры гораздо чаще лежат в одном классе, чем в разных. 

# <img src='https://hsto.org/web/68d/a45/6f0/68da456f00f8434e87628dbe7e3f54a7.png' width=600>

# 
# Для классификации каждого из объектов тестовой выборки необходимо последовательно выполнить следующие операции:
# 
# * Вычислить расстояние до каждого из объектов обучающей выборки
# * Отобрать объектов обучающей выборки, расстояние до которых минимально
# * Класс классифицируемого объекта — это класс, наиболее часто встречающийся среди $k$ ближайших соседей

# Будем работать с подвыборкой из [данных о типе лесного покрытия из репозитория UCI](http://archive.ics.uci.edu/ml/datasets/Covertype). Доступно 7 различных классов. Каждый объект описывается 54 признаками, 40 из которых являются бинарными. Описание данных доступно по ссылке, а так же в файле `covtype.info.txt`.

# ### Обработка данных

# In[1]:


import pandas as pd


# ССылка на датасет (лежит в в папке): https://drive.google.com/open?id=1-Z4NlDy11BzSwW13k8EgodRis0uRy1K6

# In[2]:


all_data = pd.read_csv('forest_dataset.csv',)
all_data.head()


# In[3]:


all_data.shape


# Выделим значения метки класса в переменную `labels`, признаковые описания в переменную `feature_matrix`. Так как данные числовые и не имеют пропусков, переведем их в `numpy`-формат с помощью метода `.values`.

# In[4]:


labels = all_data[all_data.columns[-1]].values
feature_matrix = all_data[all_data.columns[:-1]].values


# Сейчас будем работать со всеми 7 типами покрытия (данные уже находятся в переменных `feature_matrix` и `labels`, если Вы их не переопределили). Разделите выборку на обучающую и тестовую с помощью метода `train_test_split`, используйте значения параметров `test_size=0.2`, `random_state=42`. Обучите логистическую регрессию  на данном датасете.

# In[7]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[19]:


train_feature_matrix, test_feature_matrix, train_labels, test_labels = train_test_split(feature_matrix, labels, test_size=0.2, random_state=42)

# нормируйте данные по параметрам нормировки для train_feature_matrix
scaler = StandardScaler()
train_feature_matrix = scaler.fit_transform(train_feature_matrix)
test_feature_matrix = scaler.transform(test_feature_matrix)


# ### Обучение модели

# Качество классификации/регрессии методом ближайших соседей зависит от нескольких параметров:
# 
# * число соседей `n_neighbors`
# * метрика расстояния между объектами `metric`
# * веса соседей (соседи тестового примера могут входить с разными весами, например, чем дальше пример, тем с меньшим коэффициентом учитывается его "голос") `weights`
# 

# Обучите на датасете `KNeighborsClassifier` из `sklearn`.

# In[21]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

clf = KNeighborsClassifier()
clf.fit(train_feature_matrix, train_labels)
pred_labels = clf.predict(test_feature_matrix)
accuracy_score(test_labels, pred_labels)


# ### Вопрос 1:
# * Какое качество у вас получилось?

# Подбирем параметры нашей модели

# * Переберите по сетке от `1` до `10` параметр числа соседей
# 
# * Также вы попробуйте использоввать различные метрики: `['manhattan', 'euclidean']`
# 
# * Попробуйте использовать различные стратегии вычисления весов: `[‘uniform’, ‘distance’]`

# In[66]:


from sklearn.model_selection import GridSearchCV
params = {'weights': ["uniform", "distance"], 'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'metric': ["manhattan", "euclidean"]}

clf_grid = GridSearchCV(clf, params, cv=5, scoring='accuracy', n_jobs=-1)
clf_grid.fit(train_feature_matrix, train_labels)


# Выведем лучшие параметры

# In[23]:


clf_grid.best_params_


# ### Вопрос 2:
# * Какую metric следует использовать?

# In[67]:


print(clf_grid.best_params_['metric'])


# ### Вопрос 3:
# * Сколько n_neighbors следует использовать?

# In[68]:


print(clf_grid.best_params_['n_neighbors'])


# ### Вопрос 4:
# * Какой тип weights следует использовать?

# In[69]:


print(clf_grid.best_params_['weights'])


# Используя найденное оптимальное число соседей, вычислите вероятности принадлежности к классам для тестовой выборки (`.predict_proba`).

# In[43]:


optimal_clf = KNeighborsClassifier(n_neighbors = 10)
optimal_clf.fit(train_feature_matrix, train_labels)
pred_prob = optimal_clf.predict_proba(test_feature_matrix)


# In[50]:


import matplotlib.pyplot as plt
import numpy as np

unique, freq = np.unique(test_labels, return_counts=True)
freq = list(map(lambda x: x / len(test_labels),freq))

pred_freq = pred_prob.mean(axis=0)
plt.figure(figsize=(10, 8))
plt.bar(range(1, 8), pred_freq, width=0.4, align="edge", label='prediction')
plt.bar(range(1, 8), freq, width=-0.4, align="edge", label='real')
plt.legend()
plt.show()


# ### Вопрос 5:
# * Какая прогнозируемая вероятность pred_freq класса под номером 3(до 2 знаков после запятой)?

# In[56]:


round(pred_freq[2], 2)

