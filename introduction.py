
# coding: utf-8

# In[1]:


True + 4


# In[2]:


list(range(3))[3]


# In[5]:


text = 'hello'

print(text[4:100])


# In[24]:


items = [('one', 'two'), ('three', 'four'), ('five', 'six'), ('string', 'a')]

sorted_items = sorted(items, key=lambda x: x[1][-1])
                      
sorted_items


# In[25]:


items = [1, 56, 74, 3, 5]

for i, item in enumerate(items):

    item += 3

print(items[1])


# In[36]:


items = (4, 7, 2, 9)

for i in items:

    items[i] += 3

print(items[1])


# In[37]:


def square(x):

    return x**2

numbers = [1, 2, 3, 4]

something = list(map(lambda x: square(x), numbers))

print(something[2])


# In[38]:


numbers = '1 2 3 4 5 6 7'.split(' ')

i = 0

while numbers[i] < 5:

    print(numbers[i], end='')

    i += 1


# In[43]:


m = [3, 67, 4, 32, 4, 3, 7]

m = list(set(m))

m[1] = 256

for item in m:

    print(item, end=',')


# In[44]:


file = open("out.txt", "w")

for i in range(5):

    file.write(str(i))

file.close()

file = open("out.txt", "w")

for i in range(5, 10):

    file.write(str(i))

file.close()


# In[50]:


x = [1, 2, 3, 4, 5]
x[::-2] = [-1, -3, -5]

print(x)

