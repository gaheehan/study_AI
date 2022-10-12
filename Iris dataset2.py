#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

iris = load_iris()


# In[4]:


dir(iris) #dir() 함수로 해당 객체의 변수와 메소드 확인 directory


# In[5]:


iris.feature_names


# In[6]:


iris.target_names


# In[7]:


display(iris.target, iris.target.shape)


# In[8]:


display(iris.target[iris.target==0].shape, iris.target[iris.target==1].shape, iris.target[iris.target==2].shape)


# In[9]:


display(iris.data.shape, iris.data[:5])


# In[10]:


#훈련세트와 테스트세트로 분리

from sklearn.model_selection import train_test_split


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target)
display(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# 아이리스 데이터가 150x4, 타겟이 150x1인데 데이터를 훈련용으로 112x4, 테스트용으로38x4, 타켓을 훈련용으로 112x1, 테스트용으로 38x1 로 나눈다.


# In[14]:


display(X_train[:5], y_train[:5])


# In[15]:


#4가지 속성에 대해 산점도 그리기

import pandas as pd

iris_df = pd.DataFrame(X_train, columns=iris.feature_names)
iris_df[:5]


# In[16]:


pd.plotting.scatter_matrix(iris_df, c=y_train, s=60, alpha=0.8, figsize=[12,12])
# scatter_matrix라는 함수를 이용하기 위해서 pandas를 사용했다.
# 데이터에서 2개씩 짝지어서 6개의 산점도를 그렸다.
# 자기자신은 히스토그램
print('')


# In[17]:


"""k-NN (최근접 이웃) 예측모델 적용
k-NN 모델은 가장 가까이에 있는 k 갯수의 이웃 점들을 기준으로 예측하는 머신러닝 모델이다.
모델은 훈련세트로 훈련을 시키므로, X_train 과 y_train 을 활용한다.
아래 코드와 같이, 모델을 정의하고 fit() 함수를 호출하는 두 줄로 모델 훈련은 끝난다.
기하적인 특성"""


# In[22]:


from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=1) #기본값은 5
model.fit(X_train, y_train) #fit데이터 줄테니 모델 만들어라 


# In[23]:


model.predict([[6,3,4,1.5]]) # 샘플이 하나라도 2차원 어레이를 넘겨야 한다
# predict 예측.


# In[24]:


score = model.score(X_test, y_test) #score을 이용해 평가
print(score)


# In[25]:


pred_y=model.predict(X_test)
pred_y==y_test


# In[26]:


(model.predict(X_test)==y_test).mean()


# In[28]:


'''전체코드'''

import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target)

model = KNeighborsClassifier(n_neighbors=1)  # 섬이 생길 수 있다.
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print(score)


# In[29]:


import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print(score)


# In[38]:


import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target)

model = KNeighborsClassifier(n_neighbors=10)
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print(score)


# In[39]:


'''n_neighbors 의 숫자가 커질수록 직선 경향이 커진다.'''


# In[54]:


#SVC model 특징: 매끄러운 곡선(다차원:곡면)

import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target)

model = SVC(C=1.0, gamma=0.1)

model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print(score)

