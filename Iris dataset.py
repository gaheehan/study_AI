#!/usr/bin/env python
# coding: utf-8

# In[1]:


#iris 데이터 로드
from sklearn.datasets import load_iris


# In[2]:


iris = load_iris() #임시 변수에 데이터셋 저장
iris


# In[4]:


features = iris['data']
features[:5]


# In[6]:


feature_names = iris['feature_names']
feature_names 
#sepal: 꽃받침 petal: 꽃잎 


# In[7]:


labels = iris['target']
labels 
#Label 데이터 (Y)


# In[8]:


# target에 대한 클래쓰 이름을 확인
target_names = iris['target_names']
target_names 


# In[9]:


#데이터 셋을 DataFrame으로 변환
import pandas as pd


# In[10]:


df = pd.DataFrame(features, columns=feature_names)
df.head()


# In[14]:


df = pd.DataFrame(iris['data'], columns=iris['feature_names'])
df.head()


# In[15]:


#target 데이터를 새로운 컬럼을 만들어 추가
df['target'] = iris['target']


# In[16]:


df.head()


# In[17]:


#로드한 DataFrame 시각화

import matplotlib.pyplot as plt
import seaborn as sns


# In[18]:


#Sepal (꽃받침)의 길이 넓이에 따른 꽃의 종류가 어떻게 다르게 나오는가

plt.figure(figsize=(10,7))
sns.scatterplot(df.iloc[:,0], df.iloc[:,1], hue=df['target'], palette='muted')
plt.title('Sepal', fontsize=17)
plt.show()


# In[19]:


#데이터 분할 (train_test_split)
df.head()


# In[20]:


from sklearn.model_selection import train_test_split


# In[21]:


#feature(x) 데이터 분할

x = df.iloc[:, :4]
x.head()


# In[22]:


#label(y) 데이터 분할

y = df['target']
y.head()


# In[23]:


"""주요 hyperparameter

test_size: validation set에 할당할 비율 (20% -> 0.2), 기본값 0.25
stratify: 분할된 샘플의 class 갯수가 동일한 비율로 유지
random_state: 랜덤 시드값
shuffle: 셔플 옵션, 기본값 True"""


# In[24]:


x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y,test_size = 0.2, random_state=30)


# In[25]:


#원본 x의 shape
x.shape


# In[26]:


#분할된 x의 shape
x_train.shape, x_test.shape


# In[27]:


#원본 y의 shape
y.shape


# In[28]:


#분할된 y의 shape
y_train.shape, y_test.shape


# In[31]:


#train(학습)데이터 세트로 학습수행하기
# 의사결정나무 분류기
# 데이터에 있는 규칙을 학습을 통해 자동으로 찾아내 트리 기반의 분류 규칙을 만드는 알고리즘

from sklearn.tree import DecisionTreeClassifier


# In[32]:


# DecisionTreeClassifier 객체 생성하기
dt_clf = DecisionTreeClassifier(random_state=42)

# 학습시키기
# fit() 함수는 학습기계에 지정된 data를 학습시킨다. 
dt_clf.fit(x_train, y_train)


# In[33]:


DecisionTreeClassifier(random_state=42)


# In[34]:


#test(테스트)데이트 세트로 예측(predict)수행하기
# 학습이 완료된 DecisionTreeClassfier 객체에서 test 데이터 세트로 예측 수행한다.
# predict() 함수는 주어진 데이터세트로 결과를 예측한다.

pred = dt_clf.predict(x_test)


# In[36]:


#예측 정확도 평가하기
from sklearn.metrics import accuracy_score #정확도 계산하기 함수

#소수점 4자리실수로 format쓰기
#예측
print("예측 정확도는 바로!!:{0: .4f}".format(accuracy_score(y_test,pred)))

