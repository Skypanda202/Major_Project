#!/usr/bin/env python
# coding: utf-8

# # CNN :  Google Bert Augmented Resturant data

# In[12]:


# COLAB
from google.colab import files
from google.colab import drive
# SYS
import sys
# IPYNB
get_ipython().system('pip install import-ipynb')
import import_ipynb
# UTIL
import importlib.util


# In[13]:


get_ipython().system('pip install import-ipynb')


# In[13]:





# In[14]:


import pandas as pd
import re
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation , Flatten
from sklearn.preprocessing import LabelEncoder
from keras import utils as np_utils
from tensorflow.keras.layers import Dropout,Embedding , BatchNormalization
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tensorflow.keras.utils import plot_model
from tensorflow.keras import regularizers
from tensorflow.keras.regularizers import l2
import xgboost as xgb
import sklearn.metrics as metrics


# In[15]:


from google.colab import drive
drive.mount('/content/gdrive')


# ### Loading preprocessing class

# In[16]:


#https://stackoverflow.com/questions/62117483/import-module-in-google-colab-from-google-drive-python
sys.path.append('/content/gdrive/MyDrive/CS--2/')
#import module_1_xml_to_df
#from module_1_xml_to_df import convert_xml_to_DataFrame
from module_2_preprocessing import Data_Preprocessing


# ## 1. Loading the data

# In[17]:


get_ipython().system('ls -l /content/gdrive/MyDrive/CS--2/')


# In[18]:


restaurant_data = pd.read_csv("/content/gdrive/MyDrive/Code/augmented_data_restaurant_bert.csv")


# In[ ]:


restaurant_data.shape


# In[ ]:


restaurant_data.head()


# In[ ]:


restaurant_data.groupby('aspect_category').size().sort_values(ascending=False)


# In[ ]:


catagories = restaurant_data['aspect_category'].unique()
catagories , len(catagories)


# ## 2. Aspect Based Sentiment Analysis

# In[ ]:


X = restaurant_data['text']
Y = restaurant_data['aspect_category']


# ### 2.1. Splitting the data

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)


# In[ ]:


print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[ ]:


dp = Data_Preprocessing()


# In[ ]:


X_train_review = dp.preprocess_text(X_train)


# In[ ]:


X_test_review = dp.preprocess_text(X_test)


# In[ ]:


from keras.preprocessing.text import Tokenizer

vocab_size = 6000 # We set a maximum size for the vocabulary
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(X)
X_train_tokenized = pd.DataFrame(tokenizer.texts_to_matrix(X_train_review))
X_test_tokenized = pd.DataFrame(tokenizer.texts_to_matrix(X_test_review))


# In[ ]:


label_encoder = LabelEncoder()
integer_category = label_encoder.fit_transform(y_train)
encoded_y_train = np_utils.to_categorical(integer_category)

integer_category = label_encoder.fit_transform(y_test)
encoded_y_test = np_utils.to_categorical(integer_category)


# ### 2.2 CNN Aspect Based model

# In[ ]:


acbsa_model = Sequential()
acbsa_model.add(Dense(512, input_shape=(6000,), activation='relu'))
#acbsa_model.add((BatchNormalization()))
acbsa_model.add((Dense(256, activation='relu')))
acbsa_model.add((Dropout(0.3)))
acbsa_model.add((Dense(128, activation='relu')))
#acbsa_model.add((Dropout(0.1)))
#acbsa_model.add((Dense(64, activation='relu')))
acbsa_model.add(Dense(5, activation='softmax'))
#compile model
acbsa_model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])


# In[ ]:


acbsa_model.summary()


# In[ ]:


plot_model(acbsa_model, 'model_1.png',show_shapes=True)


# In[ ]:


#fit aspect classifier
history = acbsa_model.fit(X_train_tokenized , encoded_y_train , validation_data=(X_test_tokenized ,encoded_y_test) , epochs= 5, verbose=1)


# In[ ]:


predicted_cat = label_encoder.inverse_transform(acbsa_model.predict_classes(X_test_tokenized))
#print(new_polarity)


# In[ ]:


predicted_cat[0:10]


# ### 2.3 Classification report

# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test, predicted_cat ,target_names=catagories))


# ### Observation :
# This is the classification report of aspect category based sentiment model showing the precision, recall, f1_score and support values for all 5 unique catagories with accuracy 91 percent

# In[ ]:


import matplotlib.pyplot as plt
def plot_accuracy(history, miny=None):
  acc = history.history['accuracy']
  test_acc = history.history['val_accuracy']
  epochs = range(len(acc))
  plt.plot(epochs, acc)
  plt.plot(epochs, test_acc)
  if miny:
    plt.ylim(miny, 1.0)
  plt.title('accuracy')
  plt.xlabel('epoch')
  plt.figure()


# In[ ]:


plot_accuracy(history)


# ### 2.4.  Confusion matrix Representation

# In[ ]:


# code borrowed from Microsoft Malware Detection Assignment
def plot_confusion_matrix(test_y, predict_y ,labels):
    C = confusion_matrix(test_y, predict_y)      # calculation of confusion matrix
    print("Number of misclassified points ",(len(test_y)-np.trace(C))/len(test_y))    # number of misclassified points while predicting y

    A =(((C.T)/(C.sum(axis=1))).T)
    B =(C/C.sum(axis=0))

    labels = labels
    cmap=sns.light_palette("purple")
    # representing A in heatmap format
    print("-"*50, "Confusion matrix", "-"*50)
    plt.figure(figsize=(10,5))
    sns.heatmap(C, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.show()

    print("-"*50, "Precision matrix", "-"*50)
    plt.figure(figsize=(10,5))
    sns.heatmap(B, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.show()
    print("Sum of columns in precision matrix",B.sum(axis=0))

    # representing B in heatmap format
    print("-"*50, "Recall matrix" , "-"*50)
    plt.figure(figsize=(10,5))
    sns.heatmap(A, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.show()
    print("Sum of rows in precision matrix",A.sum(axis=1))


# In[ ]:


plot_confusion_matrix(y_test,  predicted_cat, catagories)


# ## 3. Polarity Based sentiment model

# In[ ]:


X = restaurant_data['text']
Y = restaurant_data['polarity']


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)


# In[ ]:


print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[ ]:


label_encoder = LabelEncoder()
integer_category = label_encoder.fit_transform(y_train)
encoded_y_train = np_utils.to_categorical(integer_category)

integer_category = label_encoder.fit_transform(y_test)
encoded_y_test = np_utils.to_categorical(integer_category)


# ### 3.1 CNN sentiment model

# In[ ]:


sentiment_model = Sequential()
sentiment_model.add(Dense(512, input_shape=(6000,), activation='relu'))
sentiment_model.add((Dense(256, activation='relu')))
sentiment_model.add((Dropout(0.3)))
sentiment_model.add((Dense(128, activation='relu')))
sentiment_model.add(Dense(4, activation='softmax'))
sentiment_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


sentiment_model.summary()


# In[ ]:


plot_model(sentiment_model, 'model_2.png',show_shapes=True)


# In[ ]:


#fit aspect classifier
history = sentiment_model.fit(X_train_tokenized , encoded_y_train , validation_data=(X_test_tokenized ,encoded_y_test) , epochs=6, verbose=1)


# In[ ]:


predicted_polarity = label_encoder.inverse_transform(sentiment_model.predict_classes(X_test_tokenized))
#print(new_polarity)


# In[ ]:


predicted_polarity[0:10]


# In[ ]:


polarity = restaurant_data["polarity"].unique()
polarity


# ### 3.2 Classification repoort

# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test, predicted_polarity ,target_names= polarity))


# In[ ]:


plot_accuracy(history)


# ### 3.3 Confusion matrix representation

# In[ ]:


plot_confusion_matrix(y_test,  predicted_polarity, polarity)


# ## 4. creating a resultant dataframe

# In[ ]:


def create_result_dataframe(pred_1,pred_2):
  # Calling DataFrame constructor on predicted outputs
  resultant_df = pd.DataFrame(list(zip(pred_1,pred_2)), columns = ["predicted_catagories" , "predicted_polarity"])
  result  = pd.crosstab(resultant_df.predicted_catagories,resultant_df.predicted_polarity ,margins = True , margins_name = "Total")
  result["Ranking"] = ( result.Total/resultant_df.shape[0]) * 5.0
  result["Negative in %"] = (result.negative/result.Total) * 100
  result["Neutral in %"] = (result.neutral/result.Total) * 100
  result["Positive in %"] = (result.positive/result.Total) * 100
  result["conflict in %"] = (result.conflict/result.Total) * 100
  del result["negative"]
  del result["neutral"]
  del result["positive"]
  del result["Total"]
  del result["conflict"]

  return result


# In[ ]:


result = create_result_dataframe(predicted_cat,predicted_polarity)


# In[ ]:


result


# ### Observation :
#    Above result shows that the Ranking and all Polarities in percentages for all aspect catagories from which we can see that anecdotes/miscellaneous have the highest ranking of 1.79 . and food catagory have highest positive polarity

# In[ ]:




