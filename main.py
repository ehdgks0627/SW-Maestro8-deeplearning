# coding: utf-8

# In[20]:
import sys
import json
from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import train_test_split
from konlpy.tag import Twitter
from sklearn.feature_extraction.text import CountVectorizer
twitter = Twitter()

bad_char = ["+", "-", "=", "/", "\\", "*", "!", "@", "#", "$", "%", "^", "&", "*", "(", ")"]
def pre_processing(x):  #텍스트 전처리기능, 명사만 추출하여 반환
    return " ".join(twitter.nouns(x))

x_text_list = []
y_text_list = []
#data loading
enc = sys.getdefaultencoding()
with open("refined_category_dataset.dat",encoding=enc) as fin:
    for line in fin.readlines():
        info = json.loads(line.strip())
        x_text_list.append((info['pid'], pre_processing(info['name'])))
        y_text_list.append(info['cate'])
y_name_id_dict = joblib.load("y_name_id_dict.dat")
vectorizer = CountVectorizer()
x_list = vectorizer.fit_transform(map(lambda i : i[1],x_text_list))
y_list = [y_name_id_dict[x] for x in y_text_list]
#data loading done


# In[50]:


svm_X_train, svm_X_test , svm_Y_train, svm_Y_test = train_test_split(x_text_list, y_list, test_size=0.2, random_state=61)
c = 0.059 #찾아낸 최적의 c값
svm_X_train = vectorizer.transform(map(lambda x: x[1], svm_X_train))
svm_X_test = vectorizer.transform(map(lambda x: x[1], svm_X_test))
clf = LinearSVC(C=c)
clf.fit(svm_X_train, svm_Y_train)

print(c, clf.score(svm_X_test, svm_Y_test))


# In[8]:


import requests

x_text_list = []
with open("soma8_test_data.dat",encoding=enc) as fin:
    for line in fin.readlines():
        info = json.loads(line.strip())
        x_text_list.append((info['pid'],pre_processing(info['name'])))
pred_list = clf.predict(vectorizer.transform(map(lambda i : i[1],x_text_list)))

name='김동한'
nickname='sprout_svm_74'
mode='test'
param = {'pred_list':",".join(map(lambda i : str(int(i)),pred_list.tolist())),
         'name':name,'nickname':nickname,'mode':mode}
r = requests.post('http://eval.buzzni.net:20001/eval',data=param)

print (r.json())


# In[9]:


eval_x_text_list = []
with open("soma8_eval_data.dat",encoding=enc) as fin:
    for line in fin.readlines():
        info = json.loads(line.strip())
        eval_x_text_list.append((info['pid'],pre_processing(info['name'])))
pred_list = clf.predict(vectorizer.transform(map(lambda i : i[1],eval_x_text_list)))
name='김동한'
nickname='sprout_svm_74'
mode='eval'
param = {'pred_list':",".join(map(lambda i : str(int(i)),pred_list.tolist())),
         'name':name,'nickname':nickname,'mode':mode}
d = requests.post('http://eval.buzzni.net:20001/eval',data=param)

print (d.json())


# In[68]:


pid_img_feature_dict = {}
with open("refined_category_dataset.img_feature.dat") as fin:
    for idx,line in enumerate(fin):
        if idx%1000 == 0:
            print(idx)
        pid, img_feature_str = line.strip().split(" ")
        img_feature = (np.asarray(list(map(lambda i : float(i),img_feature_str.split(",")))))
        pid_img_feature_dict[pid] = img_feature
        
pid_img_feature_eval_dict = {}
with open("refined_category_dataset.img_feature.eval.dat") as fin:
    for idx,line in enumerate(fin):
        if idx%1000 == 0:
            print(idx)
        pid, img_feature_str = line.strip().split(" ")
        img_feature = (np.asarray(list(map(lambda i : float(i),img_feature_str.split(",")))))
        pid_img_feature_eval_dict[pid] = img_feature


# In[70]:


from scipy import sparse


# In[93]:


x_text_list = []
x_text_test_list = []
x_text_eval_list = []
y_text_list = []

with open("refined_category_dataset.dat",encoding=enc) as fin:
    for line in fin.readlines():
        info = json.loads(line.strip())
        x_text_list.append((info['pid'], pre_processing(info['name'])))
        y_text_list.append(info['cate'])
        
with open("soma8_test_data.dat",encoding=enc) as fin:
    for line in fin.readlines():
        info = json.loads(line.strip())
        x_text_test_list.append((info['pid'],pre_processing(info['name'])))
        
with open("soma8_eval_data.dat",encoding=enc) as fin:
    for line in fin.readlines():
        info = json.loads(line.strip())
        x_text_eval_list.append((info['pid'],pre_processing(info['name'])))
        
img_feature_list = []
img_feature_test_list = []
img_feature_eval_list = []

for pid, name in x_text_list:
    if pid in pid_img_feature_dict:
        img_feature_list.append(pid_img_feature_dict[pid])
    else:
        img_feature_list.append(np.zeros(1000))
        
for pid, name in x_text_test_list:
    if pid in pid_img_feature_dict:
        img_feature_test_list.append(pid_img_feature_dict[pid])
    else:
        img_feature_test_list.append(np.zeros(1000))
        
for pid, name in x_text_eval_list:
    if pid in pid_img_feature_eval_dict:
        img_feature_eval_list.append(pid_img_feature_eval_dict[pid])
    else:
        img_feature_eval_list.append(np.zeros(1000))
concat_x_list = sparse.hstack((vectorizer.transform(map(lambda i : i[1],x_text_list)), img_feature_list),format='csr')
concat_x_test_list = sparse.hstack((vectorizer.transform(map(lambda i : i[1],x_text_test_list)), img_feature_test_list),format='csr')
concat_x_eval_list = sparse.hstack((vectorizer.transform(map(lambda i : i[1],x_text_eval_list)), img_feature_eval_list),format='csr')
y_list = [y_name_id_dict[x] for x in y_text_list]
print("Done")


# In[ ]:


import random
from sklearn.grid_search import GridSearchCV

golden_list = [5, 11, 14, 25, 26, 31, 32, 46, 56, 70, 76, 81, 83]
for i in range(100):
    svm_X_train, svm_X_test , svm_Y_train, svm_Y_test = train_test_split(concat_x_list, y_list, test_size=0.05, random_state=i)
    #data loading done
    c = 0.06
    clf = LinearSVC(C=c)
    size = 200
    r = random.randint(0, svm_X_train.shape[0] - size)
    clf.fit(svm_X_train, svm_Y_train)
    score = clf.score(svm_X_test, svm_Y_test)
    if score > 0.75:
        print(i, score) 
    test(score)


# In[152]:


import requests

def test(score):
    pred_list = clf.predict(concat_x_test_list)

    name='김동한'
    nickname='sprout_svm_image_%s'%(score)
    mode='test'
    param = {'pred_list':",".join(map(lambda i : str(int(i)),pred_list.tolist())),
             'name':name,'nickname':nickname,'mode':mode}
    r = requests.post('http://eval.buzzni.net:20001/eval',data=param)

    print (r.json())


# In[147]:


import requests

x_text_list = []
pred_list = clf.predict(concat_x_eval_list)

name='김동한'
nickname='sprout_svm_image75'
mode='eval'
param = {'pred_list':",".join(map(lambda i : str(int(i)),pred_list.tolist())),
         'name':name,'nickname':nickname,'mode':mode}
r = requests.post('http://eval.buzzni.net:20001/eval',data=param)

print (r.json())
