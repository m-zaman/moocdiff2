
# coding: utf-8

# In[24]:

#This file trains an RNN on student actions starting with a student action log
import csv
import json
import pandas
#from datetime import datetime
import numpy as np
import datetime
import os
from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.wrappers import TimeDistributed
import csv
#import mooc_constants
import pandas as pd
print("beginning")

#Step 1: Specify the mooc log file.
log_file_name = 'data/DelftX_AE1110x_2T2015-events.log'
sorted_file = 'ORDERED_DelftX_AE1110x_2T2015-events.log'
#Step 2: Sort by time, in case it is not already sorted.
def generate_ordered_event_copy(event_log_file_name):
    """
    Takes in an event log file, with one action per row, orders the actions by time.
    """
    output_name = "ORDERED_" + event_log_file_name.split('/')[-1]
    
    try:
        os.remove(output_name)
    except OSError:
        pass
 
    all_data_paired_with_time = []
    with open(event_log_file_name) as data_file:
        for line in data_file.readlines():
            try:
                data = json.loads(line)
            except:
                print(line)
                continue
            time_element = data['time']
            if '.' in time_element:
                date_object = datetime.datetime.strptime(time_element[:-6], '%Y-%m-%dT%H:%M:%S.%f')
            else:
                date_object = datetime.datetime.strptime(time_element[:-6], '%Y-%m-%dT%H:%M:%S')
            all_data_paired_with_time.append((line, date_object))
    print('sorting by time ...')
    s = sorted(all_data_paired_with_time, key=lambda p: p[1])
    to_output = [pair[0] for pair in s]
    #return to_output
    print("dumping json to",output_name)
    with open(output_name, mode='w') as f:
        for line in to_output:
            f.write(line)
    return output_name

# print("ordering")
# ordered_log = generate_ordered_event_copy(log_file_name)

#Step 3: Preprocess to only grab rows we are interested in. For the purpose of this example, we only want actions related to which page the student is at. Thus, we exclude events such as quiz taking, video viewing, etc.
def generate_courseware_and_seq_events(log_file, earliest_time = datetime.datetime.min, latest_time = datetime.datetime.max, require_problem_check = False, bug_test = False):
    """
    log_file is the name of a sorted log of student actions where each row is a json object
    9/13 update: produces (pageview index, time) pairs in list
    """
    user_to_pageview_and_time_pairs = {}
    user_to_all_json = {}
    with open(log_file) as data_file:
        seq_events = ['seq_next', 'seq_prev', 'seq_goto']
        special_events = ['play_video', 'pause_video', 'stop_video', 'page_close', 'edx.done.toggled', 'problem_show', 'problem_save']
        users_that_have_a_problem_check = set()
        x=[]
        for line in data_file.readlines():
            data = json.loads(line)
            user = data['username']
            if user not in user_to_pageview_and_time_pairs:
                user_to_pageview_and_time_pairs[user] = []
                if bug_test:
                    user_to_all_json[user] = []
            if bug_test:
                user_to_all_json[user].append(data)
            t = datetime.datetime.strptime(data['time'][:-6], '%Y-%m-%dT%H:%M:%S.%f' if '.' in data['time'][:-6] else '%Y-%m-%dT%H:%M:%S')
            if not earliest_time <= t <= latest_time:
                continue
            et = data['event_type']
            #name = data['name']
#             if et == 'problem_check_fail':
#                 print(data)
            if et == 'problem_check':
                users_that_have_a_problem_check.add(user)
                continue
            if et in seq_events:
                event = json.loads(data['event'])
                action = str(event['new']) + '_' + str(et) + '_' + event['id'].split('/')[-1]
            elif et in special_events:
                x.append(et)
                action = et
            elif et[0] == '/' and 'courseware' in et and 'data:image' not in et and '.css' not in et:
                action = et
            else:
                continue
            user_to_pageview_and_time_pairs[user].append((action, t))
    print("hello")
    print(x[:20])
    filtered_user_to_pageviews = user_to_pageview_and_time_pairs
    #filtered_user_to_pageviews = {user: pairs for user, pairs in user_to_pageview_and_time_pairs.items() if user in users_that_have_a_problem_check}
    if bug_test:
        return filtered_user_to_pageviews, user_to_all_json
    return filtered_user_to_pageviews

# #Step 4: Convert output from step 3 into a URL-esque representation of where the student is at.
# course_axis = pandas.read_csv('data/axis_BerkeleyX_Stat2.1x_2013_Spring.csv')
# print('finished axis')

def convert_to_verticals(user_and_action_dict, course_axis, drop_chapter_events = False):
    print("drop_chapter_events is set to", drop_chapter_events)
    chap_drops = 0
    seq_events = ['seq_next', 'seq_prev', 'seq_goto']
    seq_counts = [0, 0, 0]
    every_category = [0, 0, 0, 0]
    prev_next_conversions = [0, 0]
    only_sequentials = course_axis[course_axis.category == 'sequential']
    all_paths = list(course_axis.path)
    sequential_paths = list(only_sequentials.path)
    ordered_vertical_paths = list(course_axis[course_axis.category == 'vertical'].path)
    sequential_to_chap = {}
    special_gotos = set()
    for path in sequential_paths:
        seq = path.split('/')[-1]
        chap = path.split('/')[-2]
        sequential_to_chap[seq] = chap
#    chapter_set = set()
    only_chapters = course_axis[course_axis.category == 'chapter']
    chapter_set = set([elem[1:] for elem in list(only_chapters.path)])
    print(chapter_set)
    def construct_vertical(sequential, chapter, vertical=-1):
        """
        .
        """
        return '/' + chapter + '/' + sequential + '/' + str(vertical)
    user_to_pageviews = {}
    x = []
    special_events = ['problem_check_fail', 'play_video', 'pause_video', 'stop_video', 'page_close', 'edx.done.toggled', 'problem_show', 'problem_save']
    for userid, pairlist in user_and_action_dict.items():
        chapter_location = {} #key is chapter, value is [sequential, vertical]
        sequential_location = {}
        for chapter in chapter_set:
            for p in sequential_paths:
                if chapter in p:
                    chapter_location[chapter] = [p.split('/')[-1], 1]
                    break
        for sequential in sequential_to_chap:
            sequential_location[sequential] = 1
        new_actions = []
        new_times = []
        for pair in pairlist:
#            print(pair)
#            print(userid)
            action = pair[0]
            time = pair[1]
#            if action == '/courses/BerkeleyX/Stat_2.1x/1T2014/courseware/':
                #print('skipping', action)
#                continue
            if action in special_events:
                x.append(action)
                new_action = action
                new_actions.append(new_action)
                new_times.append(time)
                continue
            is_seq = False
            for event in seq_events:
                if event in action:
                    is_seq = True
#                    print(event)
                    break
            if is_seq:
#                print(action)
                split = action.split('_')
                new_vertical = int(split[0])
                split = action.split('@')
                seq_id = split[-1]
#                print("LOOKING AT VERTICAL:", new_vertical, "WITH SEQUENTIAL ID:", seq_id)
                if seq_id not in sequential_to_chap:
#                    print("skipping", seq_id)
                    continue
                chap_id = sequential_to_chap[seq_id]
                new_action = construct_vertical(seq_id, chap_id, new_vertical)
                test_string = new_action
                if test_string not in all_paths:
#                    print(test_string)
                    if event == 'seq_prev':
                        corresponding_vertical_index = ordered_vertical_paths.index(construct_vertical(seq_id,chap_id,new_vertical+1))
                        new_action = ordered_vertical_paths[corresponding_vertical_index-1]
                        split = new_action.split('/')
                        new_vertical = split[-1]
                        seq_id = split[2]
                        chap_id = split[1]
                        prev_next_conversions[0]+=1
                    elif event == 'seq_next':
#                        if new_vertical == 7 and '4555126bb263441a99fa8eea3771801c' in action:
#                            print("skipping new element...")
#                            print(action)
#                            continue
                        corresponding_vertical_index = ordered_vertical_paths.index(construct_vertical(seq_id,chap_id,new_vertical-1))
                        new_action = ordered_vertical_paths[corresponding_vertical_index+1]
                        split = new_action.split('/')
                        new_vertical = split[-1]
                        seq_id = split[2]
                        chap_id = split[1]
                        prev_next_conversions[1]+=1
                    else:
                        special_gotos.add(action)
                        continue
                sequential_location[seq_id] = new_vertical
                if event == 'seq_prev':
                    seq_counts[0] += 1
                elif event == 'seq_next':
                    seq_counts[1] += 1
                elif event == 'seq_goto':
                    seq_counts[2] += 1
                else:
                    raise Exception()
                every_category[0] += 1
            else:
                split = action.split('/')
#                if 'courseware' in action:
#                    print(split)
        #        print(action)
                last_elem = split[-1]
                if len(last_elem) == 1 or len(last_elem) == 2:
         #           print(action)
                    #already has a direct vertical
                    seq_id = split[-2]
                    if seq_id not in sequential_to_chap:
                        continue
                    chap_id = split[-3]
                    try:
                        new_vertical = int(last_elem)
                    except:
                        new_vertical = 999999
#                        print("skipping...", action)
#                        continue
                    new_action = construct_vertical(seq_id, chap_id, new_vertical)
                    test_string = new_action
                    if test_string not in all_paths:
                        print("Nonsense vertical: ", test_string, 1)
                        #new_vertical = 1 #resolve nonsense direct vertical to vertical 1
                        #new_action = construct_vertical(seq_id, chap_id, new_vertical)
                        continue
                    sequential_location[seq_id] = new_vertical
                    every_category[1] += 1
                elif split[-3] == 'courseware':
                    #is a chapter event with no related sequential
                    if drop_chapter_events:
                        print("skipping chapter event...", action)
                        chap_drops += 1
                        continue
                    chap_id = split[-2]
                    if chap_id not in chapter_location:
#                        print("couldn't find this chapter in course axis:", action)
                        continue
                    seq_id = chapter_location[chap_id][0]
                    new_vertical = int(chapter_location[chap_id][1])
                    new_action = construct_vertical(seq_id, chap_id, int(new_vertical))
                    test_string = new_action
                    if test_string not in all_paths:
                        print("Nonsense vertical: ", test_string, 2)
                        new_vertical = 1 #resolve nonsense direct vertical to vertical 1
                        new_action = construct_vertical(seq_id, chap_id, new_vertical)

                    every_category[2] += 1
                    #is a chapter event with no related sequential
                else:
                    #is a sequential event with no vertical
                    seq_id = split[-2]
                    chap_id = split[-3]
                    if chap_id not in chapter_location:
#                        print("couldn't find this chapter in course axis:", action)
                        continue
                    if seq_id not in sequential_location:
                        #print("skipping", seq_id)
                        #print(action)
#                        print("skipping", seq_id)
                        continue
                    new_vertical = sequential_location[seq_id]
                    new_action = construct_vertical(seq_id, chap_id, int(new_vertical))
                    test_string = new_action
                    if test_string not in all_paths:
                        print("Nonsense vertical: ", test_string, 3)
                        new_vertical = 1 #resolve nonsense direct vertical to vertical 1
                        new_action = construct_vertical(seq_id, chap_id, new_vertical)

                    every_category[3] += 1
            chapter_location[chap_id] = [seq_id, new_vertical]
            new_actions.append(new_action)
            new_times.append(time)
        user_to_pageviews[userid] = list(new_actions)
    print("hi")
    print(x[:30])
    print("number of chap drops:", chap_drops)
    print(special_gotos)
    print(seq_counts)
    print(every_category)
    print(prev_next_conversions)
    return {k:v for k, v in user_to_pageviews.items() if v}

# print("generating first actions")
# u_p = generate_courseware_and_seq_events(sorted_file)
# #u_p = generate_courseware_and_seq_events(ordered_log)
# print("resolving to verticals")
# u_to_page = convert_to_verticals(u_p, course_axis, drop_chapter_events=False)

# unique_set = set()
# for u in u_to_page:
#     val = [p[0] for p in u_to_page[u]]
#     for v in val:
#         unique_set.add(v)


# mapping = {1: 'pre_start'}
# current_index = 2
# for action in list(course_axis[course_axis.category=='vertical'].path):
#     mapping[current_index] = action
#     current_index += 1
# r_mapping = {v: k for k, v in mapping.items()}


# mapped_actions = {}
# for u in u_to_page:
#     pageviews = [p[0] for p in u_to_page[u]]
#     times = [p[1] for p in u_to_page[u]]
#     converted_pageviews = [r_mapping[elem] for elem in pageviews]
#     converted_pageviews.reverse()
#     converted_pageviews.append(r_mapping['pre_start'])
#     converted_pageviews.reverse()
#     converted_times = times
#     converted_times.reverse()
#     converted_times.append("PRESTARTTIME")
#     converted_times.reverse()
#     mapped_actions[u] = list(zip(converted_pageviews,converted_times))

# no_repeat_mapped_actions = {}
# old_lens = []
# new_lens = []
def remove_continguous_repeats(pairlist):
    """
    """
    previous_elem = False
    result_list = []
    for i in range(len(pairlist)):
        current_elem = pairlist[i]
        if previous_elem:
            if current_elem[0] == previous_elem:
                continue
            else:
                previous_elem = current_elem[0]
                result_list.append(current_elem)
        else:
            previous_elem = current_elem[0]
            result_list.append(current_elem)
    return result_list

# for u in mapped_actions:
#     mapped = mapped_actions[u]
#     old_lens.append(len(mapped))
#     no_repeat_mapped_actions[u] = remove_continguous_repeats(mapped)
#     new_lens.append(len(no_repeat_mapped_actions[u]))

# mapped_actions = no_repeat_mapped_actions
# actions = [[p[0] for p in lst] for lst in list(mapped_actions.values())]
# #times = [[p[1] for p in lst] for lst in list(mapped_actions.values())]
# userids = mapped_actions.keys()
# mappings = mapping

# window_len = 1000
# x_windows = []
# y_windows = []

# #for a in actions:
# #    if len(a) < 2:
# #        continue
# #    x_windows.append(a[:-1])
# #    y_windows.append(a[1:])

# vocab_size = len(mappings)
# #X = sequence.pad_sequences(x_windows, maxlen=window_len, padding='post', truncating='post')
# #X[0]




# In[25]:

print("ordering")
ordered_log = generate_ordered_event_copy(log_file_name)

course_axis = pandas.read_csv('data/axis_DelftX_AE1110x_2T2015.csv')
print('finished axis')


# In[26]:

print("generating first actions")
u_p = generate_courseware_and_seq_events(sorted_file)
#u_p = generate_courseware_and_seq_events(ordered_log)
print("resolving to verticals")
u_to_page = convert_to_verticals(u_p, course_axis, drop_chapter_events=False)


# In[27]:

def convertPath(axis, path):
    special_events = ['problem_check_fail', 'play_video', 'pause_video', 'stop_video', 'page_close', 'edx.done.toggled', 'problem_show', 'problem_save']
    if path in special_events:
        return path
    vertical = axis.loc[axis['path'] == path].iloc[0]['module_id'].split('/')
    if vertical[-2] == 'vertical':
        return vertical[-1]
    else:
        print('ho')


# In[28]:

for u in u_to_page:       
    u_to_page[u] = [convertPath(course_axis, i) for i in u_to_page[u]]
#x.apply(lambda y: convertPath(course_axis, y['coursePath']), axis = 1)
# convertPath(course_axis, x.iloc[0]['coursePath'])


# In[29]:

x = pandas.DataFrame(list(u_to_page.items()), columns=['user', 'coursePath'])


# In[30]:


actions = x['coursePath'].tolist()


# In[9]:

with open("studentCSV/DelftX2015.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(actions)


# In[1]:

import os
import re
import pandas as pd
import gensim, logging
import numpy as np
# from scipy import PCA

# class MySentencesModel1(object):
#     def __init__(self, dirname):
#         self.dirname = dirname
 
#     def __iter__(self):
#         for fname in os.listdir(self.dirname):
#             for line in open(os.path.join(self.dirname, fname)):
#                 yield from [sentence.split() for sentence in re.split("(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", line)]

class MySentencesModel2(object):
    def __init__(self, fname):
        self.fname = fname
 
    def __iter__(self):
        #for fname in os.listdir(self.dirname):
        for line in open(self.fname):
            yield line.split(",")
 
#sentences = MySentencesModel1('studentCSV/') # a memory-friendly iterator
sentences = MySentencesModel2('studentCSV/DelftX2015.csv') # a memory-friendly iterator
#sentences = [[1,2,4,1,3], [5,4,3,1]]
model = gensim.models.Word2Vec(sentences, min_count=1)
# model2 = gensim.models.Word2Vec(sentences2)


# In[2]:

model.scale_vocab(min_count=None, sample=None, dry_run=True, keep_raw_vocab=False, trim_rule=None, update=False)


# In[3]:

model['18fc4f8bc7a84a2dafd026ac1603f696']


# In[4]:

model.save("DelftX2014")


# In[1]:

import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import gensim, logging
model1 = gensim.models.Word2Vec.load("stat2year13model")
model = gensim.models.Word2Vec.load("DelftX2014")


# In[2]:

from os import listdir
from os.path import isfile, join
mypath = "data/BerkeleyX-Stat2.1x-2013_Spring/vertical"
stat13verticals = [f[:-4] for f in listdir(mypath)]
myotherpath = "data/DelftX-AE1110x-1T2014/vertical"
delftx14verticals = [f[:-4] for f in listdir(myotherpath)]


# In[18]:

data = {}
for vertical in stat13verticals:
    if vertical in model1:
        data[vertical] = model1[vertical]
stat13dataframe = pandas.DataFrame(data = data).T
stat13dataframe.shape


# In[4]:

datadelft = {}
for vertical in delftx14verticals:
    if vertical in model:
        datadelft[vertical] = model[vertical]
delftx14dataframe = pandas.DataFrame(data = datadelft).T


# In[5]:

totalData = stat13dataframe.append(delftx14dataframe)


# In[6]:

import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

#totalData = pandas.read_csv("totalData.csv", index_col=0)
totalData['class'] = numpy.zeros(totalData.shape[0])+4
totalData


# In[7]:

# # fix random seed for reproducibility
# seed = 7
# numpy.random.seed(seed)

# # load dataset
# dataframe = totalData
# dataset = dataframe.values
# X = dataset[:,0:100].astype(float)
# Y = dataset[:,100]


# In[8]:

# dummy_y = np_utils.to_categorical(Y).astype(int)


# In[9]:

# # define baseline model
# def baseline_model():
#     # create model
#     model = Sequential()
#     model.add(Dense(100, input_dim=100, init='normal', activation='relu'))
#     model.add(Dense(3, init='normal', activation='sigmoid'))
#     # Compile model
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     return model
# estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=200, batch_size=5, verbose=0)


# In[10]:

# # kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
# # results = cross_val_score(estimator, X, dummy_y, cv=kfold)
# # print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
# model = Sequential()
# model.add(Dense(100, input_dim=100, init='normal', activation='relu'))
# model.add(Dense(3, init='normal', activation='sigmoid'))
# # Compile model
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(X, dummy_y, nb_epoch=200, batch_size=5, verbose=0)
totalData.shape


# In[11]:

# totalData["class"]["0035f2152349446dacfaf5513d52c14d"] = 1
# totalData

import FindDiffBetweenCourses
import os

diffDic = FindDiffBetweenCourses.getDiffOnProbAndVideo()
for elem in diffDic["same"]:
    vertical = os.path.splitext(elem)[0]
    if vertical in totalData.index:
        totalData.set_value(vertical, 'class', 0)
    
for elem in diffDic["changed"]:
    vertical = os.path.splitext(elem)[0]
    if vertical in totalData.index:
        totalData.set_value(vertical, 'class', 1)
    
for elem in diffDic["deleted"]:
    vertical = os.path.splitext(elem)[0]
    if vertical in totalData.index:
        totalData.set_value(vertical, 'class', 2)

totalData


# In[12]:

totalData.to_csv("totalData.csv")


# In[21]:

import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# totalData = pandas.read_csv("totalData.csv", index_col=0)
# totalData['class'] = numpy.random.choice(range(0, 3), totalData.shape[0])

seed = 6
numpy.random.seed(seed)

# load dataset
dataframe = totalData
dataset = dataframe.values
X = dataset[57:,0:100].astype(float)
Y = dataset[57:,100]
dummy_y = np_utils.to_categorical(Y).astype(int)
dummy_y = Y

# Split-out validation dataset
validation_size = 0.30
seed = 8
X_train, X_validation, Y_train, Y_validation = cross_validation.train_test_split(X, dummy_y, test_size=validation_size, random_state=seed)

# Test options and evaluation metric
num_folds = 10
num_instances = len(X_train)
seed = 6
scoring = 'accuracy'


# In[22]:

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
	cv_results = cross_validation.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)


# In[23]:

SVM = KNeighborsClassifier()
SVM.fit(X_train, Y_train)
KNN = KNeighborsClassifier()
SVM.fit(X_train, Y_train)
predictions = SVM.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


# In[16]:

SVM.predict(X_validation)


# In[40]:

Y_validation


# In[ ]:




# In[ ]:



