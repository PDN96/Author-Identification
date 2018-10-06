import numpy as np
import math
import matplotlib.pyplot as plt

#############################################################
#                      Task 1                               #
#############################################################

def freq_calc(s, train_data):
    dict = {}
    for i in range(s):
        dict[train_data[i]] = dict.get(train_data[i], 0.0) + 1.0
    return dict

def model_pd(alpha, total_alpha, training_size, training_dict):
    dict_pd = {}
    for k in range(0,K):
        m_k = training_dict.get(distinct[k], 0.01)
        dict_pd[distinct[k]] = (m_k + alpha) / (training_size + total_alpha)
    return dict_pd

def model_map(alpha, total_alpha, training_size, training_dict):
    dict_mp = {}
    for k in range(0,K):
        m_k = training_dict.get(distinct[k], 0.01)
        dict_mp[distinct[k] ]= (m_k + alpha - 1) / (training_size + total_alpha - K)
    return dict_mp

def model_ml(training_size, training_dict):
    dict_ml = {}
    for k in range(0, K):
        m_k = training_dict.get(distinct[k], 0.01)
        dict_ml[distinct[k]] = m_k / training_size
    return dict_ml

def perplexity(dict, training_size, data):
    perplexity = 0
    for j in range(training_size):
        perplexity += np.log(dict[data[j]])
    perplexity = np.exp(perplexity * (-1.0) / training_size)
    return perplexity

train_file = "./training_data.txt"
test_file = "./test_data.txt"

fp_tr = open(train_file)
data = fp_tr.read()
train_data = data.split()
fp_tr.close()


fp_tt = open(test_file)
data = fp_tt.read()
test_data = data.split()

N = len(train_data)

full_dict = {}
for raw_word in train_data:
    if raw_word not in full_dict:
        full_dict[raw_word] = 0
    full_dict[raw_word] = full_dict[raw_word] + 1

for raw_word in test_data:
    if raw_word not in full_dict:
        full_dict[raw_word] = 1

distinct = list(full_dict.keys())
K = len(distinct)

#alpha value is given as 2
alpha = 2

#Calculating total alpha
total_alpha = 0
for i in range(0,K):
    total_alpha = total_alpha + alpha

#Different values of N are given as N/128, N/64, N/16, N/4, N
sizes = [int(N/128), int(N/64), int(N/16), int(N/4), int(N)]

full_ml_pp_test = []
full_mp_pp_test = []
full_pd_pp_test =[]

full_ml_pp_train = []
full_mp_pp_train = []
full_pd_pp_train =[]


for i in range(len(sizes)):
    training_size = sizes[i]
    training_dict = freq_calc(training_size, train_data)

# We must build 3 different models for each approach
    dict_mp = model_map(alpha, total_alpha, training_size, training_dict)
    dict_ml = model_ml(training_size, training_dict)
    dict_pd = model_pd(alpha, total_alpha, training_size, training_dict)


# Computing the perplexities for train data

    pp_ml_train = perplexity(dict_ml, training_size, train_data)
    pp_mp_train = perplexity(dict_mp, training_size, train_data)
    pp_pd_train = perplexity(dict_pd, training_size, train_data)

    full_ml_pp_train.append(pp_ml_train)
    full_mp_pp_train.append(pp_mp_train)
    full_pd_pp_train.append(pp_pd_train)

# Computing the perplexities for test data

    pp_ml_test = perplexity(dict_ml, training_size, test_data)
    pp_mp_test = perplexity(dict_mp, training_size, test_data)
    pp_pd_test = perplexity(dict_pd, training_size, test_data)

    full_ml_pp_test.append(pp_ml_test)
    full_mp_pp_test.append(pp_mp_test)
    full_pd_pp_test.append(pp_pd_test)

print("################################################")

print("Perplexities for MLE approach in increasing sizes")
print("On Test Set")
print(full_ml_pp_test)
print("On Train Set")
print(full_ml_pp_train)

print("################################################")

print("Perplexities for MAP approach in increasing sizes")
print("On Test Set")
print(full_mp_pp_test)
print("On Train Set")
print(full_mp_pp_train)

print("################################################")


print("Perplexities for PD approach in increasing sizes")
print("On Test Set")
print(full_pd_pp_test)
print("On Train Set")
print(full_pd_pp_train)

#Plots for Task 1
'''
sizing_factor = [128, 64, 16, 4, 1]
#Dividing ML values to get them to fit inside the graph
temp_ml_test = [full_ml_pp_test[i] for i in range(5)]
temp_ml_test[0] = temp_ml_test[0]/3
temp_ml_test[1] = temp_ml_test[1]/2
f = plt.figure(1)
plt.plot(sizing_factor, full_ml_pp_train,'rv--',label='ML_train')
plt.plot(sizing_factor, full_mp_pp_train,'bD-',label='MAP_train')
plt.plot(sizing_factor, full_pd_pp_train,'g*-' ,label='PD_train')
plt.legend()
plt.xlabel('Dividing Factor of N')
plt.ylabel('Perplexity')
plt.plot()

g = plt.figure(2)
plt.plot(sizing_factor, temp_ml_test,'r^--' , label='ML_test')
plt.plot(sizing_factor, full_mp_pp_test,'bx--', label='MAP_test')
plt.plot(sizing_factor, full_pd_pp_test,'g8--' , label='PD_test')
plt.legend()
plt.xlabel('Dividing Factor of N')
plt.ylabel('Perplexity')
plt.plot()

plt.show()
'''

#############################################################
#                      Task 2                               #
#############################################################

def calc_log_evidence(s, a):
    l_e = 0
    for i in range(s):
        l_e += (-1.0) * np.log(a + i)
    return l_e


# Brute force grid search for alphas
alphas =[1,2,3,4,5,6,7,8,9,10]

#Size is fixed as per given parameter
training_size = int(N/128)

full_pp_pd = []
log_evidence = []

for alpha in alphas:

    alpha_zero = K * alpha
#Computing Log of Evidence
    temp_log_evidence = calc_log_evidence(training_size, alpha_zero)

    training_dict = {}
    training_dict = freq_calc(training_size, train_data)

    pd = {}
    pd = model_pd(alpha, alpha_zero, training_size, training_dict)

# compute the perplexity on test data and log evidence on training data
    for k in range(K):
        m_k = training_dict.get(distinct[k], 0.01)
        if (m_k >= 1):
            for i in range(int(m_k)):
                temp_log_evidence += np.log(alpha + i)

    pp_pd = perplexity(pd, training_size, test_data)
    full_pp_pd.append(pp_pd)
    log_evidence.append(temp_log_evidence)

pd_test = [(int)(item) for item in full_pp_pd]
log_evidence = [(int)(item) for item in log_evidence]

print("The perplexities on test set for alphas 1 to 10 in ascending order")
print(pd_test)
print("The log evidence for alphas 1 to 10 in ascending order")
print(log_evidence)

#Plots for Task 2
'''
f = plt.figure(1)
plt.plot(alphas, pd_test,'r*-')
plt.xlabel('Value of Alpha')
plt.ylabel('Perplexity on the Test Data')
plt.plot()

g = plt.figure(2)
plt.plot(alphas, log_evidence,'b*-')
plt.xlabel('Value of Alpha')
plt.ylabel('Log of Evidence Function')
plt.plot()

plt.show()
'''

#############################################################
#                      Task 3                               #
#############################################################

full_dict = {}
training_dict = {}

test_data141 = []
test_data1400 = []
train_data121 = []

#Getting the train data from file pg121

f = open("./pg121.txt.clean")
for row in f:
        temp_words = row.strip().split(' ')
        for word in temp_words:
            if (len(word.strip()) < 1):
                continue
            full_dict[word] = full_dict.get(word, 0) + 1
            train_data121.append(word)
            training_dict[word] = training_dict.get(word, 0) + 1
f.close()

#Reading the file pg141

f = open("./pg141.txt.clean")
for row in f:
        temp_words = row.strip().split(' ')
        for word in temp_words:
            if (len(word.strip()) < 1):
                continue
            full_dict[word] = full_dict.get(word, 0) + 1
            test_data141.append(word)

f.close()

#Reading the file pg1400

f = open("./pg1400.txt.clean")
for row in f:
    temp_words = row.strip().split(' ')
    for word in temp_words:
            if (len(word.strip()) < 1):
                continue

            full_dict[word] = full_dict.get(word, 0) + 1
            test_data1400.append(word)

f.close()

training_size = len(train_data121)
alpha = 2.0

distinct_words = list(full_dict.keys())
K = len(distinct_words)
total_alpha = K * alpha

pd = {}
for k in range(0, K):
    m_k = training_dict.get(distinct_words[k], 0.01)
    pd[distinct_words[k]] = (m_k + alpha) / (training_size + total_alpha)

pp_141 = 0.0
pp_1400 = 0.0

for j in range(len(test_data141)):
    pp_141 += math.log(pd[test_data141[j]])

for j in range(len(test_data1400)):
    pp_1400 += math.log(pd[test_data1400[j]])

final_pp_141 = np.exp((-1.0) / len(test_data141) * pp_141)
final_pp_1400 = np.exp((-1.0) / len(test_data1400) * pp_1400 )

print("Perplexity of PD on pg141.txt.clean")
print(final_pp_141)

print("Perplexity of PD on pg1400.txt.clean")
print(final_pp_1400)

if final_pp_141<final_pp_1400:
    print("\nThus pg141 is more likely to be written by author of pg121")
else:
    print("\nThus pg1400 is more likely to be written by author of pg121")
