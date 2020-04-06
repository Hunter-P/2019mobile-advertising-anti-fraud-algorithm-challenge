# -*- encoding:utf-8 -*-
# author: Jiaxin Peng

import pandas as pd
from sklearn.externals import joblib
import numpy as np
import time

# path = 'data/'
#
# train1 = pd.read_table(path+"round1_iflyad_anticheat_traindata.txt")
# print(len(train1))
# train2 = pd.read_table(path+"round2_iflyad_anticheat_traindata.txt")
# train2.drop(index=[320443, 4456598, 1080823], inplace=True)
# print(len(train2))
# print(len(train1)+len(train2))
# train = pd.concat([train1, train2])
# del train1, train2
# testA = pd.read_table(path+"round2_iflyad_anticheat_testdata_feature_A.txt")
# testB = pd.read_table(path+"round2_iflyad_anticheat_testdata_feature_B.txt")
# test = pd.concat([testA, testB])
# del testA, testB
# data = train.append(test).reset_index(drop=True)


def remove_same_ip_from_train():
    train1 = pd.read_table("data/round1_iflyad_anticheat_traindata.txt")
    train2 = pd.read_table("data/round2_iflyad_anticheat_traindata.txt")
    train = pd.concat([train1, train2], axis=0).reset_index(drop=True)
    test_A = pd.read_table('data/round2_iflyad_anticheat_testdata_feature_A.txt')
    test_B = pd.read_table("data/round2_iflyad_anticheat_testdata_feature_B.txt")
    test_ip = set(pd.concat((test_A, test_B), axis=0).ip)

    black_l = joblib.load('data/black_list')
    res01 = black_l[0] | black_l[1]
    res01 = res01 & set(test_ip)
    print(len(res01))
    remove_index = []
    s = time.time()
    ip = train['ip']
    for i in range(train.shape[0]):
        if ip[i] in res01:
            remove_index.append(i)
    train = train.drop(remove_index).reset_index(drop=True)
    print(train.shape)
    print('time:', time.time() - s)
    return train


remove_same_ip_from_train()



def reverse_same_ip_from_train():
    train1 = pd.read_table("data/round1_iflyad_anticheat_traindata.txt")
    train2 = pd.read_table("data/round2_iflyad_anticheat_traindata.txt")
    train = pd.concat([train1, train2], axis=0)
    test_A = pd.read_table('data/round2_iflyad_anticheat_testdata_feature_A.txt')
    test_B = pd.read_table("data/round2_iflyad_anticheat_testdata_feature_B.txt")
    test_ip = set(pd.concat((test_A, test_B), axis=0).ip)

    black_l = joblib.load('data/black_list')
    res0, res1 = black_l[0], black_l[1]
    res0 = res0 & set(test_ip)
    res1 = res1 & set(test_ip)
    print(len(res0), len(res1))

    ip = list(train.ip)
    label = list(train.label)
    s = time.time()
    new_label = []
    for i in range(len(ip)):
        if ip[i] in res1:
            new_label.append(1)
        elif ip[i] in res0:
            new_label.append(0)
        else:
            new_label.append(label[i])
    print('time:', time.time() - s)
    return new_label



def pkgname_feature(data, p_data):
    """
    pkgname	string	包名(MD5加密)
    """

    # p_data['nan_pkgname'] = data['pkgname'].apply(lambda x: np.nan if x == 'empty' else x).astype('str')  # 0.549
    # p_data['ifnan_pkgname'] = data['pkgname'].isnull().astype('str')  # 0.0
    # p_data['pkgname_ver'] = data['ver'].astype('str') + data['pkgname'].astype('str')  # 0.34
    p_data['pkgname_mediashowid'] = data['mediashowid'].astype('str') + data['pkgname'].astype('str')  # 19.17
    p_data['pkgname_make'] = data['make'].astype('str') + data['pkgname'].astype('str')  # 8.9
    p_data['pkgname_osv'] = data['osv'].astype('str') + data['pkgname'].astype('str')
    p_data['pkgname_model'] = data['model'].astype('str') + data['pkgname'].astype('str')
    p_data['pkgname_apptype'] = data['apptype'].astype('str') + data['pkgname'].astype('str')
    p_data['log_count_pkgname_mediashowid'] = np.log10(p_data.groupby(['pkgname_mediashowid'])['sid'].transform('count'))
    p_data['pkgname_adunitshowid'] = data['adunitshowid'].astype('str') + data['pkgname'].astype('str')  # 1.56
    p_data['adunitshowid_mediashowid'] = data['adunitshowid'].astype('str') + data['mediashowid'].astype('str')  # 1.85
    return p_data


def certain_province(data, p_data):
    try:
        areas = pd.read_csv('data/2016.11.areas.csv', encoding='gbk')
    except:
        areas = pd.read_csv('/data1/pengjiaxin/round2/2016.11.areas.csv', encoding='gbk')
    areas['city'] = areas.areaname
    information = areas.iloc[:, [4, 5, 8, 9]]
    print(information)
    p_data['city'] = data.city
    p_data = p_data.merge(information, on='city', how='left')
    p_data.drop('city', axis=1, inplace=True)
    p_data['sort'] = p_data['sort'].astype('str')
    return p_data


def ver_feature(data, p_data):
    # p_data['pkgname_ver0'] = data['ver_0'].astype('str') + data['pkgname'].astype('str')  # 0.818


    return p_data


# 黑名单制度
def black_list(ip=None, label=None):
    print(len(ip), len(label))
    # train1 = pd.read_table("data/round1_iflyad_anticheat_traindata.txt")
    # train2 = pd.read_table("data/round2_iflyad_anticheat_traindata.txt")
    # train = pd.concat([train1, train2])
    # res1 = set(train['imeimd5'][train.label == 1])
    # res0 = set(train['imeimd5'][train.label == 0])
    # res11 = res1-res0
    # res00 = res0-res1
    # joblib.dump([res00, res11], 'data/black_list_imeimd5')
    black_l = joblib.load('data/black_list_imeimd5')
    res0, res1 = black_l[0], black_l[1]
    res0 = res0 & set(ip)
    res1 = res1 & set(ip)
    # print((res0 + res1)[:10])
    # print(([0]*len(res0) + [1]*len(res1))[:10])
    # res_d = dict([(res0 + res1), ([0]*len(res0) + [1]*len(res1))])
    # for i in range(len(ip)):
    #     try:
    #         label[i] = res_d[ip[i]]
    #     except KeyError:
    #         continue
    print(len(res0), len(res1))
    s = time.time()
    res_d = dict()
    for i in res0:
        res_d[i]=0
    for i in res1:
        res_d[i]=1
    print('time:', time.time() - s)

    def f(x):
        try:
            return res_d[x[1]]
        except KeyError:
            return x[0]

    # ip_label['label'] = ip_label.apply(lambda x: f(x), axis=1)
    # ip_label = ip_label.reset_index(drop=True)
    new_label = []
    for i in range(len(ip)):
        if ip[i] in res1:
            new_label.append(1)
        elif ip[i] in res0:
            new_label.append(0)
        else:
            new_label.append(label[i])
    print('time:', time.time() - s)
    return new_label

# ip相似性


"""
0:	learn: 0.9615426	test: 0.9647196	best: 0.9647196 (0)	total: 1.06s	remaining: 26m 26s
50:	learn: 0.9766646	test: 0.9784378	best: 0.9784378 (50)	total: 1m 6s	remaining: 31m 32s
100:	learn: 0.9781509	test: 0.9791553	best: 0.9791704 (94)	total: 2m 8s	remaining: 29m 42s
150:	learn: 0.9790505	test: 0.9794830	best: 0.9794969 (148)	total: 3m 18s	remaining: 29m 32s
200:	learn: 0.9798282	test: 0.9797306	best: 0.9797867 (190)	total: 4m 26s	remaining: 28m 41s
250:	learn: 0.9804092	test: 0.9798817	best: 0.9798961 (222)	total: 5m 31s	remaining: 27m 30s
300:	learn: 0.9809814	test: 0.9798561	best: 0.9799916 (267)	total: 6m 32s	remaining: 26m 5s
350:	learn: 0.9815028	test: 0.9797805	best: 0.9799916 (267)	total: 7m 36s	remaining: 24m 54s
400:	learn: 0.9820486	test: 0.9801191	best: 0.9801863 (396)	total: 8m 45s	remaining: 24m 1s
450:	learn: 0.9825840	test: 0.9800381	best: 0.9801863 (396)	total: 10m 48s	remaining: 25m 8s
500:	learn: 0.9830165	test: 0.9799848	best: 0.9801863 (396)	total: 13m 6s	remaining: 26m 8s
550:	learn: 0.9836819	test: 0.9800914	best: 0.9801863 (396)	total: 15m 41s	remaining: 27m
600:	learn: 0.9841460	test: 0.9800220	best: 0.9801863 (396)	total: 18m 10s	remaining: 27m 11s
650:	learn: 0.9846430	test: 0.9800321	best: 0.9801863 (396)	total: 20m 55s	remaining: 27m 17s
700:	learn: 0.9850639	test: 0.9800998	best: 0.9801863 (396)	total: 23m 33s	remaining: 26m 51s
750:	learn: 0.9855237	test: 0.9801287	best: 0.9802097 (709)	total: 26m 8s	remaining: 26m 3s
800:	learn: 0.9859316	test: 0.9801969	best: 0.9802508 (771)	total: 28m 25s	remaining: 24m 50s
850:	learn: 0.9864226	test: 0.9802663	best: 0.9803068 (826)	total: 29m 32s	remaining: 22m 33s
900:	learn: 0.9868478	test: 0.9803335	best: 0.9803351 (874)	total: 30m 39s	remaining: 20m 24s
950:	learn: 0.9872420	test: 0.9803175	best: 0.9803874 (940)	total: 31m 46s	remaining: 18m 21s
1000:	learn: 0.9876056	test: 0.9803313	best: 0.9803996 (964)	total: 32m 54s	remaining: 16m 25s
1050:	learn: 0.9880303	test: 0.9804391	best: 0.9804396 (1047)	total: 34m 4s	remaining: 14m 34s
1100:	learn: 0.9884056	test: 0.9803319	best: 0.9804396 (1047)	total: 35m 11s	remaining: 12m 45s
1150:	learn: 0.9888810	test: 0.9802252	best: 0.9804396 (1047)	total: 36m 16s	remaining: 11m
1200:	learn: 0.9892684	test: 0.9803313	best: 0.9804396 (1047)	total: 37m 26s	remaining: 9m 19s
1250:	learn: 0.9896017	test: 0.9800998	best: 0.9804396 (1047)	total: 38m 39s	remaining: 7m 42s
1300:	learn: 0.9899453	test: 0.9800961	best: 0.9804396 (1047)	total: 39m 53s	remaining: 6m 6s
1350:	learn: 0.9902680	test: 0.9802321	best: 0.9804396 (1047)	total: 41m 9s	remaining: 4m 32s
1400:	learn: 0.9905302	test: 0.9801243	best: 0.9804396 (1047)	total: 42m 23s	remaining: 2m 59s
1450:	learn: 0.9908672	test: 0.9800582	best: 0.9804396 (1047)	total: 43m 46s	remaining: 1m 28s
1499:	learn: 0.9911803	test: 0.9800849	best: 0.9804396 (1047)	total: 45m 2s	remaining: 0us
"""