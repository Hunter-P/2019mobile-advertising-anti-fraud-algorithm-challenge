# -*- encoding:utf-8 -*-
# author: Jiaxin Peng

import pandas as pd
from sklearn.externals import joblib
import numpy as np
import catboost
import time
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from xpinyin import Pinyin
from sklearn.metrics import f1_score
import re
from datetime import timedelta, datetime
import math
from tqdm import tqdm
import os
import gc
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class FeatureEngineering:
    def __init__(self, train_data, test_data):
        """
        训练数据和测试数据的每个样本的sid是唯一的，除去sid，训练数据有2个重复样本，测试数据没有重复样本
        :param train_data:
        :param test_data:
        """
        self.train_data = train_data
        self.test_data = test_data
        self.labels = self.train_data['label']
        self.train_data = self.train_data.drop('label', axis=1)
        self.test_data_sid = test_data['sid']
        self.data = pd.concat([self.train_data, self.test_data], axis=0)  # 训练集 1000000 rows
        print(self.data)

    def run(self):
        """
        :return: train / test / label
        """
        try:
            data = joblib.load('/data1/pengjiaxin/lgb_all_data')
        except:
            data = self.feature_to_lower(self.data)
            data = self.transfer_empty_to_nan(data)
            data = self.osv_feature(data)
            print('osv')
            data = self.fill_null(data)
            data = self.lan_feature(data)
            print('lan')
            data = self.ver_feature(data)
            data = self.nginxtime_feature(data)
            print('nginxtime')
            data = self.sid_feature(data)
            print('sid')
            data = self.h_w_ppi_feature(data)
            data = self.ip_feature(data)
            print('ip')
            data = self.model_feature(data)
            data = self.make_feature(data)
            data = self.ntt_feature(data)
            print('ntt')
            data = self.apptype_feature(data)
            data = self.dvctype_feature(data)
            print('dvctype')
            data = self.orientation_feature(data)
            data = self.province_feature(data)
            data = self.city_feature(data)
            print('city')
            data = self.pkgname_feature(data)
            print('pkgname')
            data = self.carrier_feature(data)
            data = self.m_m_l(data)
            data = self.imei(data)
            data = self.if_empty(data)
            data = self.remove_columns(data)
            data = self.combine_feature(data)
            print('combine')
            data = self.shrink_feature(data, 50)
            data = self.onehot(data)
            try:
                joblib.dump(data, "/data1/pengjiaxin/lgb_all_data")
            except:
                joblib.dump(data, "data/lgb_all_data")
            # joblib.dump(data, "/data1/pengjiaxin/lgb_all_data")
        print(data.info())

        categorical_features_indices = np.where(data.dtypes == 'object')[0]
        for c in data.columns:
            if data[c].isnull().any():
                data[c] = data[c].astype("str")
        train, test = data.iloc[:1000000, :], data.iloc[1000000:, :]

        del data
        return train, test, self.labels, categorical_features_indices

    @staticmethod
    def remove_columns(data):
        """
        去除特征
        :param data
        :return:
        """
        # 去除数据全部一样的特征
        data.drop('os', axis=1, inplace=True)
        # 去除种类特别多的特征
        # data = data.drop(['adidmd5', 'ip'], axis=1)
        # 去除缺失值特别多的特征
        data.drop(['idfamd5'], axis=1, inplace=True)
        return data

    @staticmethod
    def feature_to_lower(data):
        """
        把str类型特征都变为小写
        :param data:
        :return:
        """
        for c in data.columns:
            if str(data[c].dtype) == 'object':
                column = list(data[c])
                for i in range(len(column)):
                    try:
                        column[i] = column[i].lower()
                    except:
                        pass
                data[c] = column
        return data

    @staticmethod
    def transfer_empty_to_nan(data):
        """
        把empty变为np.nan
        :param data:
        :return:
        """
        for c in data.columns:
            if str(data[c].dtype) == 'object':
                data[c] = data[c].apply(lambda x: np.nan if x == 'empty' else x)
        return data

    @staticmethod
    def label_encoder(data):
        """
        Encode labels with value between 0 and n_classes-1.
        缺失值不管
        :param data:
        :return:
        """
        # data.fillna('-999', inplace=True)
        # le = preprocessing.LabelEncoder()
        # for c in data.columns:
        #     if str(data[c].dtype) == 'object':
        #         data[c] = le.fit_transform(data[c])

        for c in data.columns:
            if str(data[c].dtype) == 'object':
                data[c] = pd.factorize(data[c])[0]
        #         data[c].replace(-1, np.nan, inplace=True)
        return data

    @staticmethod
    def fill_null(data):
        """
        缺的不太多的特征用众数填充空值
        :param data:
        :return:
        """
        # 众数填充
        data['city'].fillna(data['city'].mode()[0], inplace=True)
        # data['make'].fillna(data['make'].mode()[0], inplace=True)
        data['osv'].fillna(data['osv'].mode()[0], inplace=True)
        data['model'].fillna(data['model'].mode()[0], inplace=True)
        data['adunitshowid'].fillna(data['adunitshowid'].mode()[0], inplace=True)
        data['mediashowid'].fillna(data['mediashowid'].mode()[0], inplace=True)
        return data

    @staticmethod
    def pkgname_feature(data):
        """
        pkgname	string	包名(MD5加密)
        :param data:
        :return:
        """
        data['pkgname'].fillna('empty', inplace=True)
        data['{}_count'.format('pkgname')] = data.groupby(['{}'.format('pkgname')])['sid'].transform('count') / 10000
        return data

    @staticmethod
    def lan_feature(data):
        """
        lan（语言）
        :param data:
        :return:
        """
        new_lan = data['lan']
        new_lan.replace('zh', "zh-cn", inplace=True)
        new_lan.replace('cn', "zh-cn", inplace=True)
        new_lan.replace('zh_cn_#hans', "zh-cn", inplace=True)
        new_lan.replace('tw', "zh-tw", inplace=True)
        new_lan.replace('zh-', "zh-cn", inplace=True)
        new_lan.replace('zh_HK_#Hant', "zh-hk", inplace=True)
        new_lan.replace('en-us', "en", inplace=True)
        new_lan.replace('en-gb', "en", inplace=True)
        # data['lan_empty'] = data['lan'].isnull().astype('str')
        data['new_lan'] = new_lan
        # data = data.drop('lan', axis=1)
        return data

    @staticmethod
    def ver_feature(data):
        """
        ver(app版本, string)
        :param data:
        :return:
        """
        # data['ver_empty'] = data['ver'].isnull().astype('str')
        data['ver_0'] = data['ver'].apply(lambda x: ''.join(re.findall('[0123456789.]', str(x))))
        data['ver_0'].replace('', np.nan, inplace=True)
        data['ver'].replace('', np.nan, inplace=True)
        # data['ver'].replace('', np.nan, inplace=True)
        ver_l = list(data['ver_0'])
        ver_1 = []
        for i in range(len(ver_l)):
            if ver_l[i] is np.nan:
                ver_1.append(np.nan)
            else:
                ver_1.append(int(ver_l[i][0]))
        return data

    @staticmethod
    def nginxtime_feature(data):
        """
        nginxtime	bigint	请求到达服务时间，单位ms
        在训练集中增加 day hour mintue
        :param data:
        :return:
        """
        data['nginxtime_datetime'] = pd.to_datetime(data['nginxtime'] / 1000, unit='s') + timedelta(hours=8)
        data['nginxtime_hour'] = data['nginxtime_datetime'].dt.hour
        data['nginxtime_day'] = data['nginxtime_datetime'].dt.day - data['nginxtime_datetime'].dt.day.min()
        data['nginxtime_min'] = data['nginxtime_datetime'].dt.minute.astype('uint8')
        data['nginxtime_weekday'] = data['nginxtime_datetime'].apply(lambda x: x.strftime("%w"))
        data['nginxtime_weekday_01'] = data['nginxtime_weekday']
        data['nginxtime_weekday_01'].replace('1', "0", inplace=True)
        data['nginxtime_weekday_01'].replace('2', "0", inplace=True)
        data['nginxtime_weekday_01'].replace('3', "0", inplace=True)
        data['nginxtime_weekday_01'].replace('4', "0", inplace=True)
        data['nginxtime_weekday_01'].replace('5', "0", inplace=True)
        data['nginxtime_weekday_01'].replace('6', "1", inplace=True)
        data['nginxtime_weekday_01'].replace('7', "1", inplace=True)
        data['hour_min'] = data['nginxtime_hour'].astype('str') + data['nginxtime_min'].astype('str')
        data['day_hour'] = data['nginxtime_hour'].astype('str') + data['nginxtime_day'].astype('str')
        data['day_hour_min'] = data['nginxtime_min'].astype('str') + data['nginxtime_hour'].astype('str') + \
                               data['nginxtime_day'].astype('str')

        # data.drop(['nginxtime'], axis=1, inplace=True)
        return data

    @staticmethod
    def sid_feature(data):
        """
        sid(样本id/请求会话sid)
        将nginxtime到sid的时间差作为新特征
        并增加 day hour mintue特征
        :param data:
        :return:
        """
        data['sid0'] = data['sid'].apply(lambda x: x.split('-')[0])
        data['sid1'] = data['sid'].apply(lambda x: x.split('-')[1])
        data['sid2'] = data['sid'].apply(lambda x: x.split('-')[2])
        data['sid3'] = data['sid'].apply(lambda x: x.split('-')[3])
        data['sid0_count'] = data.groupby(['{}'.format('sid0')])['sid'].transform('count')/10000
        data['sid1_count'] = data.groupby(['{}'.format('sid1')])['sid'].transform('count') / 10000
        data['sid2_count'] = data.groupby(['{}'.format('sid2')])['sid'].transform('count') / 10000
        data['sid3_count'] = data.groupby(['{}'.format('sid3')])['sid'].transform('count') / 10000
        data.drop('sid1', axis=1, inplace=True)
        data.drop('sid3', axis=1, inplace=True)

        # 做一下星期特征
        data['sid'] = data['sid'].apply(lambda x: float(x.split('-')[-1]))
        data['sid_datetime'] = pd.to_datetime(data['sid'] / 1000, unit='s') + timedelta(hours=8)
        data['sid_hour'] = data['sid_datetime'].dt.hour
        data['sid_day'] = data['sid_datetime'].dt.day - data['sid_datetime'].dt.day.min()
        data['sid_minute'] = data['sid_datetime'].dt.minute.astype('uint8')
        data['nginxtime-sid_time'] = data['nginxtime'] - data['sid']  # 请求会话时间 与 请求到达服务时间的差
        # data.drop(['sid'], axis=1, inplace=True)
        data.drop(['nginxtime'], axis=1, inplace=True)
        return data

    @staticmethod
    def h_w_ppi_feature(data):
        """
        处理h，w/高，宽特征
        将h*w作为新特征
        :param data:
        :return:
        """
        # 使用make填充 h w ppi值为0.0的数据
        data['h'].replace(0.0, np.nan, inplace=True)
        data['w'].replace(0.0, np.nan, inplace=True)
        # all_data['ppi'].replace(0.0, np.nan, inplace=True)
        # cols = ['h', 'w', 'ppi']
        cols = ['h', 'w']
        gp_col = 'make'
        for col in tqdm(cols):
            na_series = data[col].isna()
            names = list(data.loc[na_series, gp_col])
            # 使用均值 或者众数进行填充缺失值
            # df_fill = all_data.groupby(gp_col)[col].mean()
            df_fill = data.groupby(gp_col)[col].agg(lambda x: stats.mode(x)[0][0])
            t = df_fill.loc[names]
            t.index = data.loc[na_series, col].index
            # 相同的index进行赋值
            data.loc[na_series, col] = t
            data[col].fillna(0.0, inplace=True)
            del df_fill
            gc.collect()

        data['creative_dpi'] = data['w'].astype(str) + "_" + data['h'].astype(str)
        data['s'] = np.array(data['h'])*np.array(data['w'])
        data['size'] = (np.sqrt(data['h'] ** 2 + data['w'] ** 2) / 2.54) / 1000
        data['ratio'] = data['h'] / data['w']
        data['px'] = data['ppi\n'] * data['size']
        data['mj'] = data['h'] * data['w']
        data['hwp'] = data['h'] * data['w'] * data['ppi\n']

        for i in tqdm(['h', 'w', 'ratio']):
            data['{}_count'.format(i)] = data.groupby(['{}'.format(i)])['sid'].transform('count')
            data['{}_rank'.format(i)] = data['{}_count'.format(i)].rank(method='min')
        # data['h'].replace(0, np.nan, inplace=True)
        # 0的时候label=1很少 w, h特征强，某些w/h下能够较好区分label, 应该作为类型特征
        # data['w'].replace(0, np.nan, inplace=True)
        data['w'] = data['w'].astype('str')
        data['h'] = data['h'].astype('str')
        data['s'] = data['s'].astype('str')
        data['ppi'] = data['ppi\n'].astype('str')

        return data

    @staticmethod
    def ip_feature(data):
        """
        ip  ip分类：A B C
        :param data:
        :return:
        """
        ip_category = []
        for i in data['ip']:
            s = int(re.split(r'[.:]', i)[0])
            if s <= 127:
                ip_category.append('A')
            elif s <= 255:
                ip_category.append('B')
            else:
                ip_category.append('C')
        data['ip_category'] = ip_category
        # 将ip的计数作为新特征
        data['{}_count'.format('ip')] = data.groupby(['{}'.format('ip')])['sid'].transform('count')/10000
        data['{}_rank'.format('ip')] = data['{}_count'.format('ip')].rank(method='min')
        data['ip2'] = data['ip'].apply(lambda x: '.'.join(x.split('.')[0:2]))
        data['ip3'] = data['ip'].apply(lambda x: '.'.join(x.split('.')[0:3]))
        data['reqrealip2'] = data['reqrealip'].apply(lambda x: '.'.join(x.split('.')[0:2]))
        data['reqrealip3'] = data['reqrealip'].apply(lambda x: '.'.join(x.split('.')[0:3]))
        data['machine'] = 1000 * data['model'] + data['make']
        return data

    @staticmethod
    def model_feature(data):
        """
        model(机型)
        :param data:
        :return:
        """
        data['model'].replace('pacm00', "oppo r15", inplace=True)
        data['model'].replace('pbam00', "oppo a5", inplace=True)
        data['model'].replace('pbem00', "oppo r17", inplace=True)
        data['model'].replace('padm00', "oppo a3", inplace=True)
        data['model'].replace('pbbm00', "oppo a7", inplace=True)
        data['model'].replace('paam00', "oppo r15_1", inplace=True)
        data['model'].replace('pact00', "oppo r15_2", inplace=True)
        data['model'].replace('pabt00', "oppo a5_1", inplace=True)
        data['model'].replace('pbcm10', "oppo r15x", inplace=True)
        data['model_0'] = data['model'].apply(lambda x: x.split()[0])
        data['model_equal_make'] = (data['model_0'] == data['make']).astype('int')
        data['{}_count'.format('model_0')] = data.groupby(['{}'.format('model_0')])['sid'].transform('count') / 10000
        data['rank_' + 'model'] = data['model_0_count'].rank(method='min')
        return data

    @staticmethod
    def make_feature(data):
        """
        make(厂商)  将中文转为拼音
        :param data:
        :return:
        """
        # data['make_empty'] = data['make'].isnull().astype('str')

        def make_fix(x):
            """
            iphone,iPhone,Apple,APPLE>--apple
            redmi>--xiaomi
            honor>--huawei
            Best sony,Best-sony,Best_sony,BESTSONY>--best_sony
            :param x:
            :return:
            """

            nx = x.lower()
            if 'iphone' in nx or 'apple' in nx:
                return 'apple'
            if '华为' in nx or 'huawei' in nx or "荣耀" in nx:
                return 'huawei'
            if "魅族" in nx:
                return 'meizu'
            if "金立" in nx:
                return 'gionee'
            if "三星" in nx:
                return 'samsung'
            if 'xiaomi' in nx or 'redmi' in nx:
                return 'xiaomi'
            if 'oppo' in nx:
                return 'oppo'
            return nx
        data['make'] = data['make'].astype('str')
        data['make'] = data['make'].apply(make_fix)
        p = Pinyin()
        ls = list(data['make'])
        for i in range(len(ls)):
            try:
                ls[i] = p.get_pinyin(ls[i], '')
            except:
                pass
        data['make'] = ls
        return data

    @staticmethod
    def osv_feature(data):
        """
        osv (操作系统版本) 特征
        将osv的数字提取处理，然后分解为三列，不够的补0，多的截掉
        :param data:
        :return:
        """
        # data['osv_empty'] = data['osv'].isnull().astype('str')
        ls = list(data['osv'])
        for i in range(len(ls)):
            try:
                res = re.findall(r"\d+\d*", ls[i])
                if len(res) > 3:
                    ls[i] = [int(i) for i in res[:3]]
                elif len(res) == 3:
                    ls[i] = [int(i) for i in res]
                elif len(res) == 2:
                    ls[i] = [int(i) for i in res] + [0]
                elif len(res) == 1:
                    ls[i] = [int(i) for i in res] + [0, 0]
                else:
                    ls[i] = [np.nan, np.nan, np.nan]
            except:
                ls[i] = [np.nan, np.nan, np.nan]
        # new_data = data.drop('osv', axis=1)
        new_data = data
        new_osv = pd.DataFrame(data=ls, columns=['osv_1', 'osv_2', 'osv_3'], index=new_data.index)
        new_osv.fillna(new_osv.mode().iloc[0, :], inplace=True)  # 众数填充
        return pd.concat([new_data, new_osv], axis=1)

    @staticmethod
    def ntt_feature(data):
        """
        ntt	int	网络类型 0-未知, 1-有线网, 2-WIFI, 3-蜂窝网络未知, 4-2G, 5-3G, 6–4G
        :param data:
        :return:
        """
        # 处理 ntt 的数据特征 但是不删除之前的特征 将其归为新的一列数据
        data['new_ntt'] = data['ntt']
        data.new_ntt[(data.new_ntt == 0) | (data.new_ntt == 7)] = 0
        data.new_ntt[(data.new_ntt == 1) | (data.new_ntt == 2)] = 1
        data.new_ntt[data.new_ntt == 3] = 2
        data.new_ntt[(data.new_ntt >= 4) & (data.new_ntt <= 6)] = 3
        data['ntt_str'] = data['ntt'].astype('str')
        data['{}_count'.format('ntt')] = data.groupby(['{}'.format('ntt')])['sid'].transform('count') / 10000
        return data

    @staticmethod
    def dvctype_feature(data):
        """
        dvctype	int	设备类型 0 – 未知,1 – PC,2 – 手机, 3– 平板,4– 电视盒,5– 智能电视,6 – 可穿戴设备,7 – 智能家电,8 - 音箱,9 - 智能硬件
        :param data:
        :return:
        """
        data['dvctype'] = data['dvctype'].astype('str')
        data['{}_count'.format('dvctype')] = data.groupby(['{}'.format('dvctype')])['sid'].transform('count') / 10000
        return data

    @staticmethod
    def apptype_feature(data):
        """
        apptype	int	app所属分类
        :param data:
        :return:
        """
        # 将apptype的计数作为新特征
        data['{}_count'.format('apptype')] = data.groupby(['{}'.format('apptype')])['sid'].transform('count') / 10000
        data['apptype_str'] = data['apptype'].astype('str')
        return data

    @staticmethod
    def orientation_feature(data):
        """
        orientation	int	横竖屏:0竖屏，1横屏
        :param data:
        :return:
        """
        data.orientation[(data.orientation == 90) | (data.orientation == 2)] = 0
        data['orientation_str'] = data['orientation'].astype('str')
        data['{}_count'.format('orientation')] = data.groupby(['{}'.format('orientation')])['sid'].transform('count') / 10000
        return data

    @staticmethod
    def province_feature(data):
        """
        :param data:
        :return:
        """
        data['province_count'] = data.groupby(['{}'.format('province')])['sid'].transform('count') / 10000
        data['province'] = data['province'].astype('str')
        return data

    @staticmethod
    def city_feature(data):
        """
        :param data:
        :return:
        """
        data['city_count'] = data.groupby(['{}'.format('city')])['sid'].transform('count') / 10000
        city_divide = []
        for c in data['city']:
            if c.strip() in ['北京市', '上海市', '深圳市', '广州市']:
                city_divide.append('1')
            elif c.split('市')[0].strip() in ['杭州', '南京', '重庆', '青岛', '济南', '厦门', '成都', '武汉', '苏州',
                                             '长沙', '天津', '哈尔滨', '郑州', '沈阳', '西安', '宁波']:
                city_divide.append('2')
            else:
                city_divide.append('3')
        data['city_divide'] = city_divide
        return data

    @staticmethod
    def carrier_feature(data):
        """
        carrier	string	运营商 0-未知, 46000-移动, 46001-联通, 46003-电信
        :param data:
        :return:
        """
        data.carrier[data.carrier == -1] = 0
        data['carrier'] = data['carrier'].astype('str')
        data['carrier_count'] = data.groupby(['{}'.format('carrier')])['sid'].transform('count') / 10000
        return data

    @staticmethod
    def m_m_l(data):
        """
        :param data:
        :return:
        """
        for fea in ['model', 'make', 'lan']:
            data[fea] = data[fea].astype('str')
            data[fea] = data[fea].map(lambda x: x.upper())

            from urllib.parse import unquote

            def url_clean(x):
                x = unquote(x, 'utf-8').replace('%2B', ' ').replace('%20', ' ').replace('%2F', '/').replace('%3F',
                                                                                                            '?').replace(
                    '%25', '%').replace('%23', '#').replace(".", ' ').replace('??', ' '). \
                    replace('%26', ' ').replace("%3D", '=').replace('%22', '').replace('_', ' ').replace('+', ' ').replace(
                    '-',
                    ' ').replace(
                    '__', ' ').replace('  ', ' ').replace(',', ' ')

                if (x[0] == 'V') & (x[-1] == 'A'):
                    return "VIVO {}".format(x)
                elif (x[0] == 'P') & (x[-1] == '0'):
                    return "OPPO {}".format(x)
                elif (len(x) == 5) & (x[0] == 'O'):
                    return "Smartisan {}".format(x)
                elif ('AL00' in x):
                    return "HW {}".format(x)
                else:
                    return x
            data[fea] = data[fea].map(url_clean)
        return data

    @staticmethod
    def imei(data):
        data['{}_count'.format('imeimd5')] = data.groupby(['{}'.format('imeimd5')])['sid'].transform('count') / 10000
        data['imeimd5_01'] = data['imeimd5'].isnull().astype('str')
        return data

    @staticmethod
    def if_empty(data):
        """
        pkgname(26.32%)   adidmd5(25.42%) openudidmd5(91.84%)  macmd5 (41.82%)
        :param data:
        :return:
            """
        for c in ['pkgname', 'adidmd5', 'openudidmd5', 'macmd5']:
            data[c+'_empty'] = data[c].isnull().astype('str')
        return data

    @staticmethod
    def reqrealip_feature(data):
        """
        reqrealip	string	请求的http协议头携带IP，有可能是下游服务器的ip
        :param data:
        :return:
        """
        data['reqrealip_count'] = data.groupby(['{}'.format('reqrealip')])['sid'].transform('count') / 10000
        data['{}_rank'.format('reqrealip')] = data['{}_count'.format('reqrealip')].rank(method='min')

    @staticmethod
    def combine_feature(data):
        """
        adunitshowid	string	对外广告位ID（MD5加密）
        mediashowid	string	对外媒体ID（MD5加密）
        :param data:
        :return:
        """
        # 原始特征暴力组合
        cat_device = ['mediashowid', 'adunitshowid', 'model', 'h', 'imeimd5', 'osv', 'make', 'apptype']
        import itertools
        for i, (c1, c2) in enumerate(list(itertools.combinations(cat_device, 2))):
            data[c1 + '_' + c2] = data[c1].astype('str') + '-' + data[c2].astype('str')

        data['city'].fillna(data['city'].mode()[0], inplace=True)
        data['ip-province-city'] = data['ip'].astype('str') + data['province'] + data['city']
        data['ip-province'] = data['ip'] + data['province']
        data['province-city'] = data['city'] + data['province']
        data['countRatio_model_dvctype'] = data['ratio_count'].astype('str') + data['model'] + data['dvctype']
        data['countRatio_model_make'] = data['ratio_count'].astype('str') + data['model'] + data['make']
        data['countRatio_model_openudidmd5'] = data['ratio_count'].astype('str') + data['model'] + data['openudidmd5']
        data['countRatio_model_macmd5'] = data['ratio_count'].astype('str') + data['model'] + data['macmd5']
        data['countRatio_pkgname_ver'] = data['ratio_count'].astype('str') + data['pkgname'] + data['ver']
        return data

    @staticmethod
    def shrink_feature(data, size):
        f_f_impo = 'feature_importance.csv'
        fea_impo_shrink = list(pd.read_csv(f_f_impo)['feature'][:size])
        fea_impo = list(pd.read_csv(f_f_impo)['feature'])

        shrink_fea = []
        for i in data.columns:
            if i in fea_impo_shrink or i not in fea_impo:
                shrink_fea.append(i)
        return data[shrink_fea]

    @staticmethod
    def onehot(data):
        def one_hot_col(col):
            '''标签编码'''
            lbl = preprocessing.LabelEncoder()
            lbl.fit(col)
            return lbl

        object_cols = list(data.dtypes[data.dtypes == 'object'].index)  ##返回字段名为object类型的字段
        for col in object_cols:
            if col != 'sid':
                data[col] = one_hot_col(data[col].astype(str)).transform(data[col].astype(str))
            else:
                pass
        return data


class ModelConstruction:
    def __init__(self, train_data, test_data, label):
        self.train_data = train_data
        self.test_data = test_data
        self.label = label
        print(self.train_data.shape, self.test_data.shape, len(self.label))

    def run(self, categorical_features_indices):
        final_pred_prob = []
        cv_model = []
        skf = StratifiedKFold(n_splits=5, random_state=2019, shuffle=True)
        for index, (train_index, test_index) in enumerate(skf.split(self.train_data, self.label)):
            print(index)
            train_x, test_x, train_y, test_y = self.train_data.iloc[train_index], self.train_data.iloc[test_index], \
                                               self.label.iloc[train_index], self.label.iloc[test_index]
            model = LGBMClassifier(num_leaves=31, max_depth=-1, learning_rate=0.1, n_estimators=3000,
                                   min_child_samples=20, reg_alpha=1, random_state=7, silent=False, objective='binary')

            model.fit(train_x, train_y, eval_set=(test_x, test_y), early_stopping_rounds=500, verbose=50, eval_metric='F1')

            cv_model.append(model)
            y_test = model.predict_proba(self.test_data)
            final_pred_prob.append(y_test)

            # 保存
            try:
                joblib.dump(final_pred_prob, '/data1/pengjiaxin/lgb_final_pred_prob')
                joblib.dump(cv_model, '/data1/pengjiaxin/lgb_cv5_model')
            except:
                joblib.dump(final_pred_prob, 'data/lgb_final_pred_prob')
                joblib.dump(cv_model, 'data/lgb_cv5_model')

            # joblib.dump(final_pred_prob, '/data1/pengjiaxin/lgb_final_pred_prob')
            # joblib.dump(cv_model, '/data1/pengjiaxin/lgb_cv5_model')

        print('final_pred_prob:', final_pred_prob)


def test_data_predict(model, test_data, sid):
    test_data_sid = pd.DataFrame(data=sid, columns=['sid'])
    prediction = model.predict(test_data)
    test_data_sid['label'] = prediction
    print(sum(prediction))
    test_data_sid.to_csv('model/case4/test_predict3.csv')

try:
    ini_train_data = joblib.load('/data1/pengjiaxin/initial_train_data')
    ini_test_data = joblib.load('/data1/pengjiaxin/initial_test_data')
except:
    ini_train_data = joblib.load('data/initial_train_data')
    ini_test_data = joblib.load('data/initial_test_data')
fe = FeatureEngineering(ini_train_data, ini_test_data)
train_data, test_data, label, cfi = fe.run()
ModelConstruction(train_data, test_data, label).run(cfi)

# model = joblib.load('model/case4/catboost4')
# test_data_predict(model, test_data, fe.test_data_sid)


