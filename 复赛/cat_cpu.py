# -*- encoding:utf-8 -*-
# author: Jiaxin Peng

import pandas as pd
from sklearn.externals import joblib
import numpy as np
import catboost
import time
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


# catboost可以自己处理类别特征

class FeatureEngineering:
    def __init__(self):
        """
        训练数据和测试数据的每个样本的sid是唯一的，除去sid，训练数据有2个重复样本，测试数据没有重复样本
        :param train_data:
        :param test_data:
        """
        try:
            self.data = pd.read_table("/data1/pengjiaxin/round2/all_data.txt", delimiter=',')
            self.labels = pd.read_csv("/data1/pengjiaxin/round2/label.csv")['label']
        except:
            self.data = pd.read_table("data/all_data.txt", delimiter=',')
            self.labels = pd.read_csv("data/label.csv")['label']
        for c in self.data.columns:
            if self.data[c].dtype == 'float64':
                self.data[c] = self.data[c].astype('float32')
            if self.data[c].dtype == 'int64':
                self.data[c] = self.data[c].astype('int32')
        print(self.data.columns)

        # self.train_data.drop(index=[320443, 4456598, 1080823], inplace=True)
        # self.labels = self.train_data['label']
        # self.train_data.drop('label', axis=1, inplace=True)
        # self.test_data_sid = test_data['sid']
        # self.data = pd.concat([self.train_data, self.test_data], axis=0)  # 训练集 4999997 rows

    def run(self):
        """
        :return: train / test / label
        """
        data = self.feature_to_lower(self.data)
        data = self.transfer_empty_to_nan(data)
        data = self.fill_null(data)
        data = self.osv_feature(data)
        print('osv')
        data = self.lan_feature(data)
        print('lan')
        data = self.ver_feature(data)
        print('ver')
        data = self.nginxtime_feature(data)
        print('nginxtime')
        for c in data.columns:
            if data[c].dtype == 'float64':
                data[c] = data[c].astype('float32')
        #
        data = self.sid_feature(data)
        print('sid')
        data = self.h_w_ppi_feature(data)
        print('h,w,ppi')
        data = self.model_feature(data)
        print('model')
        data = self.make_feature(data)
        print('make')
        for c in data.columns:
            if data[c].dtype == 'float64':
                data[c] = data[c].astype('float32')
        data = self.ntt_feature(data)
        print('ntt')
        data = self.apptype_feature(data)
        print('apptype')
        data = self.dvctype_feature(data)
        print('dvctype')
        data = self.orientation_feature(data)
        print('orientation')
        for c in data.columns:
            if data[c].dtype == 'float64':
                data[c] = data[c].astype('float32')
        data = self.pkgname_feature(data)
        print('pkgname')
        data = self.carrier_feature(data)
        print('carrier')
        data = self.m_m_l(data)
        print('mml')
        data = self.imei(data)
        print('imei')
        data = self.ip_feature(data)
        print('ip')
        data = self.province_feature(data)
        print('province')
        data = self.city_feature(data)
        print('city')
        for c in data.columns:
            if data[c].dtype == 'float64':
                data[c] = data[c].astype('float32')
        data = pd.read_table("data/temp_all_data.txt", delimiter=',')
        data.to_csv("data/temp_all_data.txt", index=False)
        data = self.if_empty(data)
        data = self.remove_columns(data)
        data = self.combine_feature(data)
        print('combine')
        for c in data.columns:
            if data[c].dtype == 'float64':
                data[c] = data[c].astype('float32')
        # try:
        #     data.to_csv("/data1/pengjiaxin/round2/all_data.txt", index=False)
        # except:
        #     data.to_csv("data/all_data.txt", index=False)
        return
        self.data = self.shrink_feature(self.data, 120)

        print(self.data.info())
        for c in self.data.columns:
            if self.data[c].dtype == 'object':
                self.data[c] = self.data[c].astype("str")

        categorical_features_indices = np.where(self.data.dtypes == 'object')[0]
        # for c in self.data.columns:
        #     if self.data[c].isnull().any():
        #         self.data[c] = self.data[c].astype("str")
        train, test = self.data.iloc[:4999997, :], self.data.iloc[4999997:, :]
        del self.data
        gc.collect()  # 处理循环引用
        return train, test, self.labels, categorical_features_indices

    @staticmethod
    def remove_columns(data):
        """
        去除特征
        :param data
        :return:
        """
        # 去除数据全部一样的特征
        # data.drop('os', axis=1, inplace=True)
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
            if str(data[c].dtype) == 'object' and c != 'os':
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
        for c in data.columns:
            if str(data[c].dtype) == 'object':
                data[c] = pd.factorize(data[c])[0]
        return data

    @staticmethod
    def fill_null(data):
        """
        缺的不太多的特征用众数填充空值
        :param data:
        :return:
        """
        # 众数填充
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
        data['log_{}_count'.format('pkgname')] = np.log10(data.groupby(['{}'.format('pkgname')])['sid'].transform('count'))
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
        data['lan_empty'] = data['lan'].isnull().astype('str')
        data['new_lan'] = new_lan
        data['log_{}_count'.format('lan')] = np.log10(data.groupby(['{}'.format('lan')])['sid'].transform('count'))
        # data = data.drop('lan', axis=1)
        return data

    @staticmethod
    def ver_feature(data):
        """
        ver(app版本, string)
        :param data:
        :return:
        """
        data['ver_0'] = data['ver'].apply(lambda x: ''.join(re.findall('[0123456789.]', str(x))))
        data['ver_0'].replace('', np.nan, inplace=True)
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
        data['sid1'] = data['sid'].apply(lambda x: x.split('-')[1])
        data['sid2'] = data['sid'].apply(lambda x: x.split('-')[2])
        data['sid3'] = data['sid'].apply(lambda x: x.split('-')[3])
        data['sid1_count'] = np.log10(data.groupby(['{}'.format('sid1')])['sid'].transform('count'))
        data['sid2_count'] = np.log10(data.groupby(['{}'.format('sid2')])['sid'].transform('count'))
        data['sid3_count'] = np.log10(data.groupby(['{}'.format('sid3')])['sid'].transform('count'))
        data.drop('sid1', axis=1, inplace=True)
        data.drop('sid3', axis=1, inplace=True)

        data['sid'] = data['sid'].apply(lambda x: float(x.split('-')[-1]))
        data['sid_datetime'] = pd.to_datetime(data['sid'] / 1000, unit='s') + timedelta(hours=8)
        data['sid_hour'] = data['sid_datetime'].dt.hour
        data['sid_day'] = data['sid_datetime'].dt.day - data['sid_datetime'].dt.day.min()
        data['sid_minute'] = data['sid_datetime'].dt.minute.astype('uint8')
        data['nginxtime-sid_time'] = data['nginxtime'] - data['sid']  # 请求会话时间 与 请求到达服务时间的差
        data.drop(['nginxtime'], axis=1, inplace=True)
        # 做一下星期特征
        data['sid_weekday'] = data['sid_datetime'].apply(lambda x: x.strftime("%w"))
        data['sid_weekday_01'] = data['sid_weekday']
        data['sid_weekday_01'].replace('1', "0", inplace=True)
        data['sid_weekday_01'].replace('2', "0", inplace=True)
        data['sid_weekday_01'].replace('3', "0", inplace=True)
        data['sid_weekday_01'].replace('4', "0", inplace=True)
        data['sid_weekday_01'].replace('5', "0", inplace=True)
        data['sid_weekday_01'].replace('6', "1", inplace=True)
        data['sid_weekday_01'].replace('7', "1", inplace=True)
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
        data['ppi'].replace(0.0, np.nan, inplace=True)
        cols = ['h', 'w', 'ppi']
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
        data['px'] = data['ppi'] * data['size']
        data['mj'] = data['h'] * data['w']

        for i in tqdm(['h', 'w', 'ratio']):
            data['{}_count'.format(i)] = data.groupby(['{}'.format(i)])['sid'].transform('count')
        # data['h'].replace(0, np.nan, inplace=True)
        # 0的时候label=1很少 w, h特征强，某些w/h下能够较好区分label, 应该作为类型特征
        # data['w'].replace(0, np.nan, inplace=True)
        data['w'] = data['w'].astype('str')
        data['h'] = data['h'].astype('str')
        data['s'] = data['s'].astype('str')
        data['ppi'] = data['ppi'].astype('str')
        return data

    @staticmethod
    def ip_feature(data):
        """
        ip  ip分类：A B C
        :param data:
        :return:
        """
        data = data.iloc[6000000:, :]
        data = data.reset_index(drop=True)
        print(data)
        try:
            ip_info = pd.read_csv('/data1/pengjiaxin/round2/ip_info.csv')
        except FileNotFoundError:
            ip_info = pd.read_csv('ip_info.csv')
        ip_info = ip_info.iloc[6000000:, :]
        ip_info = ip_info.reset_index(drop=True)

        data['ip_country'] = ip_info['country']
        data['ip_country'].replace('0', np.nan, inplace=True)

        ind = (ip_info['city'] == '0')
        print(ind)
        ip_info['city'][ind] = data['city'][ind]
        data['city'] = ip_info['city']

        try:
            city_province = pd.read_excel('/data1/pengjiaxin/round2/city_province.xlsx')
        except FileNotFoundError:
            city_province = pd.read_excel('../city_province.xlsx')
        data = data.merge(city_province, on='city', how='left')
        print(data)
        del city_province
        gc.collect()
        ind = (ip_info['province'] == '0')
        ip_info['province'][ind] = data['province_new'][ind]
        data['province_new'] = ip_info['province']

        ind = (ip_info['carrier'] == '0')
        ip_info['carrier'][ind] = data['carrier'][ind]
        ip_info.carrier.replace(0.0, np.nan, inplace=True)
        ip_info.carrier.replace(-1.0, np.nan, inplace=True)
        print(data.shape)
        data['carrier'] = ip_info['carrier']
        del ip_info
        gc.collect()

        try:
            reqrealip_info = pd.read_csv('/data1/pengjiaxin/round2/reqrealip_info.csv')
        except FileNotFoundError:
            reqrealip_info = pd.read_csv('reqrealip_info.csv')
        reqrealip_info = reqrealip_info.iloc[6000000:, :]
        reqrealip_info = reqrealip_info.reset_index(drop=True)
        data['reqrealip_province'] = reqrealip_info.province
        data['reqrealip_province'].replace('0', np.nan, inplace=True)
        data['reqrealip_city'] = reqrealip_info.city
        data['reqrealip_city'].replace('0', np.nan, inplace=True)
        data['reqrealip_carrier'] = reqrealip_info.carrier
        data['reqrealip_carrier'].replace('0', np.nan, inplace=True)
        del reqrealip_info
        gc.collect()

        # 将ip的计数作为新特征
        data['{}_count'.format('ip')] = np.log10(data.groupby(['{}'.format('ip')])['sid'].transform('count'))
        data['{}_rank'.format('ip')] = data['{}_count'.format('ip')].rank(method='min')
        data['reqrealip_count'] = np.log10(data.groupby(['{}'.format('reqrealip')])['sid'].transform('count'))
        data['machine'] = data['model'] + data['make']
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
        data['model_equal_make'] = (data['model_0'] == data['make']).astype('str')
        data['{}_count'.format('model_0')] = np.log10(data.groupby(['{}'.format('model_0')])['sid'].transform('count'))
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
        del data
        gc.collect()
        return pd.concat([new_data, new_osv], axis=1)

    @staticmethod
    def ntt_feature(data):
        """
        ntt	int	网络类型 0-未知, 1-有线网, 2-WIFI, 3-蜂窝网络未知, 4-2G, 5-3G, 6–4G
        :param data:
        :return:
        """
        data.ntt[(data.ntt == 0) | (data.ntt == 7)] = 0
        data.ntt[(data.ntt == 1) | (data.ntt == 2)] = 1
        data.ntt[data.ntt == 3] = 2
        data.ntt[(data.ntt >= 4) & (data.ntt <= 6)] = 3
        data['ntt_str'] = data['ntt'].astype('str')
        data['{}_count'.format('ntt')] = np.log10(data.groupby(['{}'.format('ntt')])['sid'].transform('count'))
        return data

    @staticmethod
    def dvctype_feature(data):
        """
        dvctype	int	设备类型 0 – 未知,1 – PC,2 – 手机, 3– 平板,4– 电视盒,5– 智能电视,6 – 可穿戴设备,7 – 智能家电,8 - 音箱,9 - 智能硬件
        :param data:
        :return:
        """
        data['dvctype'] = data['dvctype'].astype('str')
        data['{}_count'.format('dvctype')] = np.log10(data.groupby(['{}'.format('dvctype')])['sid'].transform('count'))
        return data

    @staticmethod
    def apptype_feature(data):
        """
        apptype	int	app所属分类
        :param data:
        :return:
        """
        # 将apptype的计数作为新特征
        data['{}_count'.format('apptype')] = np.log10(data.groupby(['{}'.format('apptype')])['sid'].transform('count'))
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
        data['orientation'] = data['orientation'].astype('str')
        return data

    @staticmethod
    def province_feature(data):
        """
        :param data:
        :return:
        """
        data['province_count'] = np.log10(data.groupby(['{}'.format('province')])['sid'].transform('count'))
        data['province'] = data['province'].astype('str')
        return data

    @staticmethod
    def city_feature(data):
        """
        :param data:
        :return:
        """
        data['city_count'] = np.log10(data.groupby(['{}'.format('city')])['sid'].transform('count'))
        city_divide = []
        for c in data['city']:
            if c is not np.nan:
                if c in ['北京市', '上海市', '深圳市', '广州市']:
                    city_divide.append('1')
                elif c.split('市')[0] in ['杭州', '南京', '重庆', '青岛', '济南', '厦门', '成都', '武汉', '苏州',
                                                    '长沙', '天津', '哈尔滨', '郑州', '沈阳', '西安', '宁波']:
                    city_divide.append('2')
                else:
                    city_divide.append('3')
            else:
                city_divide.append(np.nan)
        data['city_divide'] = city_divide
        return data

    @staticmethod
    def carrier_feature(data):
        """
        carrier	string	运营商 0-未知, 46000-移动, 46001-联通, 46003-电信
        :param data:
        :return:
        """
        data['carrier_count'] = np.log10(data.groupby(['{}'.format('carrier')])['sid'].transform('count'))
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
        data['{}_count'.format('imeimd5')] = np.log10(data.groupby(['{}'.format('imeimd5')])['sid'].transform('count'))
        data['imeimd5_01'] = data['imeimd5'].isnull().astype('str')
        return data

    @staticmethod
    def if_empty(data):
        """
        pkgname 0.32057071428571426
        adunitshowid 0.0004055714285714286
        mediashowid 0.0004055714285714286
        adidmd5 0.46596642857142856
        imeimd5 0.010980714285714286
        idfamd5 0.9941844285714285
        openudidmd5 0.9576174285714286
        macmd5 0.6345358571428571
        :param data:
        :return:
            """
        for c in ['pkgname', 'adidmd5', 'openudidmd5', 'macmd5']:
            data[c+'_empty'] = data[c].isnull().astype('str')
        return data

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
        data['ip-province-city'] = data['ip'] + data['province'].astype('str') + data['city'].astype('str')
        data['ip-province'] = data['ip'] + data['province'].astype('str')
        data['province-city'] = data['city'].astype('str') + data['province'].astype('str')
        data['countRatio_model_dvctype'] = data['ratio_count'].astype('str') + data['model'].astype('str') + data['dvctype'].astype('str')
        data['countRatio_model_make'] = data['ratio_count'].astype('str') + data['model'].astype('str') + data['make'].astype('str')
        data['countRatio_model_openudidmd5'] = data['ratio_count'].astype('str') + data['model'].astype('str') + data['openudidmd5'].astype('str')
        data['countRatio_model_macmd5'] = data['ratio_count'].astype('str') + data['model'].astype('str') + data['macmd5'].astype('str')
        data['countRatio_pkgname_ver'] = data['ratio_count'].astype('str') + data['pkgname'].astype('str') + data['ver'].astype('str')

        data['mac_adunit'] = data['macmd5'].astype('str') + data['adunitshowid'].astype('str')
        data['mac_media'] = data['macmd5'].astype('str') + data['mediashowid'].astype('str')
        data['mac_adunit_count'] = np.log10(data.groupby(['mac_adunit'])['sid'].transform('count'))
        data['mac_media_count'] = np.log10(data.groupby(['mac_media'])['sid'].transform('count'))
        return data

    @staticmethod
    def shrink_feature(data, size):
        try:
            f_f_impo = 'data/feature_importance_cv5.csv'
            df = pd.read_csv(f_f_impo)
        except:
            f_f_impo = '/data1/pengjiaxin/round2/feature_importance_cv5.csv'
            df = pd.read_csv(f_f_impo)
        df = df.sort_values(by='importance', ascending=False)
        fea_impo_shrink = list(df['feature_name'][:size])
        # fea_impo = list(pd.read_csv(f_f_impo)['feature'])
        shrink_fea = []
        # for i in data.columns:
        #     if i in fea_impo_shrink or i not in fea_impo:
        #     # if i in fea_impo_shrink:
        #         shrink_fea.append(i)
        data = data[fea_impo_shrink]
        return data


class ModelConstruction:
    def __init__(self, train_data, test_data, label):
        self.train_data = train_data
        self.test_data = test_data
        self.label = label
        print(self.train_data.shape, self.test_data.shape, len(self.label))

    def run(self, categorical_features_indices):
        final_pred_prob = []
        valid_final_pred_prob = []
        cv_model = []

        self.train_data['label'] = self.label
        self.train_data = self.train_data.sample(n=300000, axis=0, random_state=2019)
        joblib.dump(self.train_data, "new_feature_testdata_300000")
        self.label = self.train_data['label']
        self.train_data.drop(['label'], axis=1, inplace=True)

        skf = StratifiedKFold(n_splits=5, random_state=2019, shuffle=True)
        for index, (train_index, test_index) in enumerate(skf.split(self.train_data, self.label)):
            print(index)
            train_x, test_x, train_y, test_y = self.train_data.iloc[train_index], self.train_data.iloc[test_index], \
                                               self.label.iloc[train_index], self.label.iloc[test_index]
            model = catboost.CatBoostClassifier(iterations=3000, depth=8, cat_features=categorical_features_indices,
                                                early_stopping_rounds=500, verbose=50,
                                                learning_rate=0.1, eval_metric='F1',
                                                custom_loss='F1')
            model.fit(train_x, train_y, eval_set=(test_x, test_y))
            del train_x, train_y, test_x, test_y
            gc.collect()

            # cv_model.append(model)
            # y_test = model.predict_proba(self.test_data)
            # y_valid = model.predict_proba(self.train_data)
            # final_pred_prob.append(y_test)
            # valid_final_pred_prob.append(y_valid)
            # 保存
            # try:
            #     joblib.dump(final_pred_prob, '/data1/pengjiaxin/round2/'+str(index)+'_final_pred_prob')
            #     print("final_pred_prob is dumped.")
            #     joblib.dump(valid_final_pred_prob, '/data1/pengjiaxin/round2/' + str(index) + '_vali_final_pred_prob')
            #     joblib.dump(cv_model, '/data1/pengjiaxin/round2/'+str(index) + '_cv5_model')
            # except:
            #     joblib.dump(final_pred_prob, 'data/' + str(index) + '_final_pred_prob')
            #     joblib.dump(valid_final_pred_prob, 'data/' + str(index) + '_vali_final_pred_prob')
            #     joblib.dump(cv_model, 'model/' + str(index) + '_cpu_cv5_model')

            # if index == 0:
            #     fea = cv_model[index].feature_importances_
            #     feature_name = cv_model[index].feature_names_
            #     feature_importance = pd.DataFrame({'feature_name': feature_name, 'importance': fea})
            #     try:
            #         feature_importance.to_csv('/data1/pengjiaxin/round2/feature_importance_cv5.csv', index=False)
            #     except:
            #         feature_importance.to_csv('data/feature_importance_cv5.csv', index=False)
        print('final_pred_prob:', final_pred_prob)


def test_data_predict(model, test_data, sid):
    test_data_sid = pd.DataFrame(data=sid, columns=['sid'])
    prediction = model.predict(test_data)
    test_data_sid['label'] = prediction
    print(sum(prediction))
    test_data_sid.to_csv('model/case4/test_predict3.csv')

# try:
#     filename = r"data/round2_iflyad_anticheat_traindata.txt"
#     train = pd.read_table(filename)
#     filename = r"data/round2_iflyad_anticheat_testdata_feature_A.txt"
#     test_A = pd.read_table(filename)
#     filename = r"data/round2_iflyad_anticheat_testdata_feature_B.txt"
#     test_B = pd.read_table(filename)
# except:
#     filename = r"/data1/pengjiaxin/round2_iflyad_anticheat_traindata.txt"
#     train = pd.read_table(filename)
#     filename = r"/data1/pengjiaxin/round2_iflyad_anticheat_testdata_feature_A.txt"
#     test_A = pd.read_table(filename)
#     filename = r"/data1/pengjiaxin/round2_iflyad_anticheat_testdata_feature_B.txt"
#     test_B = pd.read_table(filename)

# data = pd.read_table("data/ip1234_all_data.txt", delimiter=',')


fe = FeatureEngineering()
train_data, test_data, label, cfi = fe.run()
# ModelConstruction(train_data, test_data, label).run(cfi)

# model = joblib.load('model/case4/catboost4')
# test_data_predict(model, test_data, fe.test_data_sid)


