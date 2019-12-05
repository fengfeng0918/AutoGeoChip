"""
author: dfq

date: 2019年6月11日13:56:38

version： 1.0.0

desription: A script for auto-analysising GEO database micro-chip data.
"""

from ftplib import FTP
import os
import gzip
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats import multitest
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
# from matplotlib.lines import Line2D
from matplotlib import cm
from matplotlib.colors import ListedColormap
import re
from collections import defaultdict
from itertools import permutations
import scipy
from scipy.stats import pearsonr
import shutil
import time
from plot_hcluster import hcluster2


class GetSample:
    """
    Read the sample file, and  assign the group for each .cel file in GEO.
    """

    def __init__(self,sample_file):
        self.sample_file = sample_file


    def sample(self):
        """ Read the sample file .
        :return: [pd.groupby]
        """
        if not os.path.exists(self.sample_file):
            print("Please assign the information of sample and group!\n"
                  "the sample file like:\n"
                  "accession\tgroup\n"
                  "sample1\tgroup1\n"
                  "sample2\tgroup1\n"
                  "sample3\tgroup1\n"
                  "sample4\tgroup2\n"
                  "sample5\tgroup2\n"
                  "sample6\tgroup2\n"
                  "the more columns are allowed, but not used in GeoPy!")
        sample = pd.read_csv(self.sample_file,sep="\t",header=0,index_col=0)
        return sample.groupby(sample.columns[0])


class Getgeo:
    def __init__(self, GSE_NUM:str, type="series"):
        self.GSE_NUM = GSE_NUM
        if type.upper() not in ["SERIES","SOFT"]:
            raise ValueError("Please choose 'SERIES' or 'SOFT'")
        self.type = type

    def data_download(self):
        """
        Downloading GEO data, which is in series or soft file-suffix.
        :return: [string] the downloading filename.
        """

        if self.type.upper() == "SERIES":
            print("默认下载series文件，如需下载soft文件，请添加type='soft'")
            # ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE43nnn/GSE43014/matrix/GSE43014_series_matrix.txt.gz
            file_type = "series_matrix.txt"
            dir = "matrix"
        else:
            # ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE43nnn/GSE43014/soft/GSE43014_family.soft.gz
            file_type = "family.soft"
            dir = "soft"

        file_name = "{}_{}.gz".format(self.GSE_NUM, file_type)
        if not os.path.exists(file_name):
            self._downloader(file_type, dir)
        return "{}_{}.gz".format(self.GSE_NUM, file_type)

    def _downloader(self,file_type, dir):
        """
        Download the file from GEO databases.
        :param file_type: the postfix for GEO raw file
        :param dir: the position for storing raw file
        :return: .gz
        """
        if not self.GSE_NUM.upper().startswith("GSE"):
            print("Please assign the right GEO accession! eg:GSE00000")

        file_name = "{}_{}.gz".format(self.GSE_NUM, file_type)
        fp = open(file_name, 'wb')
        ftp = FTP('ftp.ncbi.nlm.nih.gov')
        ftp.login()
        gse_catalogue = self.GSE_NUM[:-3] + "nnn"
        geo_file = 'geo/series/{0}/{1}/{3}/{1}_{2}.gz'.format(gse_catalogue, self.GSE_NUM, file_type, dir)
        ftp.retrbinary('RETR ' + geo_file, fp.write)
        ftp.quit()


class Analysis(Getgeo):
    """the main  analysising program.
    The input filename  must be like [GEO]_family.soft or [GEO]_series_matrix.txt
    """

    def __init__(self, GSE_NUM, type="series", sample_file=None, log2=True, bg_filter=True, cv=True):
        super().__init__(GSE_NUM, type)
        self.file_name = "{}_series_matrix.txt".format(self.GSE_NUM)   # matrix 文件
        self.raw_data = None
        self.sample_file = sample_file
        self.soft_file = self.GSE_NUM + "_family.soft"
        self.cv = cv
        self.log2 = log2
        self.bg_filter = bg_filter

    def make_sample_file(self,accession,group,title,*Args):
        file = open("sample_auto.txt", "w", encoding="UTF-8")
        file.write("accession\tgroup\ttitle\n")
        for i in zip(accession,group,title,*Args):
            file.write("\t".join(i)+"\n")
        file.close()

    def data_parse(self):
        """
        mian function
        :return:
        """

        if self.type == 'series':
            if not os.path.exists(self.file_name):
                file_downloaded = self.data_download()
                self.un_gz(file_downloaded)

        if self.type == 'soft':
            if not os.path.exists(self.soft_file):
                file_downloaded = self.data_download()
                self.un_gz(file_downloaded)
            self.parse_soft()

        self.get_matrix()

        # 获取sample信息
        if self.sample_file:
            self.sample = GetSample(self.sample_file).sample()
        else:
            self.sample = GetSample("sample_auto.txt").sample()

        # TODO 获取compare信息
        g1 = 'Resistant 4OHT'
        g2 = 'Resistant control'

        # 统计分析
        self.statistic(g1,g2,self.log2, self.bg_filter, self.cv)
        # 差异筛选

        # replace the empty value
        # filtering background  (cut the probe which  < 20% in all sample, must be)
        # filtering  coefficient of variation(cut the probe which > 25% in each group,default is true, and can be ignored)
        # statistical test (P value and FDR, default FDR<0.05, and support the custom-defined)
        # filtering fold change ( user-defined )
        # filter the priority of probe

    def parse_soft(self):
        # if not os.path.exists(self.file_name):
        matrix = open(self.file_name,"w",encoding='utf-8')
        annot = open(self.GSE_NUM + "_annot.txt","w",encoding='utf-8')

        if not os.path.exists(self.soft_file):
            raise FileExistsError("{}文件不存在".format(self.soft_file))

        # SERIES = re.compile(r"^\^SERIES = (.*)\n")
        # PLATFORM = re.compile(r"^\^PLATFORM = (.*)\n")
        SAMPLE = re.compile(r"^\^SAMPLE = (.*)\n")
        platform_table_begin = re.compile(r"^!platform_table_begin")
        platform_table_end = re.compile(r"^!platform_table_end")
        # sample_table_begin = re.compile(r"^!sample_table_begin")
        sample_table_end = re.compile(r"^!sample_table_end")
        title = re.compile(r"^ID_REF\tVALUE")
        sample_group = re.compile(r"^!Sample_source_name_ch1 = (.*)\n")
        sample_title = re.compile(r"^!Sample_title = (.*)\n")

        sample_accession_list = list()
        matrix_dict = defaultdict(list)
        sample_group_list = list()
        sample_title_list = list()
        with open(self.soft_file, "r", encoding='utf-8') as f:
            for line in f:
                if re.match(platform_table_begin,line):
                    while True:
                        line_annot = f.__next__()
                        if re.match(platform_table_end,line_annot):
                            break
                        annot.write(line_annot)
                if re.findall(SAMPLE,line):
                    sample_accession_list.append(re.findall(SAMPLE,line)[0])
                if re.findall(sample_group,line):
                    sample_group_list.append(re.findall(sample_group,line)[0])
                if re.findall(sample_title,line):
                    sample_title_list.append(re.findall(sample_title,line)[0])
                if re.match(title,line):
                    while True:
                        line_matrix = f.__next__()
                        if re.match(sample_table_end,line_matrix):
                            break
                        line_matrix = line_matrix.strip().split("\t")
                        matrix_dict[line_matrix[0]].append(line_matrix[1])
        matrix.write("ID_REF\t{}\n".format("\t".join(sample_accession_list)))
        for key, value in matrix_dict.items():
            matrix.write("{}\t{}\n".format(key,"\t".join(value)))

        self.make_sample_file(sample_accession_list,sample_group_list,sample_title_list)

    def get_matrix(self):
        """
        Transform the raw series matrix into matrix with pure expression value.
        :return:
        """
        # file = open(file_name,"r")
        raw_file_name = "{}_raw.txt".format(self.GSE_NUM)
        file_clean = open(raw_file_name,"w",encoding='utf-8')

        sample_group = re.compile(r"^!Sample_source_name_ch1\t(.*)\n")
        sample_title = re.compile(r"^!Sample_title\t(.*)\n")
        sample_accession = re.compile(r"^!Sample_geo_accession\t(.*)\n")

        sample_accession_list = list()
        sample_group_list = list()
        sample_title_list = list()
        with open(self.file_name,"r",encoding='utf-8') as f:
            for line in f:
                if re.findall(sample_accession,line):
                    sample_accession_list = re.findall(sample_accession,line)[0].replace('"',"").split("\t")
                if re.findall(sample_group,line):
                    sample_group_list = re.findall(sample_group,line)[0].replace('"',"").split("\t")
                if re.findall(sample_title,line):
                    sample_title_list = re.findall(sample_title,line)[0].replace('"',"").split("\t")
                if not line.startswith("\n") and not line.startswith("!"):  #todo empty line ---done
                    line= line.replace('"',"")
                    file_clean.write(line)
        file_clean.close()

        # print(sample_accession_list,sample_group_list, sample_title_list)
        df = pd.read_csv(raw_file_name, sep="\t",index_col =0)
        df = df.fillna(value=0.000001)
        print("Notes:Filling the empty value with 0.000001!")
        self.raw_data = df

        if sample_accession_list and sample_group_list:
            self.make_sample_file(sample_accession_list, sample_group_list, sample_title_list)

    def statistic(self, group1, group2, log2, bg_filter, cv, bg_value=0.2,cv_value=0.25):

        df = self.raw_data[self.sample.groups[group1].union(self.sample.groups[group2])]

        if log2:
            print("[{0}_vs_{1}]Notes: The data has been transfer with log2(x+1)!\t{2}"
                  .format(group1,group2,df.shape))
            df = np.log2(df + 1)   # Apply log2 transformation to a pandas DataFrame and avoid the signal less zero

        if bg_filter:
            if not (0 < bg_value < 1):
                raise ValueError("The bg_value for backgroud filter must be in(0,1).")
            bg_less_20_percent = df.describe(percentiles=[bg_value]).loc["{0}%".format(int(bg_value*100))]   # 获取每个样本20%背景值
            for index,col in enumerate(df.columns.values):
                ff = bg_less_20_percent[index]
                df =df.loc[df[col] > ff,:]                    # 1.过滤背景
            print("[{0}_vs_{1}]Notes: The probe, which signal  is less then 20% ,has been deleted for each sample!\t{2}"
                  .format(group1,group2,df.shape))
            # df.to_csv("bg_less_20_percent.txt", sep="\t")

        if cv:
            if not (0 < cv_value <= 1):
                raise ValueError("The cv_value for CV filter must be in (0,1].")
            for group in [group1,group2]:
                sample_name = self.sample.groups[group]
                df["CV_{}".format(group)] = df.loc[:, sample_name].apply(lambda x:np.std(x)/np.mean(x),axis=1)   # 2.过滤CV
                df = df[df["CV_{}".format(group)] <= cv_value]
            print("[{0}_vs_{1}]Notes: The probe, which CV value is more than 25% ,has been deleted!\t{2}"
                  .format(group1,group2,df.shape))

        # 计算P值
        # 方法1 ~52s 50k行
        # sample1 = self.sample.groups[group1]
        # sample2 = self.sample.groups[group2]
        # df["p_value"] = df.apply(lambda x: stats.ttest_ind(x[sample1], x[sample2])[1], axis=1)
        # TODO 此步骤非常慢，找到问题并优化！！
        # # 以上x[sample_name[0]]操作非常耗时

        # 方法2 ~22s 50k行
        # num = len(self.sample.groups[group1])
        # num_all = num + len(self.sample.groups[group2])
        # df["p_value"] = df.apply(lambda x: stats.ttest_ind(x[:num], x[num:num_all])[1], axis=1)

        # 方法3 ~12s 50k行
        # map(), 列表解析，手动循环耗时基本一致
        temp = df.values
        num = len(self.sample.groups[group1])
        num_all = num + len(self.sample.groups[group2])
        p_value = list(map(lambda x: stats.ttest_ind(x[:num], x[num:num_all])[1], temp))
        df["p_value"] = pd.Series(p_value,index=df.index)
        print("[{0}_vs_{1}]Notes: Calculate the P-value!\t{2}"
              .format(group1, group2, df.shape))

        # 计算FDR值
        df["FDR"] = multitest.multipletests(df["p_value"],method="fdr_bh")[1]
        print("[{0}_vs_{1}]Notes: Calculate the FDR!\t{2}"
              .format(group1, group2, df.shape))

        # 计算ratio值   与计算P值耗时基本一致
        ratio = list(map(lambda x: sum(x[:num])/sum(x[num:num_all]), temp))
        fc = [i if i>=1 else -1/i for i in ratio]
        abs_fc = [abs(i) for i in fc]
        df["Ratio"] = pd.Series(ratio, index=df.index)
        df["Fold Change"] = pd.Series(fc, index=df.index)
        df["Abs_FC"] = pd.Series(abs_fc, index=df.index)
        print("[{0}_vs_{1}]Notes: Calculate the ratio,FC and Abs_FC!\t{2}"
              .format(group1, group2, df.shape))

        df.to_csv("statistic.txt", sep="\t")
        return df

    def get_diff(self,df,FC=2,FDR=0.05,p=0.05):
        pass


    def get_new_colormap(self):
        """生成自定义色板"""

        # coolwarm111 = cm.get_cmap('coolwarm', 256)
        # bwr111 = cm.get_cmap('bwr', 256)
        # coolwarm111 = coolwarm111(np.linspace(0, 1, 256))
        # bwr111 = bwr111(np.linspace(0, 1, 256))
        # pink = bwr111[114:143, :]
        # coolwarm111[114:143, :] = bwr111[114:143, :]
        # newcmp = ListedColormap(coolwarm111)

        # ss = cm.get_cmap('coolwarm', 100)

        # 生成green
        N = 256
        vals = np.ones((N, 4))
        vals[:, 0] = np.linspace(46 / 255, 1, N)
        vals[:, 1] = np.linspace(139 / 255, 1, N)
        vals[:, 2] = np.linspace(87 / 255, 1, N)
        newcmp1 = ListedColormap(vals)
        # 生成red
        N = 256
        vals = np.ones((N, 4))
        vals[:, 0] = np.linspace(255 / 255, 1, N)
        vals[:, 1] = np.linspace(0 / 255, 1, N)
        vals[:, 2] = np.linspace(0 / 255, 1, N)
        newcmp2 = ListedColormap(vals)

        top = cm.get_cmap(newcmp2, 100)
        bottom = cm.get_cmap(newcmp1.reversed(), 100)

        newcolors = np.vstack((top(np.linspace(0, 1, 50))[10:48, :],
                               bottom(np.linspace(0, 1, 50))[2:40, :]))
        newcmp = ListedColormap(newcolors, name='OrangeBlue').reversed()
        return newcmp

    def qc(self):
        # 盒装图
        df3 = self.raw_data.apply(lambda x: (x - x.mean()) / x.std())
        fig = plt.figure()
        ax = fig.add_axes([0.2, 0.3, 0.75, 0.65])
        fig, ax = plt.subplots()
        ax.boxplot((df3.apply(lambda x: (x - x.mean()) / x.std())).T, showfliers=False)
        ax.set_xticklabels(df3.columns, rotation=90)

        ax.axhline(0, color="r", linestyle="dashed", linewidth=0.5)
        # boxplots.set_color("red")
        # print(boxplots['boxes'])
        fig.savefig("box_s.png",)


        # pearson图
        dfrr = self.raw_data.corr() # pearson 相关系数  datadrame
        fig = plt.figure()  # figsize=(9,9))
        ax = fig.add_axes([0.3,0.3,0.65,0.65])
        ax = sns.heatmap(dfrr,ax=ax,cmap=self.get_new_colormap())
        plt.setp(ax.get_xticklabels(), rotation=90,)
        plt.setp(ax.get_yticklabels(), rotation=0) #, ha="right",rotation_mode="anchor")
        # ax.set_xticklabels()
        fig.savefig("pearson.png")



        # PCA图
        self.pca_plot()

    def pca_plot(self):
        """绘制PCA图"""
        def eigsorted(cov):
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            return vals[order], vecs[:, order]

        def _plot_pca(pca_pc, sample_name, ax, color, marker):   #  todo  生成函数
            nstd = 2
            x = pca_pc.loc[sample_name,"x"]
            y = pca_pc.loc[sample_name,"y"]
            cov = np.cov(x, y)
            vals, vecs = eigsorted(cov)
            theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
            w, h = 2 * nstd * np.sqrt(vals)
            ell = Ellipse(xy=(np.mean(x), np.mean(y)), width=w, height=h, angle=theta, color=color)
            ell.set_facecolor(color)
            ell.set_alpha(0.1)
            ax.add_artist(ell)
            ax.scatter(x, y, c=color, marker=marker)

        self.raw_data = self.raw_data.T
        pca = PCA(n_components=2)
        pca.fit(self.raw_data)
        pca_pc = pd.DataFrame(pca.transform(self.raw_data),index=self.raw_data.index,columns=["x","y"])
        # print(pca_pc)
        fig, ax = plt.subplots()
        marker_list = ('o', 'v', 's', 'p', '*', 'd')
        color_list = ["green", "red", "blue", "deeppink","darkviolet","darkorange"]
        for i in range(len(self.sample.groups)):
            group = list(self.sample.groups.keys())[i]
            sample_name = list(list(self.sample.groups.values())[i])
            color = color_list[i]
            marker = marker_list[i]
            _plot_pca(pca_pc, sample_name, ax, color, marker)

        y_max = pca_pc.y.abs().max()
        x_max = pca_pc.x.abs().max()
        ax.set_ylim(-y_max*1.5, y_max*1.5)
        ax.set_xlim(-x_max*1.5, x_max*1.5)
        ax.axvline(0, color="gray", linestyle="dashed", linewidth=0.5)
        ax.axhline(0, color="gray", linestyle="dashed", linewidth=0.5)

        # ax.spines['right'].set_color('none')
        # ax.spines['top'].set_color('none')
        # ax.xaxis.set_ticks_position('bottom')g
        # ax.spines['bottom'].set_position(('data', 0))
        # ax.yaxis.set_ticks_position('left')
        # ax.spines['left'].set_position(('data', 0))

        fig.savefig("pca.png")

    def un_gz(self, file_name):
        """ungz zip file"""
        f_name = file_name.replace(".gz", "") # 获取文件的名称，去掉
        g_file = gzip.GzipFile(file_name) # 创建gzip对象
        open(f_name, "wb").write(g_file.read())  # gzip对象用read()打开后，写入open()建立的文件中。
        g_file.close() # 关闭gzip对象


class AdvantageAanlaysis():

    # 富集分析 2种实现方式
    # 层次聚类
    # 趋势聚类
    def trend_Clustering(self, file, method="direct"):
        """
        趋势聚类。两种聚类模式可以设置。
        :param file: 输入文件
        :param method: 设置趋势聚类使用的方法，“direct"直接根据变化值进行分类，”Trend“利用距离分类。
        :return:
        """

        def count_up_or_down(data):
            """
            计算每两个点之间的变化。上调为1，下调为-1，不变为0
            :param data: pandas Dataframe
            :return: pandas Dataframe
            """
            for i in range(len(data.columns) - 1):
                time1 = data.columns[i]
                time2 = data.columns[i + 1]
                new_col = "FC{}".format(i)
                # 计算每两个时间点的FC
                data[new_col] = data[time2] / data[time1]
                data.loc[data[new_col] < 1, new_col] = -1 / data[new_col]
                # 计算表达模式profile
                profile = "profile{}".format(i)
                data[profile] = np.nan
                data.loc[data[new_col] > 1.5, profile] = 1
                data.loc[data[new_col] < -1.5, profile] = -1
                data.loc[data[new_col].abs() < 1.5, profile] = 0
            return data

        def profile_index(data):
            """
            根据count_up_or_down函数计算各个基因属于的profile
            :param data: pandas DataFrame
            :return:  pandas DataFrame
            """
            # 输出profile

            name = [col for col in data.columns if col.startswith("profile")]
            data_profile = data[name]

            # print(data_profile.head())

            def _profile_index(line):
                line = list(line + 1)
                line.reverse()
                profile_index = 0
                for index, value in enumerate(line):
                    profile_index += value * 3 ** index
                return profile_index

            data["profile_index"] = data_profile.apply(_profile_index, axis=1)
            return data

        def profile_plot(data, time_point_num, file_name='MainProfile.png'):
            """
            根据profile分类，画图。
            :param data:
            :param time_point_num:
            :param file_name:
            :return:
            """

            def trim_axs(axs, N):
                """little helper to massage the axs list to have correct length..."""
                axs = axs.flat
                for ax in axs[N:]:
                    ax.remove()
                return axs[:N]

            group = data.groupby(["profile_index"])
            N = len(group)
            x = round(np.sqrt(N))
            y = round(N / x)
            if x * y < N:
                y += 1

            plt.tight_layout()
            fig, axes = plt.subplots(int(x), int(y), figsize=(20, 15))
            axes = trim_axs(axes, N)
            for ax, (name, signal) in zip(axes.flat, data.groupby(["profile_index"])):
                # print("profile_{}".format(name))
                data = signal.iloc[:, 0:time_point_num].apply(lambda x: x - x[0], axis=1)
                ax.plot(data.T, color="green", alpha=0.12)

                # mean_point = list(data.mean().T)
                # mean_point_copy = mean_point.copy()
                # for i in range(1,len(mean_point)-1):
                #     if 1/1.5<mean_point[i+1]/mean_point[i]<1.5:
                #         mean_point_copy[i + 1] = mean_point_copy[i]
                # ax.plot(mean_point_copy,color="red")
                ax.plot(data.mean().T, color="red")
                ax.set_title("profile_{}".format(name))
                ax.set_xticks([])
                ax.set_yticks([])

            fig.savefig(file_name)

        def _get_profile_name(time_point_num):
            """
            生成profile的名称，如(0,1,2,3) (0,1,1,2)
            :param time_point_num:
            :return: list
            """
            start = [[0]]
            for i in range(time_point_num-1):  # 第一个点为初始点（0点）
                temps = list()
                for temp in start:
                    for num in [1,0,-1]:
                        temps.append(list(temp) + [temp[-1]+num])
                start = temps
            return start

        def get_min_distance(line,profile_name):
            """
            距离计算。如果是计算pearson相关系数则用argmax(),如果用欧几里得距离则用argmin().
            :param line:
            :param profile_name:
            :return:
            """
            idx = np.array([pearsonr(i,line.values)[0] for i in profile_name.values]).argmax()
            return profile_name.index[idx]

        def get_profile_with_distance(data,time_point_num):
            # data = data.T
            # data = (2 * (data - data.min())/(data.max() - data.min()) - 1).T  # premnmx 转换方法，将数据转换为[-1,1]
            # data = data.apply(lambda x:x-data.iloc[:,0])  # 减去第一个点
            # data = data * time_point_num   # 数据缩放至点个数范围
            profile_name = _get_profile_name(time_point_num)  # [[0,1,2,3],[0,1,1,1,],...]
            profile_name.remove([0]*time_point_num)
            profile_name = pd.DataFrame(profile_name,index=[",".join(map(lambda i: str(i),x)) for x in profile_name])
            profile_index = data.apply(get_min_distance,profile_name=profile_name,axis=1)
            #todo  当设置profile个数小于3**（time_point_num-1）时，如何缩减？
            profile_index = pd.Series(profile_index,index=data.index)
            data['profile_index'] = profile_index
            return data

        data = pd.read_csv(file, sep="\t", header=0, index_col=0)
        time_point_num = len(data.columns)

        if method == "direct":
            data = count_up_or_down(data)
            data = profile_index(data)

        if method == "trend":
            data = get_profile_with_distance(data,time_point_num)

        data.to_csv("profile.txt", sep="\t")
        profile_plot(data, time_point_num)

    def heatmap(self,file):
        """
        层次聚类
        :param data:
        :return:
        """
        hcluster2(file)

    def enrichment(self):
        """
        富集分析
        :return:
        """
        pass

if __name__ == "__main__":
    sss = time.time()
    ff = Analysis("GSE26459","series")
    # ff = Analysis("GSE26459","soft")
    ff.data_parse()
    # ff.data_clean()
    # print("the time of main:{}".format(time.time()-sss))
    # ff.parse_soft()
    # ff.data_download()
    # ff.get_matrix("GSE43014_series_matrix.txt")
    # ff.data_parse()
    # ff.data_clean(False)
    # ff.qc()
    # ff.pca_plot()

    sample = GetSample("sample_auto.txt").sample()
    # sample = sample.sample()
    # print(sample.groups.keys())
    # print(sample.groups.items())

