## 2018中国高校计算机大赛——大数据挑战赛

baseline：线上0.8201

提取特征的程序：extract_features.py

设置时间窗和预测活跃用户数量的阈值：param.py

训练lgb模型生成submission文件：train.py

## 时间窗划分记录

行为 标签

1-16 17-23 train1 

2-17 18-24 train2

3-18 19-25 train3

4-19 20-26 train4

5-20 21-27 train5

6-21 22-28 train6

7-22 23-29 train7

8-23 24-30 train8

一、

train1+train5

train2+train6

train3+train7

train4+train8

一共4份训练集，每个跑5折交叉，结果集成：0.820670

二、

训练集 验证集

train1 train5

train2 train6

train3 train7

train4 train8

train5 train1

train6 train2

train7 train3

train8 train4

一共8份训练集/验证集，结果集成：0.8199

三、

训练集

train1+train8

一份训练集跑5折交叉：0.820165
