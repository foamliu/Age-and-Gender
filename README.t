# 同时识别年龄与性别
基于PyTorch 实现多任务学习，在同时识别年龄与性别。


## 数据集

[IMDB-WIKI](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) 数据集，460,723张图片。

![image](https://github.com/foamliu/Joint-Estimation-of-Age-and-Gender/raw/master/images/imdb-wiki-teaser.png)

这里为简洁只使用 IMDB 数据集。

### 年龄分布：

460723张照片为平衡每个年龄最多只保存5000张，清洗后得到163065张，按年龄分布作图：

![image](https://github.com/foamliu/Joint-Estimation-of-Age-and-Gender/raw/master/images/age.png)

## 用法

### 数据预处理
提取163065张图片：
```bash
$ python pre_process.py
```

### 训练
```bash
$ python train.py
```

### 准确度比较

#|图片大小|网络|性别准确度(验证集)|年龄MAE(验证集)|年龄损失函数|批量大小|Loss|
|---|---|---|---|---|---|---|---|
|1|112x112|ResNet-18|90.669%|5.838|回归(L1Loss)|32|0.9888|
|2|224x224|ResNet-18|90.887%|5.694|回归(L1Loss)|32|0.9719|
|3|224x224|ResNet-18|90.064%|8.475|分类(交叉熵)|32|TBD|
|4|224x224|ResNet-50|90.034%|TBD|分类(交叉熵)|32|TBD|


### Demo
```bash
$ python demo.py
```

效果图如下：

原图 | 校准 | 识别 | 标注 |
|---|---|---|---|
|![image](https://github.com/foamliu/Age-and-Gender/raw/master/images/0_raw.jpg)|![image](https://github.com/foamliu/Age-and-Gender/raw/master/images/0_img.jpg)|$(result_out_0)|$(result_true_0)|
|![image](https://github.com/foamliu/Age-and-Gender/raw/master/images/1_raw.jpg)|![image](https://github.com/foamliu/Age-and-Gender/raw/master/images/1_img.jpg)|$(result_out_1)|$(result_true_1)|
|![image](https://github.com/foamliu/Age-and-Gender/raw/master/images/2_raw.jpg)|![image](https://github.com/foamliu/Age-and-Gender/raw/master/images/2_img.jpg)|$(result_out_2)|$(result_true_2)|
|![image](https://github.com/foamliu/Age-and-Gender/raw/master/images/3_raw.jpg)|![image](https://github.com/foamliu/Age-and-Gender/raw/master/images/3_img.jpg)|$(result_out_3)|$(result_true_3)|
|![image](https://github.com/foamliu/Age-and-Gender/raw/master/images/4_raw.jpg)|![image](https://github.com/foamliu/Age-and-Gender/raw/master/images/4_img.jpg)|$(result_out_4)|$(result_true_4)|
|![image](https://github.com/foamliu/Age-and-Gender/raw/master/images/5_raw.jpg)|![image](https://github.com/foamliu/Age-and-Gender/raw/master/images/5_img.jpg)|$(result_out_5)|$(result_true_5)|
|![image](https://github.com/foamliu/Age-and-Gender/raw/master/images/6_raw.jpg)|![image](https://github.com/foamliu/Age-and-Gender/raw/master/images/6_img.jpg)|$(result_out_6)|$(result_true_6)|
|![image](https://github.com/foamliu/Age-and-Gender/raw/master/images/7_raw.jpg)|![image](https://github.com/foamliu/Age-and-Gender/raw/master/images/7_img.jpg)|$(result_out_7)|$(result_true_7)|
|![image](https://github.com/foamliu/Age-and-Gender/raw/master/images/8_raw.jpg)|![image](https://github.com/foamliu/Age-and-Gender/raw/master/images/8_img.jpg)|$(result_out_8)|$(result_true_8)|
|![image](https://github.com/foamliu/Age-and-Gender/raw/master/images/9_raw.jpg)|![image](https://github.com/foamliu/Age-and-Gender/raw/master/images/9_img.jpg)|$(result_out_9)|$(result_true_9)|

