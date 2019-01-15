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

#|图片大小|网络|性别准确度(验证集)|年龄TOP-5准确度(验证集)|
|---|---|---|---|---|
|1|96x112|ResNet-18|89.764%|18.060%|
|2|224x224|ResNet-50|90.034%|19.325%|


### Demo
```bash
$ python demo.py
```

效果图如下：

原图 | 校准 | 识别 | 标注 |
|---|---|---|---|
|![image](https://github.com/foamliu/Joint-Estimation-of-Age-and-Gender/raw/master/images/0_raw.jpg)|![image](https://github.com/foamliu/Joint-Estimation-of-Age-and-Gender/raw/master/images/0_img.jpg)|性别：男, 年龄：36|性别：男, 年龄：35|
|![image](https://github.com/foamliu/Joint-Estimation-of-Age-and-Gender/raw/master/images/1_raw.jpg)|![image](https://github.com/foamliu/Joint-Estimation-of-Age-and-Gender/raw/master/images/1_img.jpg)|性别：女, 年龄：27|性别：女, 年龄：31|
|![image](https://github.com/foamliu/Joint-Estimation-of-Age-and-Gender/raw/master/images/2_raw.jpg)|![image](https://github.com/foamliu/Joint-Estimation-of-Age-and-Gender/raw/master/images/2_img.jpg)|性别：女, 年龄：25|性别：女, 年龄：50|
|![image](https://github.com/foamliu/Joint-Estimation-of-Age-and-Gender/raw/master/images/3_raw.jpg)|![image](https://github.com/foamliu/Joint-Estimation-of-Age-and-Gender/raw/master/images/3_img.jpg)|性别：男, 年龄：28|性别：男, 年龄：22|
|![image](https://github.com/foamliu/Joint-Estimation-of-Age-and-Gender/raw/master/images/4_raw.jpg)|![image](https://github.com/foamliu/Joint-Estimation-of-Age-and-Gender/raw/master/images/4_img.jpg)|性别：男, 年龄：39|性别：男, 年龄：55|
|![image](https://github.com/foamliu/Joint-Estimation-of-Age-and-Gender/raw/master/images/5_raw.jpg)|![image](https://github.com/foamliu/Joint-Estimation-of-Age-and-Gender/raw/master/images/5_img.jpg)|性别：男, 年龄：27|性别：男, 年龄：10|
|![image](https://github.com/foamliu/Joint-Estimation-of-Age-and-Gender/raw/master/images/6_raw.jpg)|![image](https://github.com/foamliu/Joint-Estimation-of-Age-and-Gender/raw/master/images/6_img.jpg)|性别：女, 年龄：28|性别：女, 年龄：31|
|![image](https://github.com/foamliu/Joint-Estimation-of-Age-and-Gender/raw/master/images/7_raw.jpg)|![image](https://github.com/foamliu/Joint-Estimation-of-Age-and-Gender/raw/master/images/7_img.jpg)|性别：男, 年龄：27|性别：男, 年龄：29|
|![image](https://github.com/foamliu/Joint-Estimation-of-Age-and-Gender/raw/master/images/8_raw.jpg)|![image](https://github.com/foamliu/Joint-Estimation-of-Age-and-Gender/raw/master/images/8_img.jpg)|性别：男, 年龄：28|性别：女, 年龄：34|
|![image](https://github.com/foamliu/Joint-Estimation-of-Age-and-Gender/raw/master/images/9_raw.jpg)|![image](https://github.com/foamliu/Joint-Estimation-of-Age-and-Gender/raw/master/images/9_img.jpg)|性别：女, 年龄：25|性别：女, 年龄：6|

