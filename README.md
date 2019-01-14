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

### Demo
```bash
$ python demo.py
```

效果图如下：

原图 | 校准 | 识别 | 标注 |
|---|---|---|---|
|![image](https://github.com/foamliu/Joint-Estimation-of-Age-and-Gender/raw/master/images/0_raw.jpg)|![image](https://github.com/foamliu/Joint-Estimation-of-Age-and-Gender/raw/master/images/0_img.jpg)|性别：女, 年龄：30|性别：女, 年龄：38|
|![image](https://github.com/foamliu/Joint-Estimation-of-Age-and-Gender/raw/master/images/1_raw.jpg)|![image](https://github.com/foamliu/Joint-Estimation-of-Age-and-Gender/raw/master/images/1_img.jpg)|性别：男, 年龄：38|性别：男, 年龄：51|
|![image](https://github.com/foamliu/Joint-Estimation-of-Age-and-Gender/raw/master/images/2_raw.jpg)|![image](https://github.com/foamliu/Joint-Estimation-of-Age-and-Gender/raw/master/images/2_img.jpg)|性别：女, 年龄：29|性别：女, 年龄：23|
|![image](https://github.com/foamliu/Joint-Estimation-of-Age-and-Gender/raw/master/images/3_raw.jpg)|![image](https://github.com/foamliu/Joint-Estimation-of-Age-and-Gender/raw/master/images/3_img.jpg)|性别：男, 年龄：36|性别：男, 年龄：49|
|![image](https://github.com/foamliu/Joint-Estimation-of-Age-and-Gender/raw/master/images/4_raw.jpg)|![image](https://github.com/foamliu/Joint-Estimation-of-Age-and-Gender/raw/master/images/4_img.jpg)|性别：女, 年龄：25|性别：女, 年龄：50|
|![image](https://github.com/foamliu/Joint-Estimation-of-Age-and-Gender/raw/master/images/5_raw.jpg)|![image](https://github.com/foamliu/Joint-Estimation-of-Age-and-Gender/raw/master/images/5_img.jpg)|性别：女, 年龄：25|性别：女, 年龄：16|
|![image](https://github.com/foamliu/Joint-Estimation-of-Age-and-Gender/raw/master/images/6_raw.jpg)|![image](https://github.com/foamliu/Joint-Estimation-of-Age-and-Gender/raw/master/images/6_img.jpg)|性别：女, 年龄：30|性别：女, 年龄：30|
|![image](https://github.com/foamliu/Joint-Estimation-of-Age-and-Gender/raw/master/images/7_raw.jpg)|![image](https://github.com/foamliu/Joint-Estimation-of-Age-and-Gender/raw/master/images/7_img.jpg)|性别：男, 年龄：39|性别：男, 年龄：61|
|![image](https://github.com/foamliu/Joint-Estimation-of-Age-and-Gender/raw/master/images/8_raw.jpg)|![image](https://github.com/foamliu/Joint-Estimation-of-Age-and-Gender/raw/master/images/8_img.jpg)|性别：男, 年龄：27|性别：男, 年龄：28|
|![image](https://github.com/foamliu/Joint-Estimation-of-Age-and-Gender/raw/master/images/9_raw.jpg)|![image](https://github.com/foamliu/Joint-Estimation-of-Age-and-Gender/raw/master/images/9_img.jpg)|性别：女, 年龄：31|性别：男, 年龄：23|

