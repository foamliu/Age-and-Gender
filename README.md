# 同时识别年龄与性别
基于PyTorch 实现多任务学习，在同时识别年龄与性别。


## 数据集

[IMDB-WIKI](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) 数据集，460,723张图片。

![image](https://github.com/foamliu/Joint-Estimation-of-Age-and-Gender/raw/master/images/imdb-wiki-teaser.png)

这里为简洁只使用 IMDB 数据集。

### 年龄分布：

460723张照片为平衡每个年龄最多只保存5000张，清洗后得到163065张，按年龄分布作图：

![image](https://github.com/foamliu/Joint-Estimation-of-Age-and-Gender/raw/master/images/age.png)

### 样例数据

1 | 2 | 3 | 4 |
|---|---|---|---|
|![image](https://github.com/foamliu/Joint-Estimation-of-Age-and-Gender/raw/master/images/0_img.jpg)  | ![image](https://github.com/foamliu/Joint-Estimation-of-Age-and-Gender/raw/master/images/1_img.jpg) |![image](https://github.com/foamliu/Joint-Estimation-of-Age-and-Gender/raw/master/images/2_img.jpg)| ![image](https://github.com/foamliu/Joint-Estimation-of-Age-and-Gender/raw/master/images/3_img.jpg) |
|性别：女, 年龄：29|性别：女, 年龄：36|性别：男, 年龄：27|性别：女, 年龄：18|
|![image](https://github.com/foamliu/Joint-Estimation-of-Age-and-Gender/raw/master/images/4_img.jpg)  | ![image](https://github.com/foamliu/Joint-Estimation-of-Age-and-Gender/raw/master/images/5_img.jpg) |![image](https://github.com/foamliu/Joint-Estimation-of-Age-and-Gender/raw/master/images/6_img.jpg)| ![image](https://github.com/foamliu/Joint-Estimation-of-Age-and-Gender/raw/master/images/7_img.jpg) |
|性别：男, 年龄：26|性别：男, 年龄：60|性别：男, 年龄：41|性别：男, 年龄：44|
|![image](https://github.com/foamliu/Joint-Estimation-of-Age-and-Gender/raw/master/images/8_img.jpg)  | ![image](https://github.com/foamliu/Joint-Estimation-of-Age-and-Gender/raw/master/images/9_img.jpg) |
|性别：男, 年龄：52|性别：女, 年龄：25|
