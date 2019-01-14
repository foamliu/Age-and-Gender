# -*- coding: utf-8 -*-
import json

if __name__ == '__main__':
    with open('README.t', 'r', encoding="utf-8") as file:
        text = file.readlines()
    text = ''.join(text)

    with open('sample_preds.json', 'r', encoding="utf-8") as file:
        results = json.load(file)

    for i in range(10):
        gender = results[i]['gen_true']
        if gender == 0:
            gender = '女'
        else:
            gender = '男'
        age = results[i]['age_true']
        text = text.replace('$(result_true_{})'.format(i), '性别：{}, 年龄：{}'.format(gender, age))

        gender = results[i]['gen_out']
        if gender == 0:
            gender = '女'
        else:
            gender = '男'
        age = results[i]['age_out']
        text = text.replace('$(result_out_{})'.format(i), '性别：{}, 年龄：{}'.format(gender, age))

    with open('README.md', 'w', encoding="utf-8") as file:
        file.write(text)
