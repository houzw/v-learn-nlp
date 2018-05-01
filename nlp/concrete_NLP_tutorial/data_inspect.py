#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: houzhiwei
# time: 2018/4/29 22:14

# 高级神经网络 API https://keras.io/zh/
import keras
import nltk
import pandas as pd
import numpy as np
import re
# 字符编码转换 https://docs.python.org/3/library/codecs.html
import codecs

input_file_name = "socialmedia_relevant_cols.csv"
output_file_name = "socialmedia_relevant_cols_clean.csv"
# 数据清洗
# 输入。使用codecs.open打开，自动转换为指定编码
input_file = codecs.open(input_file_name, "r", encoding="utf-8", errors="replace")
# 输出
output_file = open(output_file_name, "w")


def santitize_characters(raw, clean):
    """
    清理不相关的字符、单词
    :param raw: 原始
    :param clean: 清理后的
    :return:
    """
    for line in input_file:
        out = line
        output_file.write(line)


santitize_characters(input_file, output_file)

questions = pd.read_csv(output_file_name)
questions.columns = ['text', 'choose_one', 'class_label']
questions.head()

