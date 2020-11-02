---
title: 小V 学 NLP
toc: true
description: ' '
date: 2018-05-14 14:39:44
tags:
categories:
---

## 主题抽取

[如何用 Python 从海量文本抽取主题？](https://www.leiphone.com/news/201707/Pe5LRySEwvi6vKiA.html)

## 词云

[如何用Python做词云？](https://www.jianshu.com/p/e4b24a734ccc)

## NER 命名实体识别

[达观数据：如何打造一个中文NER系统](http://zhuanlan.51cto.com/art/201705/540693.htm)

[NLP项目：使用NLTK和SpaCy进行命名实体识别](http://www.atyun.com/27129.html)

## 实体链接

## 信息抽取

https://blog.csdn.net/zzulp/article/details/77414113

https://zhuanlan.zhihu.com/p/39205829

简单网页抽取可使用   https://github.com/postlight/mercury-parser

- Preprocessor/Tokenizer: split story into units and ultimately word tokens
- Gazetteer: lexical look-up of important (to your task) words/phrases
- POS tagger: tag/disambiguate words w.r.t. parts of speech
- Chunk parser: find basic noun and verb phrases
- Named entity tagger: identify and classify proper names
- Relationship tagger: find relations between entities
- Event Template Analyzer: populate fact/event templates 

## 文本预处理（清理）

https://www.zhihu.com/question/268849350

1. Normalization  ——小写

2. Tokenization —— 分词（英文可以简单的以空格分隔）nltk `word_tokenize()`/`sent_tokenize()`(分句)

3. Stop words ——去停用词

    ```python
    from nltk.corpus import stopwords
    clean_text = [word for word in text_tokenized if word not in stopwords.words('english')]
    ```

4. Part-of-Speech Tagging —— 词性标注

5. Named Entity Recognition —— 命名实体识别 （`ne_chunk`）

    ```python
    from nltk.tag import pos_tag
    from nltk import ne_chunk
    from nltk.tokenize import word_tokenize
    ne_chunk(pos_tag(word_tokenize('text')))
    ```

6. Stemming and Lemmatization ——规范化（词干，还原词形）

    ```python
    from nltk.stem.wordnet import WordNetLemmatizer
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    ```

    

