---
title: NLTK 自然语言处理
toc: true
description: ' '
date: 2019-03-06 14:23:15
tags:
categories:
---

# NLTK 自然语言处理

## 统计

```python
from nltk import FreqDist
# 需要剔除空格和停用词
FreqDist(all_words_list) 
stops = stopwords.words('english')+ list(string.punctuation) + ['','The','This',1,2,3,4,5,6,7,8,9,0]
clean_words = [token for token in all_words_list if token not in stops ]
fdist = FreqDist(clean_words)
plt.figure(figsize=(16,6))
fdist.plot(50,cumulative=True) # 词频累积图
fdist.most_common(20)
# 词频统计图
plt.figure(figsize=(16,6))
fdist.plot(50,cumulative=False)
```

根据词频可以生成词云

```python
from wordcloud import WordCloud

wordcloud = WordCloud(
        background_color="white", #设置背景为白色，默认为黑色
        width=2600,             
        height=1300,              
        margin=10               
        ).generate_from_frequencies(fdist)
plt.figure(figsize=(16,6))
plt.imshow(wordcloud,interpolation="bilinear")
plt.axis("off")
plt.show()
```



## 分词

```python
from nltk.tokenize import word_tokenize
text = "demo text"
text_tokenized = [word for word in word_tokenize(text)]
```

## 规范化 （*normalize*）

### 去除停用词及标点

```python
from nltk.corpus import stopwords
import string
stops = stopwords.words('english') + list(string.punctuation)
clean_text = [word for word in text_tokenized if word not in stops]
```

## 词性标注

```python
from nltk.tag import pos_tag
# 第一次使用可能需要下载
nltk.download('averaged_perceptron_tagger') # 默认tagset
nltk.download('universal_tagset')
tagged = pos_tag(clean_text, tagset='universal')
# nltk.help.upenn_tagset() 查看词性说明
```

```
CC	Coordinating conjunction
CD	Cardinal number
DT	Determiner
EX	Existential there
FW	Foreign word
IN	Preposition/subord. conjunction
JJ	Adjective
JJR	Adjective, comparative
JJS	Adjective, superlative
LS	List item marker
MD	Modal
NN	Noun, singular or masps
NNP	Proper noun, singular
NNPS	Proper noun plural
NNS	Noun, plural
PDT	Predeterminer
POS	Possessive ending
PRP	Personal pronoun
PRP$	Possessive pronoun
RB	Adverb
RBR	Adverb, comparative
RBS	Adverb, superlative
RP	Particle
SYM	Symbol (mathematical or scientific)
TO	to
UH	Interjection
VB	Verb, base form
VBD	Verb, past tense
VBG	Verb, gerund/present participle
VBN	Verb, past participle
VBP	Verb, non-3rd ps. sing. present
VBZ	Verb,3rd ps. sing. present
WDT	wh-determiner
WP	wh-pronoun
WP$	Possessive wh-pronoun
WRB	wh-adverb
#	pound sign (currency marker)
$	dollar sign (currency marker)
''	close quote
(	open parenthesis
)	close parenthesis
,	comma
.	period
:	colon
``	open quote
```

### 词干提取

按照一定的规则剥离词缀

```python
porter = nltk.PorterStemmer()
lancaster = nltk.LancasterStemmer()
[porter.stem(t) for t in tokens]
[lancaster.stem(t) for t in tokens]
```

### 词形归并（还原）

> [不同库的类似方法](https://www.machinelearningplus.com/nlp/lemmatization-examples-python/)

如将 women 转换为 woman

```python
wnl = nltk.WordNetLemmatizer()
[wnl.lemmatize(t) for t in tokens]
```

为`lemmatize`中的`pos`参数赋值词性标注，可以提高词形还原的准确性。

需要将 pos_tagger 的标注转换为 wordnet 能够识别的四种 "syntactic categories" 之一：

> Syntactic category: n for noun, v for verb, a for adjective, r for adverb.
>
> [参考 stackoverflow](https://stackoverflow.com/questions/25534214/nltk-wordnet-lemmatizer-shouldnt-it-lemmatize-all-inflections-of-a-word/25544239#25544239)

```python
# e 表示标注后的元组中的词性标注符号， e[0] 表示词性标注符号的第一个字母
# 这个好像会报错
word_net_pos = lambda e: ('a' if e[0].lower() == 'j' else e[0].lower()) if e[0].lower() in ['n', 'r', 'v'] else 'n'

# 或
def wn_pos(nltk_tag):
    import nltk.corpus.reader.wordnet
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

```



## 信息抽取

标记模式：一个用尖括号分隔的词性标记序列，如`<DT>?<JJ>*<NN>` ， 类似正则表达式，又如  `<DT>?<JJ.*>*<NN.*>+` ，其中`JJ.*` 表示 零个或多个任何类型的形容词，如 相对形容词` JJR`

图形界面`nltk.app.chunkparser()`可用于测试标记模式 。使用此工具提供的帮助资料继续完善你的标记模式 

## 关键词抽取

rake-nltk

```python
from rake_nltk import Rake

# Uses stopwords for english from NLTK, and all puntuation characters by default
r = Rake()

# Extraction given the text.
r.extract_keywords_from_text(<text to process>)

# Extraction given the list of strings where each string is a sentence.
r.extract_keywords_from_sentences(<list of sentences>)

# To get keyword phrases ranked highest to lowest.
r.get_ranked_phrases()

# To get keyword phrases ranked highest to lowest with scores.
r.get_ranked_phrases_with_scores()
```

