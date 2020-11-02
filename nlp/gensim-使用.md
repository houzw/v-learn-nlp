---
title: gensim 使用
toc: true
description: ' '
date: 2019-02-22 15:27:51
tags:
categories:
---

# gensim 

用于主题建模的python库，支持包括 TF-IDF，LSA，LDA，word2vec 在内的多种主题模型算法， 支持流式训练，并提供诸如相似度计算，信息检索等一些常用任务的API接口

https://radimrehurek.com/gensim/

>- Scalable statistical semantics
>
>- Analyze plain-text documents for semantic structure
>
>- Retrieve semantically similar documents

## 基本概念

1. 语料 Corpus：一组电子文档的集合，原始训练语料，用于模型的输入和文档的组织，corpus的每一个元素对应一篇文档。语料在输入模型之前需要进行预处理（分词，去停用词等），转换为gensim使用的稀疏向量。
2. 向量 vector：由一组文本特征构成的列表。是一段文本在Gensim中的内部表达。
3. 稀疏向量 SparseVector：
4. 模型 model

## 语料预处理

1. 主要是分词（tokenizing）、去除停用词、规范化文本（转为小写等）、词形归并、词干提取（stemming）等（简单语料只需进行前两项即可）

```python
from nltk.tokenize import word_tokenize
# 分词 texts_tokenized = [[word for word in nltk.word_tokenize(testtext)]]
from nltk.corpus import stopwords
import string
stopwords = stopwords.words('english')+ list(string.punctuation) 
#去停用词和标点 texts_filtered = [[word for word in doc if not word in stopwords] for doc in texts_tokenized]
from nltk.stem.lancaster import LancasterStemmer
# 词干化 LancasterStemmer().stem(word)
```

2. 建立语料特征（此处是word）的索引字典，并将文本特征的原始表达转化成词袋模型对应的稀疏向量的表达。

    ```python
    from gensim import corpora
    dictionary = corpora.Dictionary(texts)
    # dic.save('test.dict) # save to disk
    bow_corpus = [dictionary.doc2bow(text) for text in texts_filtered]
    # corpora.MmCorpus.serialize('bow_corpus.mm',bow_corpus) # save to disk
    ```

     向量中第一个元素表示字典中单词 id，第二个表示该单词在某文档（本例中为text）中的词频

    *注意*：对于大型语料库，Gensim支持文档的流式处理以优化内存，即迭代读取本地文件返回稀疏向量

## 模型

TF-IDF, LDA，LSI，RP，HDP模型等

```python
from gensim import models
tfidf = models.TfidfModel(bow_corpus)
# 完成对corpus中出现的每一个特征的IDF值的统计工作,建立TF-IDF模型,把词袋表达的向量转换到另一个向量空间
test_string = "pit remove"
test_bow = 
```

## 文档相似度

> [LSI/LSA算法原理与实践Demo](LSI/LSA算法原理与实践Demo)

```python
# 构造LSI模型并将待检索的query和文本转化为LSI主题向量
# 转换之前的corpus和query均是BOW向量
lsi_model = models.LsiModel(corpus, id2word=dictionary,          num_topics=2)
documents = lsi_model[corpus]
query_vec = lsi_model[query]

```



---

>https://github.com/RaRe-Technologies/gensim/tree/develop/docs/notebooks
>
>[15分钟入门NLP神器—Gensim](https://blog.csdn.net/l7h9ja4/article/details/80220939)
>
>[用NLTK对英文语料做预处理，用gensim计算相似度](https://blog.csdn.net/sinat_36972314/article/details/79004305)
>
>https://radimrehurek.com/gensim/tut1.html
>
>https://radimrehurek.com/gensim/apiref.html
>
>gensim学习笔记[（一）](https://blog.csdn.net/John_xyz/article/details/54731403)、[（二）](https://blog.csdn.net/John_xyz/article/details/54744413)、[（三）计算文档间相似性](https://blog.csdn.net/John_xyz/article/details/54747773)
>
>[Gensim Tutorial – A Complete Beginners Guide](https://www.machinelearningplus.com/nlp/gensim-tutorial/)

https://radimrehurek.com/gensim/tut3.html （Similarity Queries）

[python根据关键词实现信息检索推荐（使用深度学习算法）](https://blog.csdn.net/sinat_29673403/article/details/80422953)

[Topic Modelling in Python with NLTK and Gensim](https://towardsdatascience.com/topic-modelling-in-python-with-nltk-and-gensim-4ef03213cd21)

https://sematext.com/blog/word-embeddings-gensim-word2vec-tutorial/

[How to Develop Word Embeddings in Python with Gensim](https://machinelearningmastery.com/develop-word-embeddings-python-gensim/)

https://rare-technologies.com/doc2vec-tutorial/

https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-IMDB.ipynb

[基于gensim的Doc2Vec简析](https://blog.csdn.net/lenbow/article/details/52120230)

[基于gensim的doc2vec实践](https://blog.csdn.net/John_xyz/article/details/79424284)

https://github.com/jarvisqi/nlp_learning/blob/master/gensim/text_similarity.py

[Doc2Vec Searching for both similar words and labels](https://github.com/RaRe-Technologies/gensim/issues/1397) 



```python
word_vec = model['word']
model.docvecs.most_similar([word_vec])
# returns similar labels
```

> [Doc2Vec Get most similar documents](https://stackoverflow.com/questions/42781292/doc2vec-get-most-similar-documents)

```python
tokens = "a new sentence to match".split()

new_vector = model.infer_vector(tokens)
sims = model.docvecs.most_similar([new_vector])
```

[用 Doc2Vec 得到文档／段落／句子的向量表达](https://cloud.tencent.com/developer/article/1083546)

使用预训练的模型建立word2vec模型

https://datascience.stackexchange.com/questions/10695/how-to-initialize-a-new-word2vec-model-with-pre-trained-model-weights



https://stackoverflow.com/questions/22129943/how-to-calculate-the-sentence-similarity-using-word2vec-model-of-gensim-with-pyt

```python
s1 = 'This room is dirty'
s2 = 'dirty and disgusting room' #corrected variable name

distance = model.wv.n_similarity(s1.lower().split(), s2.lower().split())
```



## Using TSNE to Plot a Subset of Similar Words from Word2Vec

>https://medium.com/@aneesha/using-tsne-to-plot-a-subset-of-similar-words-from-word2vec-bb8eeaea6229

I needed to display a spatial map (i.e., scatterplot) with similar words from Word2Vec. I could only find code that would display the all the words or an indexed subset using either TSNE or PCA. I’m sharing the Python code I wrote as a Gist. The code uses the fantastic gensim library as it provides easy access to the raw word vectors and a great api to perform similarity queries. The code performs the following tasks:

1. Loads a pre-trained word2vec embedding
2. Finds similar words and appends each of the similar words embedding vector to the matrix
3. Applies TSNE to the Matrix to project each word to a 2D space (i.e. dimension reduction)
4. Plots the 2D position of each word with a label

```python
# uncomment if gensim is installed
#!pip install gensim
import gensim
# Need the interactive Tools for Matplotlib
%matplotlib notebook
import numpy as np
import matplotlib.pyplot as plt
 
from sklearn.manifold import TSNE
# load pre-trained word2vec embeddings
# The embeddings can be downloaded from command prompt:
# wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
# Test the loaded word2vec model in gensim
# We will need the raw vector for a word
print(model['computer']) 

# We will also need to get the words closest to a word
model.similar_by_word('computer')

def display_closestwords_tsnescatterplot(model, word):
    
    arr = np.empty((0,300), dtype='f')
    word_labels = [word]

    # get close words
    close_words = model.similar_by_word(word)
    
    # add the vector for each of the closest words to the array
    arr = np.append(arr, np.array([model[word]]), axis=0)
    for wrd_score in close_words:
        wrd_vector = model[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)
        
    # find tsne coords for 2 dimensions
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    # display scatter plot
    plt.scatter(x_coords, y_coords)

    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
    plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
    plt.show()

display_closestwords_tsnescatterplot(model, 'tasty')
```



Figure 1 shows the words most similar to “Madonna”. No surprise that “Lady_Gaga” shows up. As gensim can load Glove pre-trained vectors, the code can easily be adapted to support Glove as well. I’ll be using the code in a follow up blog post on adding lexicon knowledge to an embedding. Have Fun!

## [Multi-Class Text Classification with Doc2Vec and t-SNE, a full tutorial.](https://medium.com/@morga046/multi-class-text-classification-with-doc2vec-and-t-sne-a-full-tutorial-55eb24fc40d3)

https://github.com/ExtraLime/medium-doc2vec





情感分析

[Sentiment Classification in 5 classes Doc2Vec](https://www.kaggle.com/tj2552/sentiment-classification-in-5-classes-doc2vec)

[现代情感分析方法](http://python.jobbole.com/87811/)

聚类

https://ai.intelligentonlinetools.com/ml/text-clustering-doc2vec-word-embedding-machine-learning/

https://www.ioiogoo.cn/2018/05/31/%e4%bd%bf%e7%94%a8k-means%e5%8f%8atf-idf%e7%ae%97%e6%b3%95%e5%af%b9%e4%b8%ad%e6%96%87%e6%96%87%e6%9c%ac%e8%81%9a%e7%b1%bb%e5%b9%b6%e5%8f%af%e8%a7%86%e5%8c%96/

https://www.jianshu.com/p/2aaf1a94b7d6

---

https://github.com/dominiek/word2vec-explorer

https://github.com/pvthuy/word2vec-visualization





https://aivic.github.io/project/Quora-question-pair-similarity/