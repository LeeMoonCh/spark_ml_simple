# 特征提取，转换以及选择
前三章，主要用来讲解spark ml的各种的术语以及相关基础知识。没有任何一个算法相关的讲解。这章，我们就会着手开始进行一些特征向量的算法讲解。包括特征向量的提取、转换以及选择。我们都知道，很多时候，我们的数据不可能直接就是ML需要的类型：特征向量。大多数情况下都是诸如：表数据、文本数据、图片数据等等，这些数据并不能直接应用到我们的ML上，而是要想办法进行特征提取，然后进行转换，最后在应用到一个算法上，进行模型建设和预测。

本章主要分为以下几个大类进行研究：

- Extraction：抽取，主要是用来从raw data(也就是原始数据)中将特征提取出来
- Transformation：转换，对特征向量进行扩展、转换和修改
- Selection：选择，在一个大特征向量中截取一部分特征
- Locality Sensitive Hashing(LSH)：局部敏感hash  @ https://blog.csdn.net/chichoxian/article/details/80290782



它们的子分类如下：

[Feature Extractors](http://spark.apache.org/docs/latest/ml-features.html#feature-extractors)

- [TF-IDF](http://spark.apache.org/docs/latest/ml-features.html#tf-idf)
- [Word2Vec](http://spark.apache.org/docs/latest/ml-features.html#word2vec)
- [CountVectorizer](http://spark.apache.org/docs/latest/ml-features.html#countvectorizer)
- [FeatureHasher](http://spark.apache.org/docs/latest/ml-features.html#featurehasher)

[Feature Transformers](http://spark.apache.org/docs/latest/ml-features.html#feature-transformers)

- [Tokenizer](http://spark.apache.org/docs/latest/ml-features.html#tokenizer)
- [StopWordsRemover](http://spark.apache.org/docs/latest/ml-features.html#stopwordsremover)
- [n](http://spark.apache.org/docs/latest/ml-features.html#n-gram)

- [-gram](http://spark.apache.org/docs/latest/ml-features.html#n-gram)
- [Binarizer](http://spark.apache.org/docs/latest/ml-features.html#binarizer)
- [PCA](http://spark.apache.org/docs/latest/ml-features.html#pca)
- [PolynomialExpansion](http://spark.apache.org/docs/latest/ml-features.html#polynomialexpansion)
- [Discrete Cosine Transform (DCT)](http://spark.apache.org/docs/latest/ml-features.html#discrete-cosine-transform-dct)
- [StringIndexer](http://spark.apache.org/docs/latest/ml-features.html#stringindexer)
- [IndexToString](http://spark.apache.org/docs/latest/ml-features.html#indextostring)
- [OneHotEncoder (Deprecated since 2.3.0)](http://spark.apache.org/docs/latest/ml-features.html#onehotencoder-deprecated-since-230)
- [OneHotEncoderEstimator](http://spark.apache.org/docs/latest/ml-features.html#onehotencoderestimator)
- [VectorIndexer](http://spark.apache.org/docs/latest/ml-features.html#vectorindexer)
- [Interaction](http://spark.apache.org/docs/latest/ml-features.html#interaction)
- [Normalizer](http://spark.apache.org/docs/latest/ml-features.html#normalizer)
- [StandardScaler](http://spark.apache.org/docs/latest/ml-features.html#standardscaler)
- [MinMaxScaler](http://spark.apache.org/docs/latest/ml-features.html#minmaxscaler)
- [MaxAbsScaler](http://spark.apache.org/docs/latest/ml-features.html#maxabsscaler)
- [Bucketizer](http://spark.apache.org/docs/latest/ml-features.html#bucketizer)
- [ElementwiseProduct](http://spark.apache.org/docs/latest/ml-features.html#elementwiseproduct)
- [SQLTransformer](http://spark.apache.org/docs/latest/ml-features.html#sqltransformer)
- [VectorAssembler](http://spark.apache.org/docs/latest/ml-features.html#vectorassembler)
- [VectorSizeHint](http://spark.apache.org/docs/latest/ml-features.html#vectorsizehint)
- [QuantileDiscretizer](http://spark.apache.org/docs/latest/ml-features.html#quantilediscretizer)
- [Imputer](http://spark.apache.org/docs/latest/ml-features.html#imputer)

[Feature Selectors](http://spark.apache.org/docs/latest/ml-features.html#feature-selectors)

- [VectorSlicer](http://spark.apache.org/docs/latest/ml-features.html#vectorslicer)
- [RFormula](http://spark.apache.org/docs/latest/ml-features.html#rformula)
- [ChiSqSelector](http://spark.apache.org/docs/latest/ml-features.html#chisqselector)

[Locality Sensitive Hashing](http://spark.apache.org/docs/latest/ml-features.html#locality-sensitive-hashing)

- LSH Operations
  - [Feature Transformation](http://spark.apache.org/docs/latest/ml-features.html#feature-transformation)
  - [Approximate Similarity Join](http://spark.apache.org/docs/latest/ml-features.html#approximate-similarity-join)
  - [Approximate Nearest Neighbor Search](http://spark.apache.org/docs/latest/ml-features.html#approximate-nearest-neighbor-search)
- LSH Algorithms
  - [Bucketed Random Projection for Euclidean Distance](http://spark.apache.org/docs/latest/ml-features.html#bucketed-random-projection-for-euclidean-distance)
  - [MinHash for Jaccard Distance](http://spark.apache.org/docs/latest/ml-features.html#minhash-for-jaccard-distance)



以上只是目录，我会在代码中对每个小章节进行代码以及原理讲解。