import jieba.analyse
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from gensim.models import KeyedVectors
import itertools
# 加载中文停用词表
def load_stopwords(filepath):
    stopwords = set()
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            stopwords.add(line.strip())
    return stopwords

# 文本预处理：去除标点符号、停用词、并进行分词
def preprocess_text(text, stopwords):
    # 使用jieba进行分词
    words = jieba.cut(text)
    # 去除停用词和标点符号
    cleaned_words = [word for word in words if word not in stopwords and len(word) > 1]
    # 返回处理后的文本
    return ' '.join(cleaned_words)

# 加载停用词
stopwords = load_stopwords('stopwordslist.txt')

# 读取原始文本
with open('示例文本.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# 对文本进行预处理
cleaned_text = preprocess_text(text, stopwords)

# 使用TextRank算法提取关键词
# topK指定返回的关键词数量，withWeight指定是否返回每个关键词的权重
keywords = jieba.analyse.textrank(cleaned_text, topK=5, withWeight=True)

# 保存关键词到'topic_word'文档
with open('topic_word_eva.txt', 'w', encoding='utf-8') as file:
    for item in keywords:
        word, weight = item
        print(f"关键词：{word}，权重：{weight:.2f}")
        file.write(f"{word} \n")

print("关键词已保存到'topic_word_eva'文档。")

with open('示例文本.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# 使用正则表达式匹配标点符号，并将它们作为分割点
# 移除空白字符串
sentences = [sentence for sentence in re.split(r'[。！]', text) if sentence.strip()]# 移除列表中的空字符串

# 停用词列表
stop_words = [line.strip() for line in open('stopwordslist.txt', encoding='UTF-8').readlines()]

# 分词函数（使用jieba分词）
def tokenize(text):
    return [word for word in jieba.cut(text) if word not in stop_words]

# 预处理函数：去除标点符号和停用词
def preprocess_text(text):
    return ''.join([char for char in text if char not in stop_words])

# 预处理文档
preprocessed_documents = [preprocess_text(doc) for doc in sentences]

# 创建TF-IDF向量化器，使用自定义的分词函数
tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize)

# 计算TF-IDF特征矩阵
tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_documents)

# 获取特征词汇列表
feature_names = tfidf_vectorizer.get_feature_names_out()

# 选择关键词及其置信度
keywords_with_confidence = {}
for i, doc in enumerate(preprocessed_documents):
    feature_index = tfidf_matrix[i,:].nonzero()[1]
    tfidf_scores = zip(feature_index, [tfidf_matrix[i, x] for x in feature_index])
    sorted_tfidf_scores = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)[:3]  # 仅保留置信度排名前3的关键词
    keywords_with_confidence[i] = [(feature_names[i], score) for i, score in sorted_tfidf_scores]

# 打印关键词及其置信度
for doc_index, keywords in keywords_with_confidence.items():
    print(f"文档 {doc_index + 1}:")
    for keyword, confidence in keywords:
        print(f"{keyword}: {confidence}")
    print()

# 将关键词保存到'key_word'文档
with open('key_word.txt_eva', 'w', encoding='utf-8') as file:
    for doc_index, keywords in keywords_with_confidence.items():
        for i, (keyword, confidence) in enumerate(keywords):
            file.write(keyword + ' ')
            if (i + 1) % 3 == 0:  # 每三个关键词后换行
                file.write('\n')
        file.write('\n')

print("关键词已保存到'key_word_eva'文档。")

with open('key_word_eva.txt', 'r', encoding='utf-8') as key_file:
    # 读取所有行并去除空行
    keyword_lines = [line.strip() for line in key_file.readlines() if line.strip()]
    # 将所有关键词合并到一个列表中
    all_keywords = list(itertools.chain.from_iterable([line.split() for line in keyword_lines]))

# 从文件中读取主题和代表句子
with open('topic_word_eva.txt', 'r', encoding='utf-8') as topic_file:
    # 读取所有行并去除空行
    topic_lines = [line.strip() for line in topic_file.readlines() if line.strip()]
    # 将每行转换为关键词列表
    topic_keywords = [line.split() for line in topic_lines]

file = r'\tencent-ailab-embedding-zh-d200-v0.2.0-s\tencent-ailab-embedding-zh-d200-v0.2.0-s.txt'
wv_from_text = KeyedVectors.load_word2vec_format(file, binary=False)
wv_from_text.fill_norms(force=False)

def calculate_similarity(word1, word2, model):

    # 确保两个词都在模型的词汇表中
    if word1 in model.key_to_index and word2 in model.key_to_index:
        # 使用 gensim 的 similarity 方法计算余弦相似度
        similarity = model.similarity(word1, word2)
        # 将相似度从 [-1, 1] 映射到 [0, 1]
        transformed_similarity = (similarity + 1) / 2
        return transformed_similarity
    else:
        # 如果有一个词不在模型中，返回默认值 0.5
        return 0.5

# 计算整段文本的关键词与每个主题词的平均相似度
average_similarity_scores = {}

# 遍历每个主题词
for i, topic in enumerate(topic_keywords, start=1):
    total_similarity = 0
    # 计算所有关键词与当前主题词的相似度
    for keyword in all_keywords:
        # 遍历每个主题词与关键词的组合
        for topic_word in topic:
            similarity = calculate_similarity(keyword, topic_word, wv_from_text)
            total_similarity += similarity
    # 计算当前主题的平均相似度
    average_similarity = total_similarity / (len(all_keywords) * len(topic))
    average_similarity_scores[f"主题 {i}"] = average_similarity

# 找出相似度最大的主题
max_topic = max(average_similarity_scores, key=average_similarity_scores.get)
max_similarity = average_similarity_scores[max_topic]

print(f"\n相似度最大的主题是: {max_topic}, 相似度: {max_similarity:.4f}")