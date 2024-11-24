import jieba.analyse
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import itertools
import statistics
from gensim.models import KeyedVectors

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
with open('topic_word_seg.txt', 'w', encoding='utf-8') as file:
    for item in keywords:
        word, weight = item
        print(f"关键词：{word}，权重：{weight:.2f}")
        file.write(f"{word} \n")

print("关键词已保存到'topic_word_seg'文档。")

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
with open('key_word.txt_seg', 'w', encoding='utf-8') as file:
    for doc_index, keywords in keywords_with_confidence.items():
        for i, (keyword, confidence) in enumerate(keywords):
            file.write(keyword + ' ')
            if (i + 1) % 3 == 0:  # 每三个关键词后换行
                file.write('\n')
        file.write('\n')

print("关键词已保存到'key_word_seg'文档。")

with open('key_word_seg.txt ', 'r', encoding='utf-8') as key_file:
    # 读取所有行并去除空行
    keyword_lines = [line.strip() for line in key_file.readlines() if line.strip()]
    # 将每行转换为关键词列表
    sentence_keywords = [line.split() for line in keyword_lines]


# 从文件中读取主题和代表句子
with open('topic_word_seg.txt', 'r', encoding='utf-8') as topic_file:
    # 读取所有行并去除空行
    topic_lines = [line.strip() for line in topic_file.readlines() if line.strip()]
    # 将每行转换为关键词列表
    topic_representative_sentences = {line: line for line in topic_lines}
    topic_keywords = list(topic_representative_sentences.keys())

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

# 计算并存储每个句子中的关键词与每个主题词之间的相似度
similarity_scores = {}

for i, keywords in enumerate(sentence_keywords, start=1):
    topic_similarities = {topic: [] for topic in topic_keywords}
    for keyword in keywords:
        for topic in topic_keywords:
            similarity = calculate_similarity(keyword, topic, wv_from_text)
            topic_similarities[topic].append(similarity)
    similarity_scores[i] = topic_similarities


# 初始化topic_confidences字典，键为topic_keywords中的主题词，值为空列表
topic_confidences = {topic: [] for topic in topic_keywords}

# 计算方法
def calculate_confidence(scores, method):
    if method == 1:
        # 方法1：选取最大值
        return max(scores)
    elif method == 2:
        # 方法2：选取中间值
        return sorted(scores)[1]
    elif method == 3:
        # 方法3：计算平均值
        return sum(scores) / len(scores)

# 选择计算方法
method = 3  # 示例使用第3种方法

# 根据similarity_scores填充topic_confidences
for doc_id, topics in similarity_scores.items():
    for topic, scores in topics.items():
        confidence = calculate_confidence(scores, method)
        topic_confidences[topic].append(confidence)

# 计算每个主题词对应的句子概率列表的方差
variance_confidences = {}
for topic, confidences in topic_confidences.items():
    if len(confidences) > 1:
        variance = statistics.variance(confidences)
    else:
        variance = 0.0  # 如果只有一个置信度值，方差设为0
    variance_confidences[topic] = variance

# 找出概率方差最大的主题词
max_variance_topic = max(variance_confidences, key=variance_confidences.get)

# 获取概率方差最大的主题词对应的概率列表
probabilities = topic_confidences[max_variance_topic]

# # 输出每个主题词及其对应的句子概率列表
# for topic in topic_keywords:
#     print(f"{topic} = {topic_confidences[topic]}")

def find_best_split(probabilities):
    n = len(probabilities)
    best_avg_prob = 0
    best_split = 0

    # 只考虑第一句与最后一句的情况
    for split_point in range(1, n - 1):
        # 计算左侧第一句的概率
        left_prob = probabilities[0]
        # 计算右侧最后一句的概率
        right_prob = probabilities[split_point]

        # 计算整体平均概率
        total_avg_prob = (left_prob + right_prob) / 2

        # 更新最佳分割点和最佳平均概率
        if total_avg_prob > best_avg_prob:
            best_avg_prob = total_avg_prob
            best_split = split_point

    return best_split, best_avg_prob

# 找到最佳分割点和最佳平均概率
best_split, best_avg_prob = find_best_split(probabilities)
def find_best_splits(probabilities, num_splits):
    n = len(probabilities)
    best_avg_prob = 0
    best_splits = []

    # 生成所有可能的分割点组合
    for split_points in itertools.combinations(range(1, n), num_splits - 1):
        split_points = tuple(sorted((0,) + split_points + (n,)))
        segments = [(probabilities[split_points[i]], probabilities[split_points[i+1]-1]) for i in range(num_splits)]

        # 计算每一段的平均概率
        segment_avg_probs = [sum(segment) / 2 for segment in segments]

        # 计算整体平均概率
        total_avg_prob = sum(segment_avg_probs) / num_splits

        # 更新最佳分割点和最佳平均概率
        if total_avg_prob > best_avg_prob:
            best_avg_prob = total_avg_prob
            best_splits = split_points

    return best_splits, best_avg_prob

def print_results(num_sentences, best_splits):
    print(f"文本共有 {num_sentences} 句。")
    print(f"最佳分割点为: {best_splits[1:-1]}，整体平均概率为{best_avg_prob:.2f}")
    print(f"最佳主题数为: {best_num_splits}")
    # 计算分段情况
    segments = [best_splits[i] - best_splits[i - 1] for i in range(1, len(best_splits))]
    print(f"输出列表为 {segments} 。")

# 遍历主题数从2到6
best_num_splits = 0
best_avg_prob = 0
best_splits = []

for num_splits in range(2, 7):
    splits, avg_prob = find_best_splits(probabilities, num_splits)
    if avg_prob > best_avg_prob:
        best_avg_prob = avg_prob
        best_splits = splits
        best_num_splits = num_splits

# 打印结果
print_results(len(probabilities), best_splits)