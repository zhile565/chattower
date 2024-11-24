import re

with open('示例评分结果.txt', 'r', encoding='utf-8') as file:
    text = file.read()

    # # 使用正则表达式去除多余的部分，只保留真正的评分5
    # cleaned_text = re.sub(r"### 评分\d: (\d) ###", r"\1", text)
    #
    # print(cleaned_text)  # 输出：5
    # 使用正则表达式提取评分信息
    ratings = re.findall(r"### 评分\d: (\d) ###", text)
    print(ratings)
    print(len(ratings))
    # exit()
    # 将评分转换为整数
    ratings = [int(rating) for rating in ratings]
    llm_scores = []
    # 每隔3个数字计算平均值
    avg_ratings = []
    for i in range(0, len(ratings), 2):
        group_ratings = ratings[i:i + 2]
        avg_rating = sum(group_ratings) / len(group_ratings)
        avg_ratings.append(avg_rating)

    # 打印输出平均评分
    for i, avg_rating in enumerate(avg_ratings, start=1):
        print(avg_rating)
        llm_scores.append(avg_rating)
print(type(llm_scores))
print(llm_scores)
print(len(llm_scores))


llm_scores = [x / 5 for x in llm_scores]
print(llm_scores)
# 原始列表
original_list = llm_scores
# 阈值
threshold = 0.9

# 标记为 Negative 的元素
marked_list = ["Negative" if item < threshold else "Positive" for item in original_list]

# 计算 Negative 和 Positive 的数量
negative_count = marked_list.count("Negative")
positive_count = marked_list.count("Positive")

# 获取被标记为 Negative 的元素索引
negative_indices = [index for index, item in enumerate(marked_list) if item == "Negative"]

# 打印标记后的列表和数量
print("标记后的列表:", marked_list)
print("Negative 的数量:", negative_count)
print("Positive 的数量:", positive_count)
print(f"Negative indices: {negative_indices}")

# exit()


# 问答对文档路径
document_path = "示例待去除问答对.txt"  # 你的问答对文档路径

txt_file_path = document_path  # 你的txt文档路径
qa_pairs_list = []
with open(txt_file_path, 'r', encoding='utf-8') as file:
    qa_pairs = []
    for line in file:
        line = line.strip()
        if line == "该字符串已执行完毕":
            qa_pairs_list.append(qa_pairs)
            qa_pairs = []
        else:
            qa_pairs.append(line)
        # 将最后一组问答对添加到列表中
    if qa_pairs:
        qa_pairs_list.append(qa_pairs)
print(qa_pairs_list)
# 去除二级列表中的空元素
cleaned_list = [[item for item in sublist if item != ''] for sublist in qa_pairs_list]
print(cleaned_list)
for sublist in cleaned_list:
    sublist[-1] = sublist[-1] + ' [标识符]'

print(cleaned_list)

# 使用列表解析将二维列表转化为一维列表
one_dimensional_list = [item for sublist in cleaned_list for item in sublist]
print(one_dimensional_list)
print(len(one_dimensional_list))

# 使用zip和列表推导式将相邻的元素组合成子列表
paired_list = [list(pair) for pair in zip(one_dimensional_list[::2], one_dimensional_list[1::2])]
print(paired_list)
print(len(paired_list))


# 假设这是您的原始列表
original_list = paired_list

# 假设这是包含要删除元素的索引的列表
indices_to_remove = negative_indices


# 使用列表推导式创建一个新列表，其中不包含索引列表中的元素

filtered_list = [sublist for index, sublist in enumerate(original_list) if index not in indices_to_remove]

# 打印结果
print(filtered_list)
print(len(filtered_list))
file_path = '问答对去除不合格的问题索引后.txt'

# 打开文件并写入列表内容
with open(file_path, 'w', encoding='utf-8') as file:
    for sublist in filtered_list:
        for item in sublist:
            file.write(str(item) + '  ')
        # 删除最后一个逗号和空格
        file.write('\n')

print(f"列表已保存到 {file_path}")
