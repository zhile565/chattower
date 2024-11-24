#coding=utf-8
import re
import matplotlib.pyplot as plt
from openai import OpenAI

def remove_blank_lines(text):
    # 然后使用列表推导式过滤掉空行（或只包含空白字符的行）
    non_blank_lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join(non_blank_lines)

def extract_stopwords(text):   #  在输入的text文本中寻找停用词
    pattern = re.compile(r'\n\d+\.\d\.\d\.\d+|\n\d+\.\d\.\d+|\n\d+\.\d+')
    stopwords_indices = [m.start() for m in re.finditer(pattern, text)]
    return stopwords_indices

def extract_text_segments(text, indices):  #  根据停用词索引把输入的text文本分割成字符串，放到一个列表中
    segments = []
    start = 0
    for index in indices:
        segments.append(text[start:index])
        start = index
    segments.append(text[start:])
    return segments

def clear_text_end(segments):
    revised_segments = []
    for segment in segments:
        # 应用正则表达式删除不需要的部分
        # segment = re.sub(r"\n\d\.\d\.\d\.\d|\n\d\.\d\d\.\d|\n\d\.\d\d\.|\n\d\.\d\d|\n\d+\.\d\.\d|\n\d\.\d\.|\n\d+\.\d|\n\d\.\d\. ", "\n", segment)
        # segment = re.sub(r"\([^)]*见图[^)]*\)", "", segment)
        segment = re.sub(r'图\d+\.\d+\-\d+|表\d+\.\d+\-\d+', '', segment)
        pattern = r'^(图\d+–\d+\(续\)|表\d+–\d+\(续\))\s*$'
        segment = re.sub(pattern, '', segment, flags=re.MULTILINE)
        segment = re.sub(r'见图\d\-\d+|见表\d\–\d+', '', segment)
        segment = re.sub(r'图\d\–\d+(续)|表\d\–\d+(续)', '', segment)
        # segment = re.sub(r"\n\d\.\d+\.\d|\n\d\.\d\d\.\d|\n\d\.\d\d|\n\d+\.", "\n", segment)
        # segment = re.sub(r'图\d\-\d+|表\d\–\d+', '', segment)
        # segment = re.sub(r"\n\d\.\d+\.\d|\n\d\.\d\d\.\d|\n\d\.\d\d|\n\d\.\d+", "\n", segment)
        segment = re.sub(r"\n第\d章.*", "", segment)
        segment = re.sub(r"\n\d\.\d\.\d\.\d+|\n\d+\.\d+\.\d+|\n\d+\.\d\d\.\d|\n\d+\.\d+", "\n", segment)
        segment = re.sub(r'^\s*图例\s*$(?m)', '', segment)
        segment = re.sub(r'^(图\d+–\d+\(续\)|表\d+–\d+\(续\))\s*$', '', segment)
        segment = re.sub(r"(图\d+\.\d+\.\d+|见图\d+\.\d+\.\d+)", "", segment)
        segment = re.sub(r"(表\d+\.\d+\.\d+|见表\d+\.\d+\.\d+)", "", segment)
        revised_segments.append(segment)
    return revised_segments


def filter_text_segments(text_segments, k):
    filtered_segments = []
    removed_segments = []
    for segment in text_segments:
        # 去除换行符和空格
        cleaned_segment = segment.replace("\n", "").replace(" ", "")
        if len(cleaned_segment) > k:
            non_chinese_ratio = calculate_non_chinese_ratio(segment)
            if non_chinese_ratio <= 2/3 and not has_consecutive_non_chinese_lines(segment):
                filtered_segments.append(segment)
            else:
                removed_segments.append(segment)
        else:
            removed_segments.append(segment)
    write_to_txt(removed_segments, "删去文本.txt")
    return filtered_segments

def calculate_non_chinese_ratio(text):
    # 去除换行符和空格
    cleaned_text = text.replace("\n", "").replace(" ", "")
    chinese_count = len(re.findall(r'[\u4e00-\u9fff]', cleaned_text))
    total_count = len(cleaned_text)
    non_chinese_count = total_count - chinese_count
    return non_chinese_count / total_count if total_count > 0 else 0

def has_consecutive_non_chinese_lines(text):
    lines = text.split("\n")
    consecutive_count = 0
    for line in lines:
        # 去除每行中的空格
        cleaned_line = line.replace(" ", "")
        if not re.search(r'[\u4e00-\u9fff]', cleaned_line):
            consecutive_count += 1
            if consecutive_count >= 4:
                return True
        else:
            consecutive_count = 0
    return False

def count_characters(text_segments):
    character_counts = []
    total_characters = 0
    over_350_characters = 0

    for i, segment in enumerate(text_segments):
        # 去除换行符和空格
        cleaned_segment = re.sub(r'\s+', '', segment)
        count = len(cleaned_segment)
        character_counts.append(count)
        total_characters += count

        if count > 350:
            over_350_characters += count

        # 打印每个段落的字数和内容
        print(f"文本段落 {i + 1} 字数: {count} 内容: {segment}")

    # 计算超过350字的文字占全部文字的百分比
    over_350_percentage = (over_350_characters / total_characters) * 100 if total_characters > 0 else 0

    print(f"\n350字以上的文字占全部文字的百分比: {over_350_percentage:.2f}%")

    return character_counts

def write_to_txt(text_segments, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as file:
        for segment in text_segments:
            file.write(segment + "\n")
    print("结果已写入到", output_file_path)

def plot_line_chart(character_counts):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(character_counts) + 1), character_counts, marker='o', linestyle='-')
    plt.xlabel('文本段落编号')
    plt.ylabel('字符数')
    plt.title('文本段落字符数统计')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # 输入文本文件地址和输出文本文件地址
    input_file_path = "生成QA示例文本.txt"

    # 读取输入文本文件内容
    with open(input_file_path, 'r', encoding='utf-8') as file:         # r  文本会在正确读取后被关闭
        input_text = file.read()
    input_text = remove_blank_lines(input_text)
    # 提取停用词的索引
    stopwords_indices = extract_stopwords(input_text)

    # 提取文本段落
    text_segments = extract_text_segments(input_text, stopwords_indices)

    text_revised = clear_text_end(text_segments)


    #  过滤字数小于20的文本段落
    filter_text_segments = filter_text_segments(text_revised, 20)

    character_counts = count_characters(filter_text_segments)
    plot_line_chart(character_counts)

    text_300plus = [segment for segment, count in zip(filter_text_segments, character_counts) if count > 350]

    filtered_segments = [segment for segment, count in zip(filter_text_segments, character_counts) if count <= 350]

    character_list = filtered_segments
    # 列表中的字符串
    string_list = character_list  # 你的字符串列表
    print(string_list)


    # 问答对文档路径
    document_path = "评价问题示例QA对.txt"  # 你的问答对文档路径
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

    # 去除二级列表中的空元素
    cleaned_list = [[item for item in sublist if item != ''] for sublist in qa_pairs_list]
    for sublist in cleaned_list:
        sublist[-1] = sublist[-1] + ' [标识符]'


    # 使用列表解析将二维列表转化为一维列表
    one_dimensional_list = [item for sublist in cleaned_list for item in sublist]

    # 获取奇数索引位置的元素
    ##odd_index_elements = one_dimensional_list[1::2]
    even_index_elements = one_dimensional_list[0::2]

    # 使用列表推导式和字符串替换方法来移除指定的字符串
    cleaned_list = [
        item.replace('问题1：', '').replace('问题2：', '').replace('问题3：', '')
        for item in even_index_elements
    ]

    duplicated_list = [item for item in string_list for _ in range(3)]

    # 假设这是您的文本和问题的列表
    list1 = duplicated_list
    list2 = cleaned_list

    # 遍历列表2
    for index2 in range(len(list2)):
        # 赋值
        textA = list1[index2]
        textB = list2[index2]
        print(textA)
        print(textB)
        API_BASE = "https://api.lingyiwanwu.com/v1"
        API_KEY = "ec0bc35ba44b475aac388332b8bafecf"
        client = OpenAI(
            api_key=API_KEY,
            base_url=API_BASE
        )
        completion = client.chat.completions.create(
            model="yi-large",
            messages=[{"role": "user",
                       "content": f"您是一位擅长分析文本内容的有用AI助手。用户指令{textA}包括一段文本。AI助手依据这段文本生成的问题{textB}已生成。请以公正的法官身份评估AI助手对上述内容与问题提供的响应的质量！您应该为生成的问题提供四个评分。评分最高分为5分，最低分为1分。评分包括：评分1：问题必须直接来源于文本。评分2：确保问题在文本中有具象详细的信息。评分3：回答该问题不需要查阅额外的资源或文档。评分4：确保问题的实用性，即用户可以通过这个问题的回答获取实际价值或对实际操作有帮助。避免任何位置偏见，并确保响应的顺序不会影响您的决定。不要让响应的长度影响您的评价，尽可能客观。直接输出分数，并严格遵循以下格式：### 评分1: 数字 ### ### 评分2: 数字 ### ### 评分3: 数字 ### ### 评分4: 数字 ### "}],
            stream=True
        )
        # 定义要写入的文件名
        output_file = '使用大模型评价问题得分.txt'

        # 打开文件以写入模式，如果文件不存在将会被创建
        with open(output_file, 'a', encoding='utf-8') as f:
            for chunk in completion:
                # 使用文件对象的 write 方法写入每一块内容
                f.write(chunk.choices[0].delta.content or "")
                #        # 确保在写入后刷新文件内容
                f.flush()