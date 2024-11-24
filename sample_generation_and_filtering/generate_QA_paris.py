#coding=utf-8
import re
import matplotlib.pyplot as plt
import json
import zhipuai

def remove_blank_lines(text):
    # 然后使用列表推导式过滤掉空行（或只包含空白字符的行）
    non_blank_lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join(non_blank_lines)

def extract_stopwords(text):  # 在输入的text文本中寻找停用词
    # pattern = re.compile(r'\n\d\.\d\.\d+|\n\d\.\d+')
    # pattern = re.compile(r'\n\d+\.\d\.\d\.\d+|\n\d+\.\d\.\d+|\n\d+\.\d+|\n\d+）')
    # pattern = re.compile(r'\n\d+\.\d\.\d\.\d+|\n\d+\.\d\.\d+|\n\d+\.\d+|\n\d+）')
    # pattern = re.compile(r'\n\d+\.\d\.\d+|\n\d+\.\d+')
    #pattern = re.compile(r'\n\d+\.\d\.\d|\n\d+\.\d+|\n\d+\.\d\.\d\.\d|\n\d\.\d+\.\d')
    #pattern = re.compile(r'\n\d+\.\d\.\d|\n\d+\.\d+|\n\d+\.\d\.\d\.\d|\n\d+\.')(生成QA示例文本.txt)
    #pattern = re.compile(r"(一、|二、|三、|四、|五、|六、|七、|八、|九、|十、|十一、|十二、|十三、|十四、|十五、|十六、|十七、|十八、|十九、|二十、)")
    # pattern = re.compile(r'\n\d+\.\d\.\d+|\n\d+\.\d+|(一、|二、|三、|四、|五、|六、|七、|八、|九、|十、|十一、|十二、|十三、|十四、|十五、|十六、|十七、|十八、|十九、|二十、)')
    pattern = re.compile(r'\n\d+\.\d\.\d\.\d+|\n\d+\.\d\.\d+|\n\d+\.\d+')
    stopwords_indices = [m.start() for m in re.finditer(pattern, text)]
    return stopwords_indices


def extract_text_segments(text, indices):  # 根据停用词索引把输入的text文本分割成字符串，放到一个列表中
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
        #
        # segment = re.sub(r"\n图\d+-.*", "", segment)
        # segment = re.sub(r"\n[一二三四五六七八九十].*", "", segment)
        pattern = r'^(图\d+–\d+\(续\)|表\d+–\d+\(续\))\s*$'
        segment = re.sub(pattern, '', segment, flags=re.MULTILINE)

        segment = re.sub(r'见图\d\-\d+|见表\d\–\d+', '', segment)
        # segment = re.sub(r'图\d\-\d+|表\d\–\d+', '', segment)
        segment = re.sub(r'图\d\–\d+(续)|表\d\–\d+(续)', '', segment)
        # segment = re.sub(r"\n\d\.\d+\.\d|\n\d\.\d\d\.\d|\n\d\.\d\d|\n\d+\.", "\n", segment)
        # segment = re.sub(r'图\d\-\d+|表\d\–\d+', '', segment)
        # segment = re.sub(r"\n\d\.\d+\.\d|\n\d\.\d\d\.\d|\n\d\.\d\d|\n\d\.\d+", "\n", segment)
        segment = re.sub(r"\n第\d章.*", "", segment)
        # segment = re.sub(r"\n\d\.\d\.\d\.\d+|\n\d+\.\d+\.\d+|\n\d+\.\d\d\.\d|\n\d+\.\d", "\n", segment)
        # segment = re.sub(r"\n\d+\.\d+\.\d+|\n\d+\.\d\d\.\d|\n\d+\.\d", "\n", segment)
        segment = re.sub(r"\n\d\.\d\.\d\.\d+|\n\d+\.\d+\.\d+|\n\d+\.\d\d\.\d|\n\d+\.\d+", "\n", segment)

        # segment = re.sub(r"\n\d\.\d+\.\d+|\n\d\.\d\d\.\d|\n\d\.\d", "\n", segment)
        segment = re.sub(r'^\s*图例\s*$(?m)', '', segment)
        segment = re.sub(r'^(图\d+–\d+\(续\)|表\d+–\d+\(续\))\s*$', '', segment)
        segment = re.sub(r"(图\d+\.\d+\.\d+|见图\d+\.\d+\.\d+)", "", segment)
        segment = re.sub(r"(表\d+\.\d+\.\d+|见表\d+\.\d+\.\d+)", "", segment)
        # segment = re.sub(r"\n\d+）", "\n", segment)
        # pattern = r'^(一、|二、|三、|四、|五、|六、|七、|八、|九、|十、|十一、|十二、|十三、|十四、|十五、|十六、|十七、|十八、|十九、|二十、)'
        # # 使用 re.sub() 函数进行替换，将匹配到的内容替换为空字符串
        # segment = re.sub(pattern, '\n', segment, flags=re.MULTILINE)
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
    # return non_chinese_count / total_count
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
    with open(input_file_path, 'r', encoding='utf-8') as file:
        input_text = file.read()
    input_text = remove_blank_lines(input_text)
    # 提取停用词的索引
    stopwords_indices = extract_stopwords(input_text)

    # 提取文本段落
    text_segments = extract_text_segments(input_text, stopwords_indices)

    text_revised = clear_text_end(text_segments)

    #   过滤字数小于10的文本段落
    filter_text_segments = filter_text_segments(text_revised, 20)

    character_counts = count_characters(filter_text_segments)
    plot_line_chart(character_counts)

    text_300plus = [segment for segment, count in zip(filter_text_segments, character_counts) if count > 350]

    filtered_segments = [segment for segment, count in zip(filter_text_segments, character_counts) if count <= 350]

    # 定义包含多个不同字符串的列表
    character_list = filtered_segments

    output_file_path = "根据示例文本生成的QA.txt"
    with open(output_file_path, 'w', encoding='utf-8') as output_file:

        # 遍历列表中的每个字符串
        for character in character_list:
            zhipuai.api_key = "24af4d578243d0d0d0f2381a9cecd553.CXbd97OqSBNVM4QP"

            # 使用当前字符串构建提示
            prompt = [
                {"role": "user",
                 "content": f"请您根据以下拆分后的塔机说明书文本，自动生成三个以内的问题及答案。请注意，生成的问题和答案必须完全来源于输入文本，不包含任何外部知识。同时，问题和答案需要全面涵盖文本中的所有内容，保持简洁和完整。问题和答案的格式必须统一，每个问题后跟一个答案。文本：{character}问题1：[根据文本内容自动生成问题]答案1：[根据文本内容自动生成答案]问题2：[根据文本内容自动生成问题]答案2：[根据文本内容自动生成答案]问题3：[根据文本内容自动生成问题]答案3：[根据文本内容自动生成答案]"},

            ]

            response = zhipuai.model_api.sse_invoke(
                model="ChatGLM_turbo",
                prompt=prompt,
                temperature=0.9,
                top_p=0.7,
                incremental=True
            )

            # 遍历 SSE 调用的事件流
            for event in response.events():
                if event.event == "add":
                    print(event.data, end="")
                elif event.event == "error" or event.event == "interrupted":
                    print(event.data, end="")
                elif event.event == "finish":
                    print(event.data)
                    print(event.meta, end="")
                else:
                    print(event.data, end="")
                output_file.write(event.data)

            print("该字符串已执行完毕")
            output_file.write("\n该字符串已执行完毕\n")
    print(f"输出结果已写入到 {output_file_path}")
