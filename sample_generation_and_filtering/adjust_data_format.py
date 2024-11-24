import re
import json

def process_text_file(input_file_path, output_file_path):
#     # 读取文档内容
    with open(input_file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # 步骤1：替换 "问题1" 和 "答案1" 为 "问题" 和 "答案"
    content = re.sub(r'问题\d', '问题', content)
    content = re.sub(r'答案\d', '答案', content)
    content = re.sub(r'\n答案', '答案', content)

    # 使用正则表达式分割文档内容为问题和答案的列表
    segments = re.split(r'\s*问题：|答案：', content)
    # 过滤空字符串
    segments = list(filter(None, segments))

    # 确保列表包含偶数个元素（问题和答案成对出现）
    if len(segments) % 2 != 0:
        print("文本格式不正确，问题和答案应该成对出现。")
        # 删除多余的部分
        segments = segments[:len(segments)//2 * 2]

    # 构建 JSON 对象列表
    json_list = []
    for i in range(0, len(segments), 2):
        question = segments[i].strip()
        answer = segments[i + 1].strip()

        data = {'question': question, 'answer': answer}
        json_list.append(data)

    # 将结果写入 JSON 文件
    with open(output_json_filename, 'w', encoding='utf-8') as json_file:
        json.dump(json_list, json_file, ensure_ascii=False, indent=2)

# 提供的文本文件路径
input_file = '问答对去除不合格的问题索引后.txt'
output_json_filename = '问答对去除不合格的问题索引后dev.json'  # 替换为你的输出 JSON 文件名
process_text_file(input_file, output_json_filename)