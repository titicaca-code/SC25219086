import os
import json
import re


def load_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_all_poems(data_dir):
    file_names = [
        "poet.song.40000.json",
        "poet.song.41000.json",
        "poet.song.42000.json",
        "poet.song.43000.json",
    ]

    all_items = []
    for name in file_names:
        file_path = os.path.join(data_dir, name)
        data = load_json_file(file_path)
        all_items.extend(data)

    return all_items


def clean_text(text):
    """
    只保留中文和常见古诗标点
    """
    text = text.strip().replace(" ", "").replace("　", "")
    return text


def extract_sentences_with_punc(paragraphs):
    """
    把 paragraphs 合并，再按中文标点切分成句子。
    返回形如：
    ['雲淡風輕近午天', '望花隨柳過前川', '旁人不識予心樂', '將謂偷閑學少年']
    """
    text = "".join([clean_text(x) for x in paragraphs])

    # 只保留中文和标点，去掉奇怪符号
    text = re.sub(r"[^\u4e00-\u9fff，。！？；、]", "", text)

    # 按标点切句
    sentences = re.findall(r"[\u4e00-\u9fff]+", text)
    return sentences


def is_qijue(sentences):
    """
    七言绝句：
    - 恰好 4 句
    - 每句 7 个汉字
    """
    if len(sentences) != 4:
        return False
    return all(len(s) == 7 for s in sentences)


def build_qijue_dataset(all_items):
    poems = []

    for item in all_items:
        paragraphs = item.get("paragraphs", [])
        sentences = extract_sentences_with_punc(paragraphs)

        if is_qijue(sentences):
            # 训练时先存成连续字符串
            poem = "".join(sentences)
            poems.append(poem)

    return poems


def save_poems(poems, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        for poem in poems:
            f.write(poem + "\n")


def main():
    data_dir = "./data"
    output_path = os.path.join(data_dir, "qijue.txt")

    all_items = load_all_poems(data_dir)
    print(f"原始诗词条目总数: {len(all_items)}")

    poems = build_qijue_dataset(all_items)
    print(f"筛选出的七言绝句数量: {len(poems)}")

    if len(poems) > 0:
        print("\n前 5 条样例：")
        for i, poem in enumerate(poems[:5], start=1):
            print(f"{i}: {poem}")

    save_poems(poems, output_path)
    print(f"\n已保存到: {output_path}")


if __name__ == "__main__":
    main()