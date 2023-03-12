import re

# 打开要替换的文件
def read_file(infile):
    with open(infile, "r", encoding="utf-8") as f:
        file_contents = f.read()
    return file_contents

# 定义要替换的旧函数名和新函数名
replacements = {
    "torch": "paddle",
    "nn.Module": "nn.Layer",
    ".zeros(": ".zeros_torch("
}

def ops_replace(file_contents, op_map={}):
    s = r"|".join(re.escape(k) for k in replacements)
    print(s)
    pattern = re.compile(s)

    new_file_contents = pattern.sub(
        lambda m: op_map[m.group(0)],
        file_contents
    )
    return new_file_contents


# 将替换后的文件写回原文件
def write_file(res, outfle):
    with open(outfle, "w", encoding="utf-8") as f:
        f.write(res)


if __name__ == '__main__':
    import sys
    infile = sys.argv[1]
    outfle = sys.argv[2]

    file_contents = read_file(infile)
    # 用正则表达式查找并替换所有匹配的函数名
    # for k, v in op_map.items():
    #     file_contents = re.sub(r"\b" + k + r"\b", v, file_contents)

    # r"\b"代表捕获单词边界
    # 将需要替换的字符串按照正则表达式格式合并成一个模式
    # pattern = re.compile(
    #     "|".join(
    #         rf"\b{re.escape(k)}\b" for k in replacements.keys()
    #     )
    # )
    #
    # # 使用正则表达式进行替换操作
    # new_file_contents = pattern.sub(
    #     lambda m: replacements[m.group(0)],
    #     file_contents
    # )

    # 定义正则表达式
    # s  = r"|".join(re.escape(k) for k in replacements)   # r自动加了转移\，若手动加会变成\\\匹配不到
    # print(s)
    # pattern = re.compile(s)
    #
    # # 使用正则表达式进行替换操作
    # new_file_contents = pattern.sub(
    #     lambda m: replacements[m.group(0)],
    #     file_contents
    # )

    new_file_contents = ops_replace(file_contents, op_map=replacements)

    write_file(new_file_contents,outfle)