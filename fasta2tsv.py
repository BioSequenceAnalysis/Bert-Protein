import re
import random
import os.path as osp

tmp = ''
# 统计正样本数量
pp = 0
# 统计负样本数量
n = 0
# p使用正则表达式将fasta序列进行分词，其中数字决定分词方式
p = re.compile(r'(\w{1})(?!$)')

with open('./input.fasta', 'r') as f1:
    with open('./out/19/train_sorted.tsv', 'w') as f2:
        with open('./out/19/dev_sorted.tsv', 'w') as f3:
            for line in f1.readlines():
                line = line.strip()
                if len(line) == 0:
                    continue
                if line[0] == '>' and len(tmp) == 0:
                    continue
                if line[0] == '>' and len(tmp) > 0:
                    tmp += '\n'
                    res = "train\t0\t\t" + p.sub(r'\1 ', tmp)
                    if random.random() < 0.1:
                        n += 1
                        if random.random() < 0.1:
                            f3.writelines(res)
                        else:
                            f2.writelines(res)
                    tmp = ''
                    continue
                tmp += line