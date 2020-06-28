import re
from collections import Counter
from collections import defaultdict

# 打印纠正后的文件
def print_correct_word(word):
    print('error word: ', word)
    print('correct word: ', correct_word(word))

# 获取数据中的单词
def get_words(path):
    return re.findall('[a-z]+', open(path).read().lower())

# 测试得到的词列表是否在已知的词库中
def known(wordlist):
    return [w for w in wordlist if w in WORDS_DICT]

# 返回所有单词编辑距离为1的集合
def edit1(word):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    length = len(word)
    del_word = [word[0:i] + word[i+1:] for i in range(length)]
    transpose_word = [word[0:i] + word[i+1] + word[i] + word[i+2:] for i in range(length-1)]
    replace_word = [word[0:i] + letter + word[i+1:] for i in range(length) for letter in letters]
    insert_word = [word[0:i] + letter + word[i:] for i in range(length+1) for letter in letters]
    return set(del_word + transpose_word + replace_word + insert_word)

# 返回所有单词编辑距离为2的集合
def edit2(word):
    return set(w2 for w1 in edit1(word) for w2 in edit1(w1))

# 获取纠正后的单词
def correct_word(word):
    result = known([word]) or known(edit1(word)) or known(edit2(word)) or [word]
    return max(result, key=lambda w: WORDS_DICT[w])

# 设置默认字典，不存在字典的单词一个默认值，设置为1，毕竟不能保证所有词都存在我们的字典
# 这种方式能帮助我们更好的推测单词
WORDS_DICT = defaultdict(lambda: 1)

WORDS = Counter(get_words('./big.txt'))
#默认字典使用update方法，可以保留默认字典的默认值属性不变
WORDS_DICT.update(dict(WORDS))

print_correct_word('wrrd')


