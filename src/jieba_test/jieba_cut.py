import jieba
from pprint import pprint

# 开启并行分词模式，参数为并发执行的进程数 并行模式仅支持posix系统,并行分词暂不支持Windows
# jieba.enable_parallel()
# 关闭并行分词模式
# jieba.disable_parallel()

strs=["我来到北京清华大学","乒乓球拍卖完了","中国科学技术大学"]


'''
************分词***********
支持四种分词模式：
    精确模式，试图将句子最精确地切开，适合文本分析；
    全模式，把句子中所有的可以成词的词语都扫描出来, 速度非常快，但是不能解决歧义；
    搜索引擎模式，在精确模式的基础上，对长词再次切分，提高召回率，适合用于搜索引擎分词。
    paddle模式，利用PaddlePaddle深度学习框架，训练序列标注（双向GRU）网络模型实现分词。同时支持词性标注。paddle模式使用需安装paddlepaddle-tiny，pip install paddlepaddle-tiny==1.6.1。目前paddle模式支持jieba v0.40及以上版本。jieba v0.40以下版本，请升级jieba，pip install jieba --upgrade
'''
# cut(self, sentence, cut_all=False, HMM=True)
# cut_all  完全模式为真，精确模式为假,默认是精确模式。
for str in strs:
    str_cut = jieba.cut(str)
    print('精确模式： ','|'.join(str_cut))
print('*'*20)
for str in strs:
    str_cut = jieba.cut(str,cut_all=True)
    print('全模式： ', '|'.join(str_cut))
print('*'*20)
for str in strs:
    str_cut = jieba.cut_for_search(str)
    print('搜索引擎模式： ', '|'.join(str_cut))
for str in strs:
    str_cut = jieba.cut(str,use_paddle=True)
    print('paddle： ', '|'.join(str_cut))
'''
lcut 分词结果返回list
'''
for str in strs:
    str_cut = jieba.lcut(str)
    print('lcut精确模式： ',str_cut)

'''
*************载入词典***************
jieba.load_userdict(file_name) # file_name 为文件类对象或自定义词典的路径
一个词占一行；每一行分三部分：词语、词频（可省略）、词性（可省略），用空格隔开，顺序不可颠倒。
add_word(word, freq=None, tag=None) 和 del_word(word) 可在程序中动态修改词典
suggest_freq(segment, tune=True) 可调节单个词语的词频
'''
print(jieba.lcut('如果放到post中将出错')) #['如果', '放到', 'post', '中将', '出错']
jieba.suggest_freq(('中','将'),True) # Ture 表示调整频率
print(jieba.lcut('如果放到post中将出错')) #['如果', '放到', 'post', '中', '将', '出错']

print(jieba.lcut('台中将投降')) #['台中', '将', '投降']
jieba.suggest_freq('中将',True)
print(jieba.lcut('台中将投降')) #['台', '中将', '投降']

'''
************关键词***TFIDF***TextRank***StopWords******
'''
import jieba.analyse

sentences = '小明硕士毕业于中国科学院计算所，后在日本京都大学深造'
# withWeight 是否返回权重值
# allowPOS 仅包括指定词性，默认为空
print(jieba.analyse.extract_tags(sentences,topK=5,withWeight=True,allowPOS=()))

# jieba.analyse.TFIDF(idf_path=None) 新建 TFIDF 实例，idf_path 为 IDF 频率文件
# ,关键词提取所使用逆向文件频率（IDF）文本语料库可以切换成自定义语料库的路径 jieba.analyse.set_idf_path(file_name) # file_name为自定义语料库的路径

#  textrank关键字提取
# jieba.analyse.textrank(sentence, topK=20, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'))

# 设置停用词
# jieba.analyse.set_stop_words()

'''
***************词性*******************
'''
import jieba
import jieba.posseg as psg
print(psg.lcut('我来到北京清华大学'))
for w,f in psg.cut('我来到北京清华大学'):
    print(w,f)
