#_*_coding:utf-8_*_
'''
Created on 2016年8月3日

@author: sugo_yzk
'''
import jieba
import comment_iterator
import global_list
from Read_Comment import *
from MyCorpus import *
from gensim import corpora, models, similarities
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

if __name__ == "__main__":
     #输入需要处理的文本
     demo = Read_Comment('test.txt')


     #对文本分词
     #实例化一个文本处理对象
     my_corpus = MyCorpus(demo)
     
     #得到字典
     #此处必须先执行一次，因为字典是遍历完才得到，不能边迭代产生每个单词的词袋模型边使用字典
     #memory_friendly不用全部存放在内存此函数不需要显式调用，在get_dict()中进行了隐式调用
     # my_corpus.get_Corpus()#得到one line one ducument,

     '''
     使用迭代法产生dict时，必须显示调用该方法，因为字典是迭代出每一条评论产生的
     但前提是one_line_one_comment必须存在
     '''
     my_corpus.get_Corpus()

     dic = my_corpus.get_dict()
     print "......得到词典......"
     print dic
     print type(dic)
     print dic.token2id

     for kkey in dic.token2id.keys():
          print kkey
          print  dic.token2id[kkey]
     # # print type(dic)
     # print "......得到各单词的id......"
     # print dic.token2id



     #必须得到词典之后才能调用
     comment = comment_iterator.comment_iterator("one_line_one_comment.txt",dic)

     for com in comment:#打印每一句话的词袋表示法
          print "句子是----------------------------------->"
          print com.split()
          '''
          打印一句评论中的每一个单词
          '''
          # print len(com.split())
          # for tem in com.split():
          #      print tem.decode('utf-8')
          print "对应的词袋是。。。"
          print dic.doc2bow(com.split())#doc2bow只计算每个唯一单词的出现次数

     #语料库是
     corpus = [dic.doc2bow(text.split()) for text in comment]
     print corpus

     #训练这个语料库，使用tdifd模型，一旦初始化完成，可应用于任意语句向量
     tfidf = models.TfidfModel(corpus)
     #使用这个模型来变换向量
     print "------使用iftdf模型变换向量------"
     corpus_tfidf = tfidf[corpus]

     iter_com = 0

     for doc in corpus_tfidf:
          if iter_com < len(global_list.comment_scores):
               print "--------------------for single comment--------------------------"
               print(doc)

               '''
               此处加上一个模块，计算每个单词的分数
               '''
               for  single in doc:
                    word_name = str(single[0])
                    # print "token no " + word_name

                    word_score = single[1] * int(global_list.comment_scores[iter_com])
                    # print word_score
                    # print '\n'
                    '''
                    将单词以及分数存入字典，相同的单词的分数进行累加求和

                    '''
                    my_corpus.update_word_score(global_list.word_score_sum, word_name, word_score)

               # print "whole comment score : " + global_list.comment_scores[iter_com].decode('utf-8')
               # print "comment " \
               #       "no :" + str(iter_com)
               # print '\n'

               '''
               next comment
               '''
               iter_com = iter_com + 1

     '''
              此处已经得到了每个单词的分数，接下来可用此分数判断每句话的感情倾向
              '''
     print "+++++++++++++++++++++++++++++++++++"
     print global_list.word_score_sum.keys()
     print global_list.word_score_sum


     '''
     此处已经得到了每个单词的分数，接下来可用此分数判断每句话的感情倾向
     对于weight之和大于1的情况，采取归一
     '''
     iter_com = 0
     sum = 0
     total_loss = 0

     for doc in corpus_tfidf:
          if iter_com < len(global_list.comment_scores):
               print "--------------------for single comment--------------------------"
               print(doc)

               weight_sum = 0
               # 组成该句子的每一个单词的权重求和，用每一个单词的权重除以这个和，得到的就是该单词对句子的重要程度

               for  single in doc:
                    weight_sum = single[1] + weight_sum
                    word_name = str(single[0])
                    sum = sum + global_list.word_score_sum.get(word_name)
               print sum / weight_sum
               # 概率值和是1的多少倍，说明结果应当相应的缩小或扩大，此为归一


               print "Precision loss: "
               print  (int(global_list.comment_scores[iter_com]) - sum / weight_sum) / int(global_list.comment_scores[iter_com])
               if (int(global_list.comment_scores[iter_com]) - sum / weight_sum) / int(global_list.comment_scores[iter_com]) > 0:
                    total_loss = total_loss + (int(global_list.comment_scores[iter_com]) - sum / weight_sum) / int(global_list.comment_scores[iter_com])



               else:
                   total_loss = total_loss - (int(global_list.comment_scores[iter_com]) - sum / weight_sum) / int(global_list.comment_scores[iter_com])



               sum = 0


     print '\n\n\n'
     print "------------------------------------------------------"
     print "---------------------Final result---------------------"
     print "------------------------------------------------------"
     print total_loss / len(global_list.comment_scores)









