#_*_coding:utf-8_*_
'''
Created on 2016��8��4��

@author: Administrator
'''

a = 1
    
def __iter__():
    global a
    while a < 10:
         a += 1
         yield a
           