#! /usr/bin/env python
# -*- coding:utf-8 -*-
import predict
import tensorflow as tf
import os

# Parameters
# ==================================================

# Eval Parameters
tf.flags.DEFINE_string("file_dir","./", "The file to classify")
tf.flags.DEFINE_string("out_dir", "./output/", "The path for output")
tf.flags.DEFINE_integer("sentences_num",10000, "The number of output sentences")
tf.flags.DEFINE_boolean("num_limits", False, "Determind if the sentences_num limit is on")
tf.flags.DEFINE_integer("readlines_num", 512, "The number of output sentences")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
def write_file(line,file,counter=0) : 
    line = line + "\n"
    if FLAGS.num_limits : 
        if counter < sentences_num : 
            file.write(line)
            return 1
        else : return 0
    else :
        file.write(line)
        return 1



def classify(file_dir=FLAGS.file_dir,out_dir=FLAGS.out_dir,sentences_num=FLAGS.sentences_num,num_limits = FLAGS.num_limits):
    input_file = open(os.path.abspath(os.path.join(FLAGS.file_dir, "text.txt")),"r",encoding='utf8')
    pos_file = open(os.path.join(FLAGS.out_dir, "pos.txt"),"w",encoding='utf8')
    neg_file = open(os.path.join(FLAGS.out_dir, "neg.txt"),"w",encoding='utf8')
    pos_sentence_counter=0
    neg_sentence_counter=0
    read_counter=0
    while 1:
        # 先把所有数据读进来，内存应该够用
        lines = [line.strip() for line in input_file.readlines()]

        pbatch = 1024 # 每次处理1024条
        iter = 0 # 迭代指针

        while (iter * pbatch) < len(lines):
            dlines = lines[iter * pbatch : (iter+1) * pbatch]
            results = predict(dlines)
            for i in range(len(results)):
                if results[i] == 1:
                    # 预测为广东话，把 dlines[i] 写入文件
                    pos_sentence_counter += write_file(dlines[i],pos_file,pos_sentence_counter)
                else:
                    # 预测为普通话，同理
                    neg_sentence_counter += write_file(dlines[i],neg_file,neg_sentence_counter)
            if num_limits :
                if pos_sentence_counter >= sentences_num and neg_sentence_counter >= sentences_num : break
        iter+=1
        print("The "+format(pos_sentence_counter+neg_sentence_counter)+" line\n"
    
    pos_file.close()
    neg_file.close()
    
if __name__ == '__main__': 
    classify()            
    