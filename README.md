[dennybritz's original code](https://github.com/dennybritz/cnn-text-classification-tf) supports python 3,but not support Chinese</br>
[indiejoseph's original code](https://github.com/indiejoseph/cnn-text-classification-tf-chinese) supports Chinese,but it does not support python3 and tensorflow 1.1</br>
I mixed them up.</br>
I dont know how it works.</br>
But it actually works.</br>
The highway has been removed because i dont know how to make it work on tf 1.1 </br>
My graphic card is GTX960 with 2GB memory,it would have two delay.One occurs when writting data before training.The other occurs when the first time evaluate.  </br>
If you have the same delay,please set the TDR Delay to 20.Or the operating system would kill it</br>
And i add pridict.py.It is used to pridict if the sentences are cantonese.
You can load your own checkpoint to make your own classification.The usage would be given in follow.

[dennybritz 的代码](https://github.com/dennybritz/cnn-text-classification-tf) 支持 python 3,但不支持中文，训练的准确率只有70%左右</br>
[indiejoseph 的代码](https://github.com/indiejoseph/cnn-text-classification-tf-chinese) 支持中文，但不能在python3，tensorflow1.1的平台上运行</br>
于是我把他们的代码拼起来了</br>
我不知道为什么</br>
反正它能跑了</br>
Highway 这个层我注释掉了并前后文做了一点修改，因为我不知道怎么样让他在tf 1.1上跑起来，我好菜啊</br>
我的古董GTX960只有两个G的显存，所以在加载数据和第一次评估的时候会卡屏</br>
如果你遇到同样的情况，请把TDR延迟调至20，否则卡两秒就被操作系统结束进程了</br>
我加了一个pridict.py，用来区分这些句子是不是广东话
你可以调用自己的存储点来做自己的分类,用法会在下面给出

特别鸣谢:睿睿老师和朱敬xua老师


The following are their original README（Mixed，of coures）:</br>
以下是他们的README（同样是组合起来了）：</br>
## CNN for Chinese Text Classification in Tensorflow
Sentiment classification forked from [dennybritz/cnn-text-classification-tf](https://github.com/dennybritz/cnn-text-classification-tf), make the data helper supports Chinese language and modified the embedding from word-level to character-level, though that increased vocabulary size, and also i've implemented the [Character-Aware Neural Language Models](http://arxiv.org/pdf/1508.06615v4.pdf) network structure which CNN + Highway network to improve the performance, this version can achieve an accuracy of 98% with the Chinese corpus

**[This code belongs to the "Implementing a CNN for Text Classification in Tensorflow" blog post.](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)**

It is slightly simplified implementation of Kim's [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882) paper in Tensorflow.

## Requirements

- Python 3
- Tensorflow > 0.12
- Numpy

## My platform

- python3.5
- Tensorflow 1.1
- Numpy
- cuDNN 5.1

## Training

Print parameters:

```bash
./train.py --help
```

```
optional arguments:
  -h, --help            show this help message and exit
  --embedding_dim EMBEDDING_DIM
                        Dimensionality of character embedding (default: 128)
  --filter_sizes FILTER_SIZES
                        Comma-separated filter sizes (default: '3,4,5')
  --num_filters NUM_FILTERS
                        Number of filters per filter size (default: 128)
  --l2_reg_lambda L2_REG_LAMBDA
                        L2 regularizaion lambda (default: 0.0)
  --dropout_keep_prob DROPOUT_KEEP_PROB
                        Dropout keep probability (default: 0.5)
  --batch_size BATCH_SIZE
                        Batch Size (default: 64)
  --num_epochs NUM_EPOCHS
                        Number of training epochs (default: 100)
  --evaluate_every EVALUATE_EVERY
                        Evaluate model on dev set after this many steps
                        (default: 100)
  --checkpoint_every CHECKPOINT_EVERY
                        Save model after this many steps (default: 100)
  --allow_soft_placement ALLOW_SOFT_PLACEMENT
                        Allow device soft device placement
  --noallow_soft_placement
  --log_device_placement LOG_DEVICE_PLACEMENT
                        Log placement of ops on devices
  --nolog_device_placement

```

Train:

```bash
./train.py
```

## Evaluating

```bash
./eval.py --eval_train --checkpoint_dir="./runs/1459637919/checkpoints/"
```

Replace the checkpoint dir with the output from the training. To use your own data, change the `eval.py` script to load your data.

## pridicting

```bash
import pridict
pridict( ( "sentence" , ) )
```

## References

- [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882)
- [A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1510.03820)
