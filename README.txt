## English
v1: basic BP (shaofan tips me that testing and validation is abcsent)
v2: BP class now can hold on general network. And I design a Matrix class to hold on the calculations, but this class is somewhat slow because too many copy between objects, need improving.

1. data
The data is zip from the source code https://github.com/mnielsen/neural-networks-and-deep-learning, the book's website is http://neuralnetworksanddeeplearning.com.
And I strongly recommend this book. It tells a lot of details of the BP algorithm and how the formulars are derived.

2. How to run the code
Before running the code, you should unzip the data.zip out.

I design the program to accept command lines parameters, you can run it like:
./main -e 3.0 -r 6000 -v 1000 -t 3000 -l 30 -d 0 -m 10

The parametes' meaning is like the following:
-e: eta for the BP model (default 1.0)
-r: the number of the trainning instance (default 50000)
-t: the number of the testing instance (default 10000)
-v: the number of the validating instance (default 10000)
-d: choose debug state, 0 is false while 1 stands for true (default false)
-l: epoches you want the trainning to run (default 30)
-m: mini_batch_size (default 10)

3. What's more
If you are not familiar with stochastic trainning, you can refer https://en.wikipedia.org/wiki/Stochastic_gradient_descent.

The data is too large and the speed using C++ fstream is too slow to stand. Therefore, I switch to use C to read the data.




## 中文版
v1: 基础的BP算法的实现（少凡提醒我还差了测试和validation的部分）
v2: 设计了一个Matrix类来方便BP算法中的各种矩阵运行，泛化BP类可以支持多层的神经网络，而不止是三层的这种。不过Matrix的效率有些低，因为太多的拷贝操作了，有待提高。

1. 数据
本次用到的数据是来自MNIST手写数字数据集，参考这份代码里的数据集：https://github.com/mnielsen/neural-networks-and-deep-learning。强烈推荐这本书： http://neuralnetworksanddeeplearning.com，它讲得十分好，既有直观形象的说明，又有严谨的数学推导，对于理解神经网络和深度学习的基本概念非常有用。

2. 如何运行
我把数据集压缩了（因为实在太大了），在运行之前需要先把data.zip解压出来。运行时有挺多参数的，我做成通过命令行传入（这样就不需要改程序再编译了），比如你可以这样运行：./main -e 3.0 -r 6000 -v 1000 -t 3000 -l 30 -d 0 -m 10
各个参数的意义如下：
-e: 学习速率eta(默认为 1.0)
-r: 训练实例数目(默认为 50000)
-t: 测试实例数目(默认为 10000)
-v: 验证实例数目(默认为 10000)
-d: 是否debug，0为否，1为是 (默认否)
-l: 训练迭代的代数 (默认为 30)
-m: 随机训练的每批数据的大小 (默认为 10)

3. 更多
如果不熟悉随机梯度下降法，可以参考一下wiki，https://en.wikipedia.org/wiki/Stochastic_gradient_descent.

发现用C++的fstream来读取数据太慢了，所以使用了C的文件操作来读取