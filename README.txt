The data is zip from the source code https://github.com/mnielsen/neural-networks-and-deep-learning, the book's website is http://neuralnetworksanddeeplearning.com.
And I strongly recommend this book. It tells a lot of details of the BP algorithm and how the formulars are derived.

Before running the code, you should unzip the data.zip out. The data is too large and the speed using C++ fstream is too slow to stand. Therefore, I switch to use C to read the data.

I design the program to accept command lines parametes, you can run it like:
./main -e 3.0 -r 7000 -t 3000 -l 30 -d 0 -m 10

The parametes' meaning is like the following:
-e: eta for the BP model (default 1.0)
-r: the number of the trainning instance (default 50000)
-t: the number of the testing instance (default 10000)
-d: choose debug state, 0 is false while 1 stands for true (default false)
-l: epoches you want the trainning to run (default 30)
-m: mini_batch_size (default 10)

If you are not familiar with stochastic trainning, you can refer https://en.wikipedia.org/wiki/Stochastic_gradient_descent.

Thanks this blog: http://fantasticinblur.iteye.com/blog/1465497. I refer its design for the interfaces.