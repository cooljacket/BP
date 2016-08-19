objs = main.o BP.o Matrix.o
CC = g++

out: $(objs)
	$(CC) -o out $(objs) -O3

main.o: main.cpp
BP.o: BP.h
Matrix.o: Matrix.h


clean:
	rm $(objs) out