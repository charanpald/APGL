all: SparseUtilsCython.so

SparseUtilsCython.so: SparseUtilsCython.pyx
	cython SparseUtilsCython.pyx
	gcc -c -fPIC SparseUtilsCython.c -I/usr/include/python2.7/
	gcc -shared -o SparseUtilsCython.so SparseUtilsCython.o
