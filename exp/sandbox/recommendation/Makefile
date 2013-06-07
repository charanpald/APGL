all: SGDNorm2RegCython.so

SGDNorm2RegCython.so: SGDNorm2RegCython.pyx
	cython SGDNorm2RegCython.pyx
	gcc -c -fPIC SGDNorm2RegCython.c -I/usr/include/python2.7/
	gcc -shared -o SGDNorm2RegCython.so SGDNorm2RegCython.o
