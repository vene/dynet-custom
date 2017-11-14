MKL_PATH ?= /opt/intel/mkl
DYNET_PATH ?= /opt/dynet
EIGEN_PATH ?= /opt/eigen

CC = g++
DEBUG = -g
INCLUDES = -I$(DYNET_PATH) -I$(EIGEN_PATH) -I$(MKL_PATH)/include
LIBS = -L$(DYNET_PATH)/build/dynet/
CFLAGS = -O3 -Wall -Wno-sign-compare -Wno-int-in-bool-context -c -fmessage-length=0 $(INCLUDES) -DEIGEN_FAST_MATH -DEIGEN_USE_MKL_ALL -fPIC -fno-finite-math-only -Wno-missing-braces -std=c++14 -funroll-loops
LFLAGS = $(LIBS) -ldynet

all: test_identity

identity.o : identity.cc identity.h
	$(CC) $(CFLAGS) identity.cc 

test_identity: identity.o test_identity.o
	$(CC) identity.o test_identity.o $(LFLAGS) -o test_identity

test_identity.o : test_identity.cc identity.h
	$(CC) $(CFLAGS) test_identity.cc

clean: 
	rm -f *.o *~ test_identity
