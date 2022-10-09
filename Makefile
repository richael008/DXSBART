MPIDIR=$(PWD)
LIBS=-lm -DARMA_DONT_USE_WRAPPER -larmadillo
CC=g++
MPICC=mpic++
MPICFLAGS=-DMPIBART
CCOPS=-Wall -g -O3 -std=c++11

SOURCE=$(wildcard ./*.cpp)
OFiles=$(patsubst %.cpp,%.o,$(SOURCE))
TARGETS=$(patsubst %.cpp,%,$(SOURCE))

all:$(TARGETS)
allo:$(OFiles)

$(TARGETS):%:%.o
	$(MPICC) -o $@ $<  $(LIBS)

$(OFiles):%.o: %.cpp
	$(MPICC) $(MPICFLAGS) $(CCOPS) -c $< -o $@	

#--------------------------------------------------
.PHONY:clean all allo
clean:
	rm -rf $(TARGETS)	
	rm -f *.o