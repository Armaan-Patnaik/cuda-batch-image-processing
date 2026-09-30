IDIR=./
COMPILER=nvcc
COMPILER_FLAGS=-I$(IDIR) --std c++17 -O2

.PHONY: all clean build run

all: clean build run

build:
	$(COMPILER) $(COMPILER_FLAGS) batch_blur.cu -o batch_blur.exe

clean:
	rm -f batch_blur.exe
	rm -rf output artifacts dataset

run:
	./run.sh
