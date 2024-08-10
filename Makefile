NVCC=nvcc
NVCCFLAGS=-std=c++17
NVCCFLAGS+=-gencode arch=compute_80,code=sm_80
NVCCFLAGS+=-lcublas
NVCCFLAGS+=-I./src/mateval/include
NVCCFLAGS+=-L./src/mateval/build -lmateval_cuda

TARGET=cublas.test

$(TARGET):src/main.cu src/mateval/build/libmateval_cuda.a
	$(NVCC) $+ -o $@ $(NVCCFLAGS)

src/mateval/build/libmateval_cuda.a:
	mkdir -p src/mateval/build
	cd src/mateval/build && cmake .. && make -j
  
clean:
	rm -f $(TARGET)
