NVCC=nvcc
NVCCFLAGS=-std=c++17
NVCCFLAGS+=-gencode arch=compute_80,code=sm_80
NVCCFLAGS+=-lcublas -lcurand
NVCCFLAGS+=-I./src/cutf/include
NVCCFLAGS+=-I./src/mateval/include
NVCCFLAGS+=-L./src/mateval/build -lmateval_cuda
NVCCFLAGS+=-I./src/curand_fp16/include
NVCCFLAGS+=-L./src/curand_fp16/build -lcurand_fp16

TARGET=cublas.test

$(TARGET):src/main.cu src/mateval/build/libmateval_cuda.a src/curand_fp16/build/libcurand_fp16.a
	$(NVCC) $+ -o $@ $(NVCCFLAGS)

src/mateval/build/libmateval_cuda.a:
	mkdir -p src/mateval/build
	cd src/mateval/build && cmake .. && make -j

src/curand_fp16/build/libcurand_fp16.a:
	mkdir -p src/curand_fp16/build
	cd src/curand_fp16/build && cmake .. && make -j
  
clean:
	rm -f $(TARGET)
