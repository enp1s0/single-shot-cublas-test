# single shot cuBLAS GEMM test

## Build
```
git submodule --update --recursive
make
```

## Run
```
./cublas.test s N T 1000 1000 1000
./cublas.test c N T 1000 1000 1000
./cublas.test z N T 1000 1000 1000
./cublas.test d N T 1000 1000 1000
./cublas.test h N T 1000 1000 1000
```

## LICENSE
MIT
