_data.zip_ contains _train.dat_ and _test.dat_ formatted for nnet.
Please extract the files to be able to run _training.c_ and _inference.c_.

### Compilation

```
gcc -I.. -O3 -o training ../nnet.c training.c
gcc -I.. -O3 -o inference ../nnet.c inference.c
```