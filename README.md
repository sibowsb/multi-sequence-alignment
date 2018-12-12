# Multisequence Alignment

This project implements a heuristic greedy progressive multiple sequence 
aligner, including the following functionalists: 
- Sequence-to-sequence global alignment 
- Sequence-to-profile alignment 
- Profile-to-profile alignment 

This is my final project for 
[CS 466 Introduction to Bioinformatics](http://www.el-kebir.net/teaching/CS466.html) 
at the University of Illinois at Urbana-Champaign, Fall 2018.

I have benchmarked this implementation against the 
[`SeqAlign`](https://github.com/niemasd/SeqAlign) implementation by `niemasd`.
Since `SeqAlign` does not have an open source license (as of 12/11/2018), I 
did not include it in this repository. To reproduce my results you need to
clone `SeqAlign` yourself.

## Demo
First, let's randomly generate 5 input sequences with maximum length 10, with a
0.2 probability that the characters get changed or dropped.
```Python
from benchmark import gen_test_input
input_seqs = gen_test_input(5, 10, 0.2)
print(input_seqs)
```

The randomly generated input sequences are:
```
{'v0': 'GACAATCCTG', 'v1': 'GACATCCTA', 'v2': 'GATGATCCTA', 'v3': 'GACCATCCTA', 'v4': 'GACCATCCTA'}

```

Now, let's use `multi_align` to perform alignment.
```
from greedy_aligner import multi_align
p = multi_align(input_seqs.values())
print(p)
```

The output is:
```
Alignment:
v1	    G	    A	    C	    -	    A	    A	    T	    C	    C	    T	    G
v2	    G	    A	    C	    -	    A	    -	    T	    C	    C	    T	    A
v3	    G	    A	    T	    G	    A	    -	    T	    C	    C	    T	    A
v4	    G	    A	    C	    C	    A	    -	    T	    C	    C	    T	    A
v5	    G	    A	    C	    C	    A	    -	    T	    C	    C	    T   	A

Profile:
A	    0.00	1.00	0.00	0.00	1.00	0.20	0.00	0.00	0.00	0.00	0.80
T   	0.00	0.00	0.20	0.00	0.00	0.00	1.00	0.00	0.00	1.00	0.00
C	    0.00	0.00	0.80	0.40	0.00	0.00	0.00	1.00	1.00	0.00	0.00
G	    1.00	0.00	0.00	0.20	0.00	0.00	0.00	0.00	0.00	0.00	0.20
-	    0.00	0.00	0.00	0.40	0.00	0.80	0.00	0.00	0.00	0.00	0.00
```
