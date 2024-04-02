# Mixed-Effects Contexutal Bandits

Python3 based implementation of the paper "Mixed-Effects Contexutal Bandits".

## Directory tree

├── intercept
│   ├── model.py
│   └── exp.py
├── coefficient
│   ├── model.py
│   └── exp.py
└── movie
    ├── model.py
    └── exp.py

- 'intercept' contains the codes for random intercept model experiments. Run 'exp.py' to generate the results.
- 'coefficient'  contains the codes for random coefficient model experiments. Run 'exp.py' to generate the results.
- 'movie'  contains the codes for MovieLens 100K dataset experiments.
- We do not include the MovieLens dataset in this supplementary material due to copyright issues.

## Requirements
- python 3
- numpy
- matplotlib
- tqdm
- pandas
