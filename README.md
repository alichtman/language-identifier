# Language Identifier

`language-identifier` accurately identifies English, French and Italian written text with up to 99% accuracy.

As this project was intended to help me better understand `n-gram analysis`, no natural language processing modules were imported. Everything was implemented from first principles.

![GIF demo](img/demo.gif)

### Usage

0. Download this repo as a `.zip`
1. `cd src`
2. `$ python3 lang-bigram-id`

Accuracy information for each model will be displayed in the terminal after the analysis is complete.

Diff the output files to see which lines were predicted differently by certain pairs of models.
Here are some commands to try:
```shell
$ diff ../output/letter-bigram-laplace-smoothing-predictions.txt ../output/letter-bigram-no-smoothing-predictions.txt
$ diff ../output/letter-bigram-laplace-smoothing-predictions.txt ../output/word-bigram-no-smoothing-predictions.txt
$ diff ../output/letter-bigram-laplace-smoothing-predictions.txt ../output/word-bigram-laplace-smoothing-predictions.txt
```

### Why?

This is was a computational linguistics experiment to see which language model, word or letter bigrams, performs the best. I also tested the impact of LaPlace smoothing on the predictive accuracy of the model.

Also for fun and profit.
