# wordlevel-tokenizer
An implementation of word-level tokenizer that only split based on whitespace character, unlike gpt-2 tokenizer which is subword-level

## Using the library
`train_vocab` method allows user to train the tokenizer on a folder of text file and/or a list of given str. After training, it can be used in similar place where gpt-2 tokenizer is used.
