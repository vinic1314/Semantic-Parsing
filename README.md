
# Geoquery parsing 
This is a project for CS 388 Natural Language Processing, spring 2021 semester.
In this project, I created a semantic parser from scratch using an Sequence2Sequence model
built in PyTorch. The evaluation code was provided and was developed by Robert Jia and Percy Liang
of Stanford.


## Goal
 Create a decoder module in PyTorch, combine it with the encoder, perform training, and perform
inference. This was to perform semantic parsing on a geoquery dataset.

Two examples from the dataset are provided below:

what is the population of atlanta ga ? <br/>
_answer ( A , ( _population ( B , A ) , _const ( B , _cityid ( atlanta , _ ) ) ) )

what states border texas ? <br/>
_answer ( A , ( _state ( A ) , _next_to ( A , B ) , _const ( B , _stateid ( texas ) ) ) )