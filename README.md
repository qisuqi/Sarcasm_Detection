# Sarcasm Detection

Sarcasm detection is one of the key subfields of the study of Sentiment Analysis because sarcastic texts can often be mistaken by machine learning models to have positive polarity, but maybe a negative opinion was conveyed. 

This paper proposes a hybrid deep learning model that is Self-Attention-based Bidirectional Long Short-term Memory (sAtt-BLSTM) and Convolutional Neural Network (CNN) for detecting sarcasm in social network contents automatically. Punctuation-, sentiment-, and semantic-based auxiliary features are merged into CNN alongside with the feature maps generated by sAtt-BLSTM. 

Three datasets are used to investigate the robustness of the proposed model: an imbalanced dataset from SemEval 2018 Task 3A (Van Hee, et al., 2018) containing 3,834 annotated tweets of which 1,923 tweets are not sarcastic and 1,911 tweets are sarcastic, an imbalanced dataset provided by (Riloff, et al., 2013) containing 877 annotated tweets of which 721 are not sarcastic and 156 are sarcastic, and a balanced dataset containing harvested real-time tweets obtained from Twitter API with 2,000 tweets annotated by the author of this paper. 

It is observed that the proposed model obtained a testing accuracy of 66%, 77%, and 90% when predicting the unseen test data on the SemEval, Riloff, and Harvested datasets, respectively. 

Three word embedding methods are also experimented with: Word2Vec, GloVe, and BERT. 

Model architecture:

![image](https://user-images.githubusercontent.com/63663984/233617676-4cf9310b-10f3-4bc9-b6e6-41b64cb7d603.png)

Riloff, E. et al., 2013. "Sarcasm as contrast between a positive sentiment and negative situation". In EMNLP, Volume Volume 13, p. 704–714.
Van Hee, C., Lefever, E. & Hoste, V., 2018. "SemEval-2018 Task 3: Irony Detection in English Tweets". In Proceedings of The 12th International Workshop on Semantic Evaluation, pp. 39-50.
