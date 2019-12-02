# The Classification of News Statement by Using Machine Learning Models

This project is about comparing the result of some different machine learning models.

* Bi-LSTM with self-attention

    Tensorflow Implementation of "[A Structured Self-attentive Sentence Embedding](https://arxiv.org/abs/1703.03130)" (ICLR 2017).

    ![image](https://user-images.githubusercontent.com/15166794/41864478-21cbf7c8-78e5-11e8-94d2-5aa035a65c8b.png)

* Naive Bayes

### Data Source
* AG's news topic classification dataset.
* The csv files were available from [here](https://github.com/mhjabreel/CharCNN/tree/master/data/ag_news_csv).

### Train 

I use some different combination about model and word to vector method

### Evaluation
If you want to evaluate the model, you should change the hyper-parameter --checkpoint_dir


## Results

| Model     | Test Accuracy |
| :---        |    :----:   |
| Bi-LSTM + One-Dense in Small Datasets   |   80%   |
| Bi-LSTM + One-Dense in Whole Datasets   |   86%   |
| **Bi-LSTM + Word2Vec in Whole Datasets**|   **89%**   |
|Naïve Bayes + One-Dense in Whole Datasets|   30%   |
|**Naïve Bayes + TF-IDF Vectorize in Whole Datasets**|  **90%**  |


## Reference
* A Structured Self-attentive Sentence Embedding (ICLR 2017), Z Lin et al. [[paper]](https://arxiv.org/abs/1703.03130)
* flrngel's [Self-Attentive-tensorflow](https://github.com/flrngel/Self-Attentive-tensorflow) github repository
* roomylee's [self-attentive-emb-tf](https://github.com/roomylee/self-attentive-emb-tf) github