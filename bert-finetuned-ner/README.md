---
license: apache-2.0
tags:
- generated_from_trainer
datasets:
- conll2003
metrics:
- precision
- recall
- f1
- accuracy
model-index:
- name: bert-finetuned-ner
  results:
  - task:
      name: Token Classification
      type: token-classification
    dataset:
      name: conll2003
      type: conll2003
      args: conll2003
    metrics:
    - name: Precision
      type: precision
      value: 0.9331020812685827
    - name: Recall
      type: recall
      value: 0.9506900033658701
    - name: F1
      type: f1
      value: 0.9418139379793263
    - name: Accuracy
      type: accuracy
      value: 0.9865926885265203
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# bert-finetuned-ner

This model is a fine-tuned version of [bert-base-cased](https://huggingface.co/bert-base-cased) on the conll2003 dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0589
- Precision: 0.9331
- Recall: 0.9507
- F1: 0.9418
- Accuracy: 0.9866

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 3

### Training results

| Training Loss | Epoch | Step | Validation Loss | Precision | Recall | F1     | Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:---------:|:------:|:------:|:--------:|
| 0.0857        | 1.0   | 1756 | 0.0621          | 0.9181    | 0.9382 | 0.9281 | 0.9836   |
| 0.0308        | 2.0   | 3512 | 0.0611          | 0.9228    | 0.9458 | 0.9342 | 0.9846   |
| 0.0223        | 3.0   | 5268 | 0.0589          | 0.9331    | 0.9507 | 0.9418 | 0.9866   |


### Framework versions

- Transformers 4.12.0
- Pytorch 1.9.0+cu111
- Datasets 2.0.0
- Tokenizers 0.10.3
