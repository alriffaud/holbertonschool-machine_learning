#!/usr/bin/env python3
"""
This script defines a Dataset class for loading and preparing
a dataset for machine translation from Portuguese to English.
"""
import tensorflow_datasets as tfds
import transformers


class Dataset:
    """
    This class loads and preprocess a dataset for machine translation.
    """
    def __init__(self):
        """
        This method initializes the Dataset class by loading the train and
        validation splits and creating tokenizers for Portuguese and English.
        """
        # Load dataset splits
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train', as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation', as_supervised=True)

        # Create tokenizers
        self.token_pt, self.token_en = self.tokenize_dataset(self.data_train)

    def tokenize_dataset(self, data):
        """
        This method creates sub-word tokenizers for the dataset using
        pre-trained models.
        Args:
            data: tf.data.Dataset containing tuples (pt, en),
                  where `pt` is a Portuguese sentence and `en` is the
                  corresponding English sentence.
        Returns:
            tokenizer_pt: Portuguese tokenizer.
            tokenizer_en: English tokenizer.
        """
        # Portuguese tokenizer: neuralmind/bert-base-portuguese-cased
        token_pt = transformers.BertTokenizerFast.from_pretrained(
            "neuralmind/bert-base-portuguese-cased")

        # English tokenizer: bert-base-uncased
        token_en = transformers.BertTokenizerFast.from_pretrained(
            "bert-base-uncased")

        # Return both tokenizers
        return token_pt, token_en
