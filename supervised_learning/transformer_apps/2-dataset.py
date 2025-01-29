#!/usr/bin/env python3
"""
This script defines a Dataset class for loading and preparing
a dataset for machine translation from Portuguese to English.
"""
import tensorflow_datasets as tfds
import transformers
import tensorflow as tf


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
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

        # Update data_train and data_valid with tokenized tensors
        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

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
        # Prepare a list of Portuguese and English sentences
        pt_sentences = []
        en_sentences = []
        # Iterate over the dataset
        for pt, en in data:
            pt_sentences.append(pt.numpy().decode('utf-8'))
            en_sentences.append(en.numpy().decode('utf-8'))

        # Create tokenizers for Portuguese and English
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased', use_fast=True,
            clean_up_tokenization_spaces=True)
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased', use_fast=True,
            clean_up_tokenization_spaces=True)

        # Train the tokenizers
        tokenizer_pt = tokenizer_pt.train_new_from_iterator(pt_sentences,
                                                            vocab_size=2**13)
        tokenizer_en = tokenizer_en.train_new_from_iterator(en_sentences,
                                                            vocab_size=2**13)

        # Return the tokenizers
        self.tokenizer_pt = tokenizer_pt
        self.tokenizer_en = tokenizer_en

        return self.tokenizer_pt, self.tokenizer_en

    def encode(self, pt, en):
        """
        This method encodes a translation into tokens.
        Args:
            pt: tf.Tensor containing the Portuguese sentence
            en: tf.Tensor containing the corresponding English sentence
        Returns:
            pt_tokens: np.ndarray containing the Portuguese tokens
            en_tokens: np.ndarray containing the English tokens
        """
        # Convert tensors to text
        pt_text = pt.numpy().decode('utf-8')
        en_text = en.numpy().decode('utf-8')

        # Get the vocab size for both tokenizers
        vocab_size_pt = self.tokenizer_pt.vocab_size
        vocab_size_en = self.tokenizer_en.vocab_size

        # Encode the sentences
        pt_tokens = self.tokenizer_pt.encode(pt_text,
                                             add_special_tokens=False)
        en_tokens = self.tokenizer_en.encode(en_text,
                                             add_special_tokens=False)

        # Add the special tokens to the encoded sentences
        pt_tokens = [vocab_size_pt] + pt_tokens + [vocab_size_pt + 1]
        en_tokens = [vocab_size_en] + en_tokens + [vocab_size_en + 1]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
        This method acts as a tensorFlow wrapper for the encode
        instance method.
        Args:
            pt: tf.Tensor containing the Portuguese sentence
            en: tf.Tensor containing the corresponding English sentence
        Returns:
            pt_tensor: Tokenized Portuguese sentence as tf.Tensor
            en_tensor: Tokenized English sentence as tf.Tensor
        """
        # Encode sentence pairs
        pt_tokens, en_tokens = tf.py_function(
            func=self.encode,
            inp=[pt, en],
            Tout=[tf.int64, tf.int64]
        )

        # Set shape dynamically (None means variable-length sequence)
        pt_tokens.set_shape([None])
        en_tokens.set_shape([None])

        return pt_tokens, en_tokens
