#!/usr/bin/env python3
"""
Question Answering using BERT from TensorFlow Hub and Hugging Face Transformers
"""
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(question, reference):
    """
    Finds the answer to a question within a given reference document.
    Args:
        question (str): The question to answer.
        reference (str): The reference document.
    Returns:
        str or None: The extracted answer or None if no answer is found.
    """
    # Initialize the BERT tokenizer from Hugging Face Transformers
    print("Initializing BERT Tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(
        "bert-large-uncased-whole-word-masking-finetuned-squad"
    )

    # Load the BERT QA model from TensorFlow Hub
    print("Loading BERT model from TensorFlow Hub...")
    model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")

    # Tokenize the inputs using the BERT tokenizer
    print("Tokenizing the question and reference document...")
    max_len = 512  # BERT maximum token length
    inputs = tokenizer(question, reference, return_tensors="tf",
                       max_length=max_len, truncation=True)

    # Prepare the input tensors for the TensorFlow Hub model:
    # - input_ids: Token IDs for the input sequence.
    # - attention_mask: Mask to indicate non-padding tokens.
    # - token_type_ids: To distinguish question from context.
    input_tensors = [
        inputs["input_ids"],
        inputs["attention_mask"],
        inputs["token_type_ids"]
    ]

    # Run inference on the model to get start and end logits
    print("Running inference on the model...")
    output = model(input_tensors)

    # Access the start and end logits from the output
    # Note: The model returns additional outputs,
    # but we only need the first two.
    start_logits = output[0]
    end_logits = output[1]

    # Get the input sequence length from the input_ids tensor shape
    sequence_length = inputs["input_ids"].shape[1]
    print(f"Input sequence length: {sequence_length}")

    # Determine the best start and end indices for the answer:
    # We ignore the special tokens at the beginning and end by slicing from
    # index 1 to sequence_length - 1.
    print("Determining the best start and end indices for the answer...")
    start_index = tf.math.argmax(start_logits[0, 1:sequence_length - 1]) + 1
    end_index = tf.math.argmax(end_logits[0, 1:sequence_length - 1]) + 1
    print(f"Start index: {start_index}, End index: {end_index}")

    # Extract the answer tokens using the best indices
    print("Extracting the answer tokens...")
    answer_tokens = inputs["input_ids"][0][start_index: end_index + 1]

    # Decode the answer tokens to obtain the final answer string
    print("Decoding the answer tokens...")
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True,
                              clean_up_tokenization_spaces=True)

    # If the decoded answer is empty or whitespace, return None
    if not answer.strip():
        print("No valid answer found.")
        return None

    print(f"Answer: {answer}")
    return answer
