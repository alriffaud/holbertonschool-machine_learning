#!/usr/bin/env python3
"""
Multi-reference Question Answering using BERT and Semantic Search.
This function answers questions from multiple reference documents.
It first selects the most relevant document from the corpus using semantic
search, then uses a QA model to extract the answer from that document.
"""
import os


def question_answer(corpus_path):
    """
    Interactive loop that answers questions using a corpus of reference
    documents.
    Args:
        corpus_path (str): The path to the corpus of reference documents.
    """
    # Import the QA function from task 0 (BERT-based QA)
    qa = __import__("0-qa").question_answer
    # Import the semantic_search function from task 3
    semantic_search = __import__("3-semantic_search").semantic_search

    # Define a set of exit commands (in lowercase) for case-insensitive
    # comparison
    exit_commands = {"exit", "quit", "goodbye", "bye"}

    # Start the interactive loop to continuously get user input
    while True:
        # Prompt the user with "Q:" and store the input in 'user_question'
        user_question = input("Q: ")

        # Check if the user wants to exit (case-insensitive)
        if user_question.strip().lower() in exit_commands:
            print("A: Goodbye")
            break

        # Use semantic search to select the most relevant reference document
        # from the corpus based on the query sentence.
        reference_document = semantic_search(corpus_path, user_question)

        # Use the QA model to extract an answer from the selected reference
        # document.
        answer = qa(user_question, reference_document)

        # If no valid answer is found (i.e., answer is empty or None),
        # print a fallback message.
        if not answer or not answer.strip():
            print("A: Sorry, I do not understand your question.")
        else:
            # Otherwise, print the extracted answer.
            print("A:", answer)
