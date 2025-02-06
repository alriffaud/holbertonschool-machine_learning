#!/usr/bin/env python3
"""
Answer loop for a QA chatbot.
This function repeatedly prompts the user for a question and uses a reference
text to find an answer using a pretrained BERT model. If no answer is found,
it prints a fallback message. If the user enters an exit command, it prints
Goodbye and exits.
"""
import importlib

# Dynamically import the question_answer function from the module "0-qa"
# (assuming "0-qa.py" is in the same directory)
qa_module = importlib.import_module("0-qa")
question_answer = qa_module.question_answer


def answer_loop(reference):
    """
    Runs an interactive loop that answers questions using a given reference
    text.
    Args:
        reference (str): The reference document used to find answers.
    """
    # Set of exit commands for case-insensitive comparison
    exit_commands = {"exit", "quit", "goodbye", "bye"}

    # Start an infinite loop to continuously prompt the user for questions
    while True:
        # Prompt the user with "Q: " and get the input
        user_question = input("Q: ")

        # Check if the user wants to exit (case-insensitive)
        if user_question.strip().lower() in exit_commands:
            print("A: Goodbye")
            break

        # Use the question_answer function to extract an answer from the
        # reference text
        answer = question_answer(user_question, reference)

        # If no valid answer is found, print the fallback message
        if not answer or not answer.strip():
            print("A: Sorry, I do not understand your question.")
        else:
            # Otherwise, print the extracted answer
            print("A:", answer)
