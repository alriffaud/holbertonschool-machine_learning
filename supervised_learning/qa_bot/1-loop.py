#!/usr/bin/env python3
"""
Interactive loop script for a QA chatbot.
This script repeatedly prompts the user with "Q:" and prints "A:" as a response
If the user types "exit", "quit", "goodbye", or "bye" (case insensitive),
it prints "A: Goodbye" and exits.
"""

# Set of words that will terminate the program (all in lowercase for easy
# comparison)
exit_commands = {"exit", "quit", "goodbye", "bye"}

# Start an infinite loop to continuously get user input
while True:
    # Prompt the user with "Q: " and store the input in 'user_input'
    user_input = input("Q: ")

    # Remove any leading/trailing spaces and convert input to lowercase,
    # then check if the input is one of the exit commands
    if user_input.strip().lower() in exit_commands:
        # If an exit command is detected, print the goodbye message
        print("A: Goodbye")
        # Break out of the loop to end the program
        break
    else:
        # Otherwise, print "A:" as an empty response (as per example)
        print("A:")
