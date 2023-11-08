import re

CLI_PROMPT = '> '

def cli_input(
    menu_prompt: str,
    input_regex: str,
    fail_regex_message: str):
    
    while True:
        print("\n" + menu_prompt)
        user_input = input(CLI_PROMPT)

        if user_input == "q":
            raise EOFError()

        if re.search(input_regex, user_input) is None:
            print(fail_regex_message)
            continue

        return user_input
