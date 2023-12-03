def test_function():
    print("I was imported!")
    # print out the contents of test_data.txt
    with open('test_data.txt', 'r') as f:
        print(f'Read test data: {f.read()}')
