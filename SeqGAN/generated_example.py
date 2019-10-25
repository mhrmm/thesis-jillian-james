
import random
import json
import argparse
import sys



def main():
    # Print 5 random examples from generated data when given an application.
    parser = argparse.ArgumentParser(description='Program for showing generator sample given previously run application.')
    parser.add_argument('app', metavar='application', type=str, default = 'obama',
            help='Enter either \'obama\' or \'haiku\'. Must run module_sequance_gan.py with specified application before.')
    args = parser.parse_args()

    if args.app == "haiku":
        filename = "haiku/generator_sample.txt"
        with open("haiku/int_to_word.json", 'r') as f:
            int_to_word = json.load(f)
    else:
        filename = "obama/generator_sample.txt"
        with open("obama/int_to_word.json", 'r') as f:
            int_to_word = json.load(f)

    text_ls = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            line = line.split()
            parse_line = [int_to_word[x] for x in line]
            " ".join(parse_line)
            text_ls.append(parse_line)

    text_ls_sample = random.choices(text_ls, k = 5)

    for text in text_ls_sample:
        print(text)
        print()

if __name__ == '__main__':
    main()