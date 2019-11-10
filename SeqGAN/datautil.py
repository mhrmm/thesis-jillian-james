import json
import random
from sklearn.model_selection import train_test_split
import argparse
import sys
import re



# Helper functions for converting between text to DataLoader form

def haiku_to_ls(f):
    '''
    Makes lists of haikus from haiku file
    '''
    full = []
    haiku = []
    n = 0
    for line in f:
        if line == "\n":
            num_stopwords = 70 - len(haiku)
            if num_stopwords > 0:
                haiku += [" _FILL_ "]*num_stopwords
                full.append(haiku)
            else:
                haiku = haiku[:70]
                full.append(haiku)
            n+=1
            haiku = []
        else:
            line = line.strip()
            line = list(line.lower())
            line.append(" _BREAK_ ")
            haiku += line
    return full
    


def obama_to_ls(f):
    '''
    Makes list of Obama speach paragraphs from Obama file.
    '''
    full = []
    paragraph = []
    n = 0
    for line in f:
        if line != "\n":
            line = line.strip()
            line = re.findall(r"[\w']+|[.,!?();-]", line.lower())
            num_stopwords = 40 - len(line)
            if line != []:
                if len(line) < 40:
                    paragraph = line + [" _FILL_ "]*num_stopwords
                    full.append(paragraph)
                else:
                    paragraph = line[:40]
                    full.append(paragraph)
                n+= 1
    return full



def create_dicts(lines_ls):
    '''
    Creates dictionaries for converting between
    token and integer form for dataloader.
    '''
    all_tokens= []
    for line in lines_ls:
        for token in line:
            all_tokens.append(token)

    all_tokens = list(set(all_tokens))
    all_tokens = sorted(all_tokens)
    int_to_word, word_to_int = {}, {}

    for i in range(len(all_tokens)):
        word_to_int[all_tokens[i]] = i
        int_to_word[i] = all_tokens[i]

    return word_to_int, int_to_word, len(all_tokens)

def remove_filler(generated):
    for j in range(len(generated)):
        generated[j] = [value for value in generated[j] if value != " _FILL_ "]
    return generated

def text_ls_to_int_ls(text_ls, word_to_int):
    '''
    Converts a token list to integer for using
    dictionary mappings
    '''

    new_full = []
    for text in text_ls:
        new_text = []
        for word in text:
            new_text.append(word_to_int[word])
        new_full.append(new_text)
    return new_full



def int_file_to_text_ls(f, int_to_word):
    '''
    Reads file in dataloader form and converts to text
    using dictionary mappings
    '''
    text_ls = []
 
    for line in f:
        line = line.strip()
        line = line.split()
        parse_line = [int_to_word[x] for x in line]
        " ".join(parse_line)
        text_ls.append(parse_line)
    return text_ls


def write_lists_to_file(filename, full_lists):
    with open(filename, "w") as f:
        for ls in full_lists:
            line = " ".join([str(x) for x in ls])+ "\n"
            f.write(line)



def write_dict_to_file(filename, d):
    with open(filename, 'w') as f:
        json.dump(d, f)



def main():
    # Take in application from user and use it to create training and validation data
    parser = argparse.ArgumentParser(description='Program for converting a datafile to dataloader form.')
    parser.add_argument('app', metavar='application', type=str, default = 'obama',
                        help='Enter either \'obama\' or \'haiku\'')
    args = parser.parse_args()
    len_train, len_valid, len_test = 0.4, 0.4, 0.2
    if args.app == 'obama':
        whole = obama_to_ls(open('obama/input.txt', 'r'))
        train_ls, remainder = train_test_split(whole, test_size = 0.4, shuffle = False)
        valid_ls, test_ls = train_test_split(remainder, test_size = 0.5, shuffle = False)
    elif args.app == 'haiku': 
        whole = haiku_to_ls(open("haiku/input.txt", 'r'))
        train_ls, remainder = train_test_split(whole, test_size = 0.4, shuffle = False)
        valid_ls, test_ls = train_test_split(remainder, test_size = 0.5, shuffle = False)
    else:
        print("Application must be haiku or obama")
        sys.exit(0)

    # Create dictionaries to map integers to tokens and vice versa
    word_to_int, int_to_word, vocab_length = create_dicts(train_ls + valid_ls + test_ls)
    train_as_int_ls = text_ls_to_int_ls(train_ls, word_to_int)
    valid_as_int_ls = text_ls_to_int_ls(valid_ls, word_to_int)
    test_as_int_ls = text_ls_to_int_ls(test_ls, word_to_int)

    print("Vocab length: ", vocab_length)
    print("Original set length", len(whole))
    print("Training set length: ", len(train_as_int_ls))
    print("Valid set length: ", len(valid_as_int_ls))
    print("Test set length: ", len(test_as_int_ls))


    # Write to correct application training and validation files
    # Write conversion dictionaries
    if args.app == 'obama':
        write_dict_to_file("obama/word_to_int.json", word_to_int)
        write_dict_to_file("obama/int_to_word.json", int_to_word)
        write_lists_to_file("obama/obama_to_int.train.txt", train_as_int_ls)
        write_lists_to_file("obama/obama_to_int.valid.txt", valid_as_int_ls)
        write_lists_to_file("obama/obama_to_int.test.txt", test_as_int_ls)
    else:
        write_dict_to_file("haiku/word_to_int.json", word_to_int)
        write_dict_to_file("haiku/int_to_word.json", int_to_word)
        write_lists_to_file("haiku/haiku_to_int.train.txt", train_as_int_ls)
        write_lists_to_file("haiku/haiku_to_int.valid.txt", valid_as_int_ls)
        write_lists_to_file("haiku/haiku_to_int.test.txt", test_as_int_ls)

if __name__ == '__main__':
    main()