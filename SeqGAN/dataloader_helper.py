import json
import random
import argparse
import sys



# Helper functions for converting between text to DataLoader form

def haiku_to_ls(file):
    '''
    Makes lists of haikus from haiku file
    '''
    with open(file, 'r') as f:
        full = []
        haiku = []
        n = 0
        for line in f:
            if line == "\n":
                num_stopwords = 70 - len(haiku)
                if num_stopwords > 0:
                    haiku += [" _FILL_ "]*num_stopwords
                    full.append(haiku)
                    n += 1
                haiku = []
            else:
                line = line.strip()
                line = list(line)
                line.append(" _BREAK_ ")
                haiku += line
    print("Size:", n )
    return full
    


def obama_to_ls(file):
    '''
    Makes list of Obama speach paragraphs from Obama file.
    '''
    with open(file, 'r') as f:
        full = []
        paragraph = []
        n = 0
        for line in f:
            if line != "\n":
                line = line.strip()
                line = line.split()
                if line != [] and len(line) <= 50:
                    num_stopwords = 50 - len(line)
                    paragraph = line + [" _FILL_ "]*num_stopwords
                    full.append(paragraph)
                    n += 1
    print("Size:", n )
    return full



def creat_dicts(lines_ls):
    '''
    Creates dictionaries for converting between
    token and integer form for dataloader.
    '''
    all_tokens= []
    for line in lines_ls:
        for token in line:
            all_tokens.append(token)

    all_tokens = list(set(all_tokens))
    random.shuffle(all_tokens)
    int_to_word, word_to_int = {}, {}
    integer_form = 0

    for token in all_tokens:
        word_to_int[token] = integer_form
        int_to_word[integer_form] = token
        integer_form += 1

    return word_to_int, int_to_word, integer_form



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



def int_file_to_text_ls(datafile, int_to_word):
    '''
    Reads file in dataloader form and converts to text
    using dictionary mappings
    '''
    text_ls = []
    with open(datafile, 'r') as f:
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
    if args.app == 'obama':
        train_ls = obama_to_ls("obama/obama.train.txt")
        valid_ls = obama_to_ls("obama/obama.valid.txt")
    elif args.app == 'haiku': 
        train_ls = haiku_to_ls("haiku/haiku.train.txt")
        valid_ls = haiku_to_ls("haiku/haiku.valid.txt")
    else:
        Print("Application must be haiku or obama")
        sys.exit(0)

    # Create dictionaries to map integers to tokens and vice versa
    word_to_int, int_to_word, vocab_length = creat_dicts(train_ls + valid_ls)
    train_as_int_ls = text_ls_to_int_ls(train_ls, word_to_int)
    valid_as_int_ls = text_ls_to_int_ls(valid_ls, word_to_int)

    # Require that files are the same length
    if len(train_as_int_ls) > len(valid_as_int_ls):
        train_as_int_ls = train_as_int_ls[:len(valid_as_int_ls)]
    else:
        valid_as_int_ls = valid_as_int_ls[:len(train_as_int_ls)]
    print("Vocab length: ", vocab_length)


    # Write to correct application training and validation files
    # Write conversion dictionaries
    if args.app == 'obama':
        write_dict_to_file("obama/word_to_int.json", word_to_int)
        write_dict_to_file("obama/int_to_word.json", int_to_word)
        write_lists_to_file("obama/obama_to_int.train.txt", train_as_int_ls)
        write_lists_to_file("obama/obama_to_int.valid.txt", valid_as_int_ls)
    else:
        write_dict_to_file("haiku/word_to_int.json", word_to_int)
        write_dict_to_file("haiku/int_to_word.json", int_to_word)
        write_lists_to_file("haiku/haiku_to_int.train.txt", train_as_int_ls)
        write_lists_to_file("haiku/haiku_to_int.valid.txt", valid_as_int_ls)




if __name__ == '__main__':
    main()