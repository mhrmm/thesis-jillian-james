
import json
import random

# Helper functions for converting between Haiku to DataLoader form

def haiku_str_to_ls(file):
    with open(file, 'r') as f:
        full = []
        haiku = []
        for line in f:
            if line == "\n":
                num_stopwords = 20 - len(haiku)
                if num_stopwords > 0:
                    haiku += ["_FILL_"]*num_stopwords
                    full.append(haiku)
                haiku = []
            else:
                line = line.strip()
                line = line.split()
                line.append("_BREAK_")
                haiku += line
    return full


def creat_dicts(haiku_ls):
    all_tokens= []
    for haiku in haiku_ls:
        for token in haiku:
            all_tokens.append(token)

    all_tokens = list(set(all_tokens))
    random.shuffle(all_tokens)
    int_to_word, word_to_int = {}, {}
    integer_form = 1

    for token in all_tokens:
        word_to_int[token] = integer_form
        int_to_word[integer_form] = token
        integer_form += 1

    return word_to_int, int_to_word, integer_form



def haiku_ls_to_int_ls(haiku_ls, word_to_int):
        new_full = []
        for haiku in haiku_ls:
            new_haiku = []
            for word in haiku:
                new_haiku.append(word_to_int[word])
            new_full.append(new_haiku)

        return new_full



def dataloader_form_to_haiku(datafile, int_to_word):
    haikus = []
    with open(datafile, 'r') as f:
        for line in f:
            line = line.strip()
            line = line.split()
            parse_line = [int_to_word[x] for x in line]
            " ".join(parse_line)
            haikus.append(parse_line)
    return haikus



def write_lists_to_file(filename, full_lists):
    with open(filename, "w") as f:
        for ls in full_lists:
            line = " ".join([str(x) for x in ls])+ "\n"
            f.write(line)

def write_dict_to_file(filename, d):
    with open(filename, 'w') as f:
        json.dump(d, f)

  

haikus_train_ls = haiku_str_to_ls("haiku.train.txt")
haikus_valid_ls = haiku_str_to_ls("haiku.valid.txt")
word_to_int, int_to_word, vocab_length = creat_dicts(haikus_train_ls+haikus_valid_ls)
haikus_train_as_int_ls = haiku_ls_to_int_ls(haikus_train_ls, word_to_int)
haikus_valid_as_int_ls = haiku_ls_to_int_ls(haikus_valid_ls, word_to_int)
print(vocab_length)
write_dict_to_file("word_to_int.json", word_to_int)
write_dict_to_file("int_to_word.json", int_to_word)
write_lists_to_file("haiku_to_int.train.txt", haikus_train_as_int_ls)
write_lists_to_file("haiku_to_int.valid.txt", haikus_valid_as_int_ls)

# with open("int_to_word.json", 'r') as f:
#     int_to_word = json.load(f)
# haikus = dataloader_form_to_haiku("generator_sample.txt", int_to_word)
# print(haikus[random.randint(0, len(haikus))])
        
        