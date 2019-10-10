
import json
import random

# Helper functions for converting between Haiku to DataLoader form

def para_str_to_ls(file):
    with open(file, 'r') as f:
        full = []
        paragraph = []
        for line in f:
            if line != "\n":
                line = line.strip()
                line = line.split()
                if line != [] and len(line) <= 50:
                    num_stopwords = 50 - len(line)
                    paragraph = line + ["_FILL_"]*num_stopwords
                    full.append(paragraph)
    return full


def creat_dicts(para_ls):
    all_tokens= []
    for para in para_ls:
        for token in para:
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



def para_ls_to_int_ls(para_ls, word_to_int):
        new_full = []
        for para in para_ls:
            new_para = []
            for word in para:
                new_para.append(word_to_int[word])
            new_full.append(new_para)

        return new_full



def dataloader_form_to_para(datafile, int_to_word):
    para_ls = []
    with open(datafile, 'r') as f:
        for line in f:
            line = line.strip()
            line = line.split()
            parse_line = [int_to_word[x] for x in line]
            " ".join(parse_line)
            para_ls.append(parse_line)
    return para_ls



def write_lists_to_file(filename, full_lists):
    with open(filename, "w") as f:
        for ls in full_lists:
            line = " ".join([str(x) for x in ls])+ "\n"
            f.write(line)

def write_dict_to_file(filename, d):
    with open(filename, 'w') as f:
        json.dump(d, f)

  

obama_train_ls = para_str_to_ls("obama.train.txt")
obama_valid_ls = para_str_to_ls("obama.valid.txt")
word_to_int, int_to_word, vocab_length = creat_dicts(obama_train_ls+obama_valid_ls)
paras_train_as_int_ls = para_ls_to_int_ls(obama_train_ls, word_to_int)
paras_valid_as_int_ls = para_ls_to_int_ls(obama_valid_ls, word_to_int)
print(vocab_length)
write_dict_to_file("word_to_int.json", word_to_int)
write_dict_to_file("int_to_word.json", int_to_word)
write_lists_to_file("obama_to_int.train.txt", paras_train_as_int_ls)
write_lists_to_file("obama_to_int.valid.txt", paras_valid_as_int_ls)

# with open("int_to_word.json", 'r') as f:
#     int_to_word = json.load(f)
# paras = dataloader_form_to_para("generator_sample.txt", int_to_word)
# print(paras[random.randint(0, len(paras))])
        
        
