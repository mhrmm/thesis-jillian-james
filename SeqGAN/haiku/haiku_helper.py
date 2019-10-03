
import json
import random

# Helper functions for converting between Haiku to DataLoader form

def haiku_to_dataloader_form(datafile):
    with open(datafile, 'r') as f:
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

        word_to_int = {'_FILL_':1, "_BREAK_":2}
        int_to_word = {1:'_FILL_', 2:"_BREAK_"}

        integer_form = 3
        for haiku in full:
            for word in haiku:
                if word not in word_to_int:
                    word_to_int[word] = integer_form
                    int_to_word[integer_form] = word
                    integer_form +=1

        new_full = []
        for haiku in full:
            new_haiku = []
            for word in haiku:
                new_haiku.append(word_to_int[word])
            new_full.append(new_haiku)

        return new_full, integer_form, word_to_int, int_to_word



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

        

# haikus, vocab_length, word_to_int, int_to_word = haiku_to_dataloader_form("haiku.train.txt")
# print(vocab_length)
# write_dict_to_file("word_to_int.json", word_to_int)
# write_dict_to_file("int_to_word.json", int_to_word)
# write_lists_to_file("haiku_to_int.train.txt", haikus)

with open("int_to_word.json", 'r') as f:
    int_to_word = json.load(f)
haikus = dataloader_form_to_haiku("generator_sample.txt", int_to_word)
print(haikus[random.randint(0, len(haikus))])
        
        