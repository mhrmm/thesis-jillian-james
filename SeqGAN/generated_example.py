import datautil
import random
import nltk
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
        int_to_word = json.load(open("haiku/int_to_word.json", 'r'))
        generated = datautil.int_file_to_text_ls(open("haiku/eval_file.txt", 'r'), int_to_word)
        references = datautil.int_file_to_text_ls(open("haiku/haiku_to_int.test.txt", 'r'), int_to_word)
    else:
        int_to_word = json.load(open("obama/int_to_word.json", 'r'))
        generated = datautil.int_file_to_text_ls(open("obama/eval_file.txt", 'r'), int_to_word)
        references = datautil.int_file_to_text_ls(open("obama/obama_to_int.test.txt", 'r'), int_to_word)


    print("Removing _FILL_ tokens ...")
    generated = datautil.remove_filler(generated)
    references = datautil.remove_filler(references)


    text_ls_sample = random.choices(generated, k = 10)

    for text in text_ls_sample:
        print("--------------------------------")
        print(text)
    
    print("Calculating BLEUscore ...")
    #Calculate BlEUscore for whole corpus (Takes a while)
    BLEUscore = nltk.translate.bleu_score.corpus_bleu([references]*len(generated), generated)
    print()
    print("********" + "Corpus BLEUscore is: " + str(BLEUscore)+ "********")

if __name__ == '__main__':
    main()