import unittest
from datautil import ObamaTokenizer, HaikuTokenizer, Vocab

class TestDatautil(unittest.TestCase):

    def test_create_dicts(self):
        haiku = [['I', ' ', 'a', 'm', ' ', 'a', ' ', 'h', 'a', 'i', 'k', 'u', 
                  '.', ' _BREAK_ ', 'O', 'r', ' ', 'a', 'm', ' ', 'I', ' ',
                  'n', 'o', 't', ' ', 'a', ' ', 'h', 'a', 'i', 'k', 'u', '?',
                  ' _BREAK_ ', 'C', 'o', 'u', 'n', 't', ' ', 'a', 'n', 'd', 
                  ' ', 'f', 'i', 'n', 'd', ' ', 'o', 'u', 't', '.', 
                  ' _BREAK_ ', ' _FILL_ ']]

        vocab = Vocab.construct(haiku)
        w2i = vocab.word_to_int
        i2w = vocab.int_to_word
        assert w2i == {' ': 0,' _BREAK_ ': 1, ' _FILL_ ':2, '.': 3, '?': 4, 
                       'C': 5, 'I': 6, 'O': 7, 'a': 8, 'd': 9, 'f': 10,
                       'h': 11, 'i': 12, 'k': 13, 'm': 14, 'n': 15, 'o': 16,
                       'r': 17, 't': 18, 'u': 19}
        assert i2w == {0: ' ', 1: ' _BREAK_ ',  2: ' _FILL_ ', 3: '.', 4: '?', 
                       5: 'C', 6: 'I', 7: 'O', 8: 'a', 9: 'd', 10: 'f',
                       11: 'h', 12: 'i', 13: 'k', 14: 'm', 15: 'n', 16: 'o',
                       17: 'r', 18: 't', 19: 'u'}
        assert len(vocab) == 20


    def test_obama_to_ls(self):
        obama_speech = ["Hello, fellow Americans. If you're walking down " +
                        "the right path and you're willing " +
                        "to keep walking -- eventually you'll make progress.\n",
                        "Thank you very much. God Bless America.\n", 
                        '\n']
        tokenizer = ObamaTokenizer(40)
        result = tokenizer(obama_speech)

        assert len(result[0]) == 40
        assert len(result[1]) == 40
        assert result == [['hello', ',', 'fellow', 'americans', '.', 'if', 
                           "you're", 'walking', 'down', 'the', 'right', 'path', 
                           'and', "you're", 'willing', 'to', 'keep', 'walking', 
                           '-', '-', 'eventually', "you'll", 'make',
                           'progress', '.'] + [' _FILL_ '] * 15,
                          ['thank', 'you', 'very', 'much', '.', 'god',
                           'bless', 'america', '.'] + [' _FILL_ '] * 31]

    def test_haiku_to_ls(self):
        haiku = ["I am a haiku.\n", 
                 "Or am I not a haiku?\n", 
                 "Count and find out.\n", 
                 '\n']
        tokenizer = HaikuTokenizer(70)
        result = tokenizer(haiku)
        assert len(result[0]) == 70
        assert result == [['i', ' ', 'a', 'm', ' ', 'a', ' ', 'h', 'a', 'i',
                           'k', 'u', '.', ' _BREAK_ ', 'o', 'r', ' ', 'a',
                           'm', ' ', 'i', ' ', 'n', 'o', 't', ' ', 'a', ' ',
                           'h', 'a', 'i', 'k', 'u', '?', ' _BREAK_ ', 'c', 
                           'o', 'u', 'n', 't', ' ', 'a', 'n', 'd', ' ', 'f', 
                           'i', 'n', 'd', ' ', 'o', 'u', 't', '.', 
                           ' _BREAK_ '] + [' _FILL_ '] * 15]
        
    def test_encode(self):
        vocab = Vocab(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
        encoded = vocab.encode([['c', 'a', 'b'], ['b', 'a', 'g']])
        assert encoded == [[2, 0, 1], [1, 0, 6]]
        
        




if __name__ == "__main__":
	unittest.main()