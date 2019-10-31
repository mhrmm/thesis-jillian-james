import unittest
from datautil import create_dicts

class TestDatautil(unittest.TestCase):

    def test_create_dicts(self):
        haiku = [['I', ' ', 'a', 'm', ' ', 'a', ' ', 'h', 'a', 'i', 'k', 'u', '.', ' _BREAK_ ', 'O', 'r',
                ' ', 'a', 'm', ' ', 'I', ' ', 'n', 'o', 't', ' ', 'a', ' ', 'h', 'a', 'i', 'k', 'u', '?',
                ' _BREAK_ ', 'C', 'o', 'u', 'n', 't', ' ', 'a', 'n', 'd', ' ', 'f', 'i', 'n', 'd', ' ',
                'o', 'u', 't', '.', ' _BREAK_ ', ' _FILL_ ']]

        w2i, i2w, num_toks = create_dicts(haiku)
        print(w2i)
        assert w2i == {' ': 0,' _BREAK_ ': 1, ' _FILL_ ':2, '.': 3, '?': 4, 'C': 5, 'I': 6, 
                       'O': 7, 'a': 8, 'd': 9, 'f': 10, 'h': 11, 'i': 12, 
                       'k': 13, 'm': 14, 'n': 15, 'o': 16, 'r': 17, 
                       't': 18, 'u': 19}
        assert i2w == {0: ' ', 1: ' _BREAK_ ',  2: ' _FILL_ ', 3: '.', 4: '?', 5: 'C', 6: 'I', 
                       7: 'O', 8: 'a', 9: 'd', 10: 'f', 11: 'h', 12: 'i', 
                       13: 'k', 14: 'm', 15: 'n', 16: 'o', 17: 'r', 
                       18: 't', 19: 'u'}
        assert num_toks == 20


if __name__ == "__main__":
	unittest.main()