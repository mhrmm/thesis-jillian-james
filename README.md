 This repository contains my thesis work so far: 
 SeqGan is not mine, the original code is from https://github.com/LantaoYu/SeqGAN.
 
 My additions include the haiku and obama subfolders which are applications meant to follow
 those in the authors' paper -- which were not released on the SeqGan github.

 I have also added the files:
    module_sequence_gan.py
    datautil.py
    generated_sample.py


To run this application, you must first run datautil.py and then module_sequence_gan.py.

Note that module_sequance_gan.py has the following required arguments
    app         Either 'obama' or 'haiku'
    gen_n       Number of generator pre-training steps 
    disc_n      Number of discriminator pre-training steps
    adv_n       Number of adversarial training steps

It also also has several optional arguments:
    -l SEQ_LEN        The length of the token sequences used for training
    -v VOCAB_SIZE     The size of the vocab from the input files (outout by datautil.py)


So to run the Obama application with default sequence length and vocab size, run:
    python3 datautil.py obama
    python3 module_sequance_gan.py obama gen_n disc_n adv_n

To run the Obama application with custom sequence length and vocab size, run:
    python3 datautil.py obama
    python3 module_sequance_gan.py obama gen_n disc_n adv_n -l SEQ_LEN -v VOCAB_SIZE       
