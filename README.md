 This repository contains my thesis work so far: 
 SeqGan is not mine, the original code is from https://github.com/LantaoYu/SeqGAN,
 but the code has been refactored and modified.


Basic training
--------------

To run this application, you must run gan.py.

Note that gan.py has the following required arguments:
    app         Either 'obama' or 'haiku' or 'synth'
    gen_n       Number of generator pre-training steps 
    disc_n      Number of discriminator pre-training steps
    adv_n       Number of adversarial training steps

So to run the Obama application, run:
    python gan.py obama gen_n disc_n adv_n

After running the training, you can look at some of generated examples via:
    python generated_example.py obama
    
    
Generating synthetic data
-------------------------

You can create synthetic data for training by running data/synth/synthetic.py:

    data/synth/synthetic.py NUM_SENTS NUM_EATERS NUM_FOOD
    
which will generate NUM_SENTS sentences like:
 
    eater32 eats food28
    eater88 consumes food55

or like:

    food28 is eaten by eater32
    food55 is consumed by eater88
    
After generating this data, you can perform adversarial training using:

    python gan.py synth gen_n disc_n adv_n
