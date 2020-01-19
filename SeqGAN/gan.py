import tensorflow as tf
import time
import os
from dataloader import Gen_Data_loader, Dis_dataloader
from generator import Generator, pre_train_generator
from discriminator import Discriminator, train_discriminator
from rollout import ROLLOUT
import argparse
from trainutil import generate_samples, target_loss
from datautil import load_task

###############################################################################
#  Generator  Hyper-parameters
###############################################################################
EMB_DIM = 32 # embedding dimension
HIDDEN_DIM = 32 # hidden state dimension of lstm cell
START_TOKEN = 0
SEED = 88
BATCH_SIZE = 64

###############################################################################
#  Discriminator  Hyper-parameters
###############################################################################
dis_embedding_dim = 64
dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
dis_dropout_keep_prob = 0.75
dis_l2_reg_lambda = 0.2
dis_batch_size = 64



def train_adversarial(sess, saver, MODEL_STRING, generator, discriminator, 
                      rollout, dis_data_loader, likelihood_data_loader, 
                      task, log, n):
    print('#################################################################')
    print('Start Adversarial Training...')
    log.write('adversarial training...\n')
    saver.restore(sess, tf.train.latest_checkpoint(MODEL_STRING))
    small_loss = float('inf')
    for total_batch in range(n):
        # Train the generator for one step
        samples = generator.generate(sess)
        rewards = rollout.get_reward(sess, samples, 16, discriminator) #I might actually need to change the value 16 here.
        feed = {generator.x: samples, generator.rewards: rewards}
        _ = sess.run(generator.g_updates, feed_dict=feed)

        # Test
        sample = generate_samples(sess, generator, BATCH_SIZE, 
                                  task.generated_num, task.eval_file)
        print("Examples from generator:")
        for sample in task.vocab.decode(samples)[:5]:
            print(sample)

        likelihood_data_loader.create_batches(task.valid_file)
        test_loss = target_loss(sess, generator, likelihood_data_loader)
        if test_loss < small_loss:
            small_loss = test_loss
            saver.save(sess, MODEL_STRING +"/model")
            print("Saving checkpoint ...")
        print("total_batch: ", total_batch, "test_loss: ", test_loss)
        buffer = "total_batch: " + str(total_batch) + "test_loss: " + str(test_loss)
        log.write(buffer)

        # Update roll-out parameters
        rollout.update_params()

        # Train the discriminator for 5 steps
        train_discriminator(sess, generator, discriminator, dis_data_loader, 
                            task, log, 5, BATCH_SIZE, task.generated_num,
                            dis_dropout_keep_prob)


def main():
    
    #  Create a parser to parse user input
    def parse_arguments():
        parser = argparse.ArgumentParser(description='Program for running several SeqGan applications.')
        parser.add_argument('app', metavar='application', type=str, choices=['obama', 'haiku', 'synth'],
                        help='Enter either \'obama\' or \'haiku\'')
        parser.add_argument('gen_n', type = int,
                        help='Number of generator pre-training steps')
        parser.add_argument('disc_n', type = int,
                        help='Number of discriminator pre-training steps')
        parser.add_argument('adv_n', type = int,
                        help='Number of adversarial pre-training steps')
        parser.add_argument('-l', metavar="seq_len", type = int, default = -1,
                        help = 'Length of the token sequences used for training.')
        parser.add_argument('-v', metavar="vocab_size", type = int, default = -1,
                        help = "The size of the vocab from the input files (outout by datautil.py)")
        parser.add_argument('-mn', metavar="model_name", type = str, default = "",
                        help = "Name for the checkpoint files. Will be stored at ./<app>/models/<model_name>")
    
        args = parser.parse_args()        
        task = load_task(args.app)
    
        #Make the /models directory if its not there.
        model_string = task.path +"/models/"
        if not os.path.exists("./"+model_string):
            os.mkdir("./"+model_string)
    
        #make the checkpoint directory if its not there.
        if args.mn == "":
            model_string += str(args.gen_n)+ "_" + str(args.disc_n) + "_" + str(args.adv_n)
            model_string += time.strftime("_on_%m_%d_%y", time.gmtime())
        else:
            model_string += args.mn
        if not os.path.exists("./"+model_string):
            os.mkdir("./"+model_string)
    
        return args.gen_n, args.disc_n, args.adv_n, model_string, task
    
    gen_n, disc_n, adv_n, MODEL_STRING, task = parse_arguments()

    # Initialize the random seed
    #random.seed(SEED)
    #np.random.seed(SEED)
    #tf.set_random_seed(SEED)
    assert START_TOKEN == 0

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    # Initialize the data loaders
    gen_data_loader = Gen_Data_loader(BATCH_SIZE, task.max_seq_length)
    likelihood_data_loader = Gen_Data_loader(BATCH_SIZE, task.max_seq_length) # For validation
    dis_data_loader = Dis_dataloader(BATCH_SIZE, task.max_seq_length)

    # Initialize the Generator
    generator = Generator(len(task.vocab), BATCH_SIZE, EMB_DIM, HIDDEN_DIM, 
                          task.max_seq_length, START_TOKEN)

    # Initialize the Discriminator
    discriminator = Discriminator(sequence_length=task.max_seq_length, 
                                  num_classes=2, 
                                  vocab_size=len(task.vocab), 
                                  embedding_size=dis_embedding_dim, 
                                  filter_sizes=dis_filter_sizes, 
                                  num_filters=dis_num_filters, 
                                  l2_reg_lambda=dis_l2_reg_lambda)

    # Set session configurations. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    saver = tf.train.Saver()
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # If restoring from a previous run ....
    if len(os.listdir("./"+MODEL_STRING)) > 0:
        saver.restore(sess, tf.train.latest_checkpoint(MODEL_STRING))


    # Create batches from the positive file.
    gen_data_loader.create_batches(task.train_file)

    # Open log file for writing
    log = open(task.log_file, 'w')

    # Pre_train the generator with MLE. 
    pre_train_generator(sess, saver, MODEL_STRING, generator, gen_data_loader, 
                        likelihood_data_loader, task, log, gen_n, BATCH_SIZE,
                        task.generated_num)
    print('Start pre-training discriminator...')

    # Do the discriminator pre-training steps
    saver.restore(sess, tf.train.latest_checkpoint(MODEL_STRING))
    train_discriminator(sess, generator, discriminator, dis_data_loader, 
                        task, log, disc_n, BATCH_SIZE, task.generated_num,
                        dis_dropout_keep_prob)
    print("Saving checkpoint ...")
    saver.save(sess, MODEL_STRING+ "/model")
    
    # Do the adversarial training steps
    rollout = ROLLOUT(generator, 0.8)
    train_adversarial(sess, saver, MODEL_STRING, generator, discriminator, 
                      rollout, dis_data_loader, likelihood_data_loader, 
                      task, log, adv_n)

    #Use the best model to generate final sample
    saver.restore(sess, tf.train.latest_checkpoint(MODEL_STRING))
    generate_samples(sess, generator, BATCH_SIZE, task.generated_num, 
                     task.eval_file)

    log.close()


if __name__ == '__main__':
    main()
