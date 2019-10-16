import numpy as np
import tensorflow as tf
import random
from dataloader import Gen_Data_loader, Dis_dataloader
from generator import Generator
from discriminator import Discriminator
from rollout import ROLLOUT
from target_lstm import TARGET_LSTM
import pickle

#########################################################################################
#  Generator  Hyper-parameters
######################################################################################
EMB_DIM = 32 # embedding dimension
HIDDEN_DIM = 32 # hidden state dimension of lstm cell
SEQ_LENGTH =  50 #20 for haikus # 20 sequence length
START_TOKEN = 0
PRE_EPOCH_NUM = 120 #25 for haikus  # 120  supervise (maximum likelihood estimation) epochs
SEED = 88
BATCH_SIZE = 64

#########################################################################################
#  Discriminator  Hyper-parameters
#########################################################################################
dis_embedding_dim = 64
dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
dis_dropout_keep_prob = 0.75
dis_l2_reg_lambda = 0.2
dis_batch_size = 64

#########################################################################################
#  Basic Training Parameters
#########################################################################################
TOTAL_BATCH =  200 # 100 for haikus # 200
log_file =  "obama/obama_log.txt"            # 'save/experiment-log.txt' "haiku/haiku_log.txt"
positive_file =  'obama/obama_to_int.train.txt'     # 'save/real_data.txt' 'haiku/haiku_to_int.train.txt'
negative_file = 'obama/generator_sample.txt' # 'save/generator_sample.txt' 'haiku/generator_sample.txt
valid_file = "obama/obama_to_int.valid.txt" # "haiku/haiku_to_int.valid.txt"
eval_file =  'obama/eval_file.txt'         # 'save/eval_file.txt' 'haiku/eval_file.txt'
generated_num = 10000 
vocab_size =  12389 #5000


def generate_samples(sess, trainable_model, batch_size, generated_num, output_file):
    # Generate Samples
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        generated_samples.extend(trainable_model.generate(sess))

    with open(output_file, 'w') as fout:
        for poem in generated_samples:
            buffer = ' '.join([str(x) for x in poem]) + '\n'
            fout.write(buffer)


def target_loss(sess, target_lstm, data_loader):
    # target_loss means the oracle negative log-likelihood tested with the oracle model "target_lstm"
    # For more details, please see the Section 4 in https://arxiv.org/abs/1609.05473
    nll = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        g_loss = sess.run(target_lstm.pretrain_loss, {target_lstm.x: batch})
        nll.append(g_loss)

    return np.mean(nll)


def pre_train_epoch(sess, trainable_model, data_loader):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        _, g_loss = trainable_model.pretrain_step(sess, batch)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)


def pre_train_generator(sess, generator, gen_data_loader, likelihood_data_loader, eval_file, valid_file, log_file, num_epochs):
    log = open(log_file, 'w')
    print('Start pre-training...')
    log.write('pre-training...\n')
    for epoch in range(num_epochs):
        loss = pre_train_epoch(sess, generator, gen_data_loader)
        if epoch % 5 == 0:
            generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file)
            likelihood_data_loader.create_batches(valid_file)
            test_loss = target_loss(sess, generator, likelihood_data_loader)
            print('pre-train epoch ', epoch, 'test_loss ', test_loss)
            buffer = 'epoch:\t'+ str(epoch) + '\tloss:\t' + str(loss) + '\n'
            log.write(buffer)

def train_discriminator(sess, generator, discriminator, dis_dataloader, negative_file, positive_file, log_file, n):
    for _ in range(n):
        generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)
        dis_data_loader.load_train_data(positive_file, negative_file)
        for _ in range(3):
            dis_data_loader.reset_pointer()
            for it in range(dis_data_loader.num_batch):
                x_batch, y_batch = dis_data_loader.next_batch()
                feed = {
                    discriminator.input_x: x_batch,
                    discriminator.input_y: y_batch,
                    discriminator.dropout_keep_prob: dis_dropout_keep_prob
                }
                _ = sess.run(discriminator.train_op, feed)


def main():
    # Initialize the random seed
    random.seed(SEED)
    np.random.seed(SEED)
    assert START_TOKEN == 0

    # Initialize the data loaders
    gen_data_loader = Gen_Data_loader(BATCH_SIZE)
    likelihood_data_loader = Gen_Data_loader(BATCH_SIZE) # For testing
    dis_data_loader = Dis_dataloader(BATCH_SIZE)

    # Initialize the Generator
    generator = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN)

    # Initialize the Discriminator
    discriminator = Discriminator(sequence_length=SEQ_LENGTH, num_classes=2, vocab_size=vocab_size, embedding_size=dis_embedding_dim, 
                                filter_sizes=dis_filter_sizes, num_filters=dis_num_filters, l2_reg_lambda=dis_l2_reg_lambda)

    # Set session configurations. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # Create batches from the positive file.
    gen_data_loader.create_batches(positive_file)

    # Pre_train the generator with MLE. 
    pre_train_generator(sess, generator, gen_data_loader, likelihood_data_loader, eval_file, valid_file, log_file, PRE_EPOCH_NUM):
    print('Start pre-training discriminator...')
    # Train 3 epoch on the generated data and do this for 50 times
    train_discriminator(sess, generator, discriminator, dis_dataloader, negative_file, positive_file, log_file, 50)
    #Set rollout
    rollout = ROLLOUT(generator, 0.8)

    print('#########################################################################')
    print('Start Adversarial Training...')
    log.write('adversarial training...\n')
    for total_batch in range(TOTAL_BATCH):
        # Train the generator for one step
        samples = generator.generate(sess)
        rewards = rollout.get_reward(sess, samples, 16, discriminator) #I might actually need to change the value 16 here.
        feed = {generator.x: samples, generator.rewards: rewards}
        _ = sess.run(generator.g_updates, feed_dict=feed)

        # Test
        if total_batch % 5 == 0 or total_batch == TOTAL_BATCH - 1:
            generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file)
            likelihood_data_loader.create_batches(valid_file)
            test_loss = target_loss(sess, generator, likelihood_data_loader) 
            print("total_batch: ", total_batch, "test_loss: ", test_loss)
            log.write(buffer)

        # Update roll-out parameters
        rollout.update_params()

        # Train the discriminator for 5 steps
            train_discriminator(sess, generator, discriminator, dis_dataloader, negative_file, positive_file, log_file, 5)

    log.close()


if __name__ == '__main__':
    main()