
# coding: utf-8

# In[8]:


"""Trains and Evaluates the MNIST network using a feed dictionary."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=missing-docstring
import argparse
import os
import sys
import time

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
import mnist
FLAGS = None


# In[9]:


def placeholder_inputs(batch_size):
    """Generate placeholder variables to represent the input tensors.
        These placeholders are used as inputs by the rest of the model building
        code and will be fed from the downloaded data in the .run() loop, below.
    Args:
        batch_size: The batch size will be baked into both placeholders.
    Returns:
        images_placeholder: Images placeholder.
        labels_placeholder: Labels placeholder.
    """
    # Note that the shapes of the placeholders match the shapes of the full
    # image and label tensors, except the first dimension is now batch_size
    # rather than the full size of the train or test data sets.
    
    images_placeholder = tf.placeholder(tf.float32, 
                                        shape=(batch_size, mnist.IMAGE_PIXELS))
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
    
    return images_placeholder, labels_placeholder
    
def fill_feed_dict(data_set, images_pl, labels_pl):
    """fills the feed dict for training the given step.
    A feed_dict takes the form of:
      feed_dict = {
          <placeholder>: <tensor of values to be passed for placeholder>,
          ....
      }
    Args:
        data_set: The set of images and labels, from input_data.read_data_sets()
        images_pl: The images placeholder, from placeholder_inputs().
        labels_pl: The labels placeholder, from placeholder_inputs().
      Returns:
        feed_dict: The feed dictionary mapping from placeholders to values.
    """
    # Create the feed_dict for the placeholders filled with the next
    # `batch size` examples.
    images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size, 
                                                   FLAGS.fake_data) 
    
    feed_dict = {
        images_pl: images_feed,
        labels_pl: labels_feed,
    }
    return feed_dict

def do_eval(sess, 
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_set):
    """ Runs one evaluation against the full epoch of data.
    Args:
        sess: The session in which the model has been trained.
        eval_correct: The Tensor that returns the number of correct predictions.
        images_placeholder: The images placeholder.
        labels_placeholder: The labels placeholder.
        data_set: The set of images and labels to evaluate, from
        input_data.read_data_sets().
    """
    #add run one epoch of eval.
    true_count = 0 #count the number of correct predictions
    steps_per_epoch = data_set.num_examples // FLAGS.batch_size
    num_examples = steps_per_epoch * FLAGS.batch_size
    for step in range(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set,
                                  images_placeholder,
                                  labels_placeholder)
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = float(true_count) / num_examples
    
    print('num_examples: %d num Correct: %d Precision @ 1: %0.04f' % 
          (num_examples, true_count, precision))


# In[10]:


def run_training():
    """Train MNIST for a number of steps."""
    # Get the sets of images and labels for training, validation, and
    # test on MNIST.
    data_sets = input_data.read_data_sets(FLAGS.input_data_dir, FLAGS.fake_data)
    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        images_placeholder, labels_placeholder = placeholder_inputs(
                                                    FLAGS.batch_size)
        
        #Build a graph that computes predictions from inference model
        logits = mnist.inference(images_placeholder,
                                 FLAGS.hidden_1,
                                 FLAGS.hidden_2)
    
        #Add to the graph the Ops for loss calculation
        loss = mnist.loss(logits, labels_placeholder)
    
        #Add to the graph the Ops that calculate and apply gradients
        train_op = mnist.training(loss, FLAGS.learning_rate)
    
        #Add the Op to compare the logits to the labels during evaluation.
        eval_correct = mnist.evaluation(logits, labels_placeholder)
    
        #Build summary tensor based on the TF collection of summaries
        summary = tf.summary.merge_all()
    
        #add the variable initializer Op.
        init = tf.global_variables_initializer()
    
        #Create a saver for writing training checkpoints
        saver = tf.train.Saver()
    
        #create a session for running Ops on the graph
        sess = tf.Session()
    
        #Instantiniate a SummaryWriter to summaries and the Graph
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
    
        #and then after everything is built
        #run the Op to initialize the variables
        sess.run(init)
    
        #start the training op
        for step in range(FLAGS.max_steps):
            start_time = time.time()
        
        #Fill a feed dictionary with the actual set of images and labels
        # for this particular training Op
            feed_dict = fill_feed_dict(data_sets.train,
                                      images_placeholder,
                                      labels_placeholder)
        
        # Run one step of the model.  The return values are the activations
        # from the `train_op` (which is discarded) and the `loss` Op.  To
        # inspect the values of your Ops or variables, you may include them
        # in the list passed to sess.run() and the value tensors will be
        # returned in the tuple from the call.
        
            _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
        
            duration = time.time() - start_time
        
        #write the summaries and print an overview fairly often.
            if step % 100 == 0:
                #print status to stdout
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                #update the events file.
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()
            
        #save a checkpoint and evaluate the model periodically.
            if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=step)
                #evaluate against the training set.
                print('Traininhg data eval:')
                do_eval(sess,
                       eval_correct,
                       images_placeholder,
                       labels_placeholder,
                       data_sets.train)
                #evaluate against the validation set.
                print("Validation data eval:")
                do_eval(sess,
                       eval_correct,
                       images_placeholder,
                       labels_placeholder,
                       data_sets.validation)
                #Evaluate against the test set.
                print("test data set")
                do_eval(sess,
                       eval_correct,
                       images_placeholder,
                       labels_placeholder,
                       data_sets.test)
            
def main(_):
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  run_training()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
          '--learning_rate',
          type=float,
          default=0.01,
          help='Initial learning rate.'
      )
    parser.add_argument(
          '--max_steps',
          type=int,
          default=2000,
          help='Number of steps to run trainer.'
      )
    parser.add_argument(
          '--hidden_1',
          type=int,
          default=128,
          help='Number of units in hidden layer 1.'
      )
    parser.add_argument(
          '--hidden_2',
          type=int,
          default=32,
          help='Number of units in hidden layer 2.'
      )
    parser.add_argument(
          '--batch_size',
          type=int,
          default=100,
          help='Batch size.  Must divide evenly into the dataset sizes.'
      )
    parser.add_argument(
          '--input_data_dir',
          type=str,
          default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                               'tensorflow/mnist/input_data'),
          help='Directory to put the input data.'
      )
    parser.add_argument(
          '--log_dir',
          type=str,
          default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                               'tensorflow/mnist/logs/fully_connected_feed'),
          help='Directory to put the log data.'
      )
    parser.add_argument(
          '--fake_data',
          default=False,
          help='If true, uses fake data for unit testing.',
          action='store_true'
      )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


# In[ ]:




