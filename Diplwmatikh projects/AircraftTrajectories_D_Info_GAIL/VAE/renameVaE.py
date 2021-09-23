import tensorflow as tf
import os
import time

current_time = time.strftime('%Y-%m-%d-%H-%M-%S')


#RENAME MAP LAYERS-BIASES-GRAPHS
OLD_CHECKPOINT_FILE = './results_gumbel_softmax/checkpoint/run10(batch-32-modes3(NOAA))' + '/encoder/encoder2000/encoder_model_e1999-i327040.ckpt'#327040 #'/encoder/encoder_model49599.ckpt'
NEW_CHECKPOINT_FILE = './results_gumbel_softmax/checkpoint/run10(batch-32-modes3(NOAA))' + '/test/encoder/encoder_model_e2000.ckpt' #'/encoder_new(2)/encoder_model_new49600.ckpt'
#OLD_CHECKPOINT_FILE = FLAGS.checkpoint_dir + '/model9999.ckpt'
#NEW_CHECKPOINT_FILE = FLAGS.checkpoint_dir + '/model_new/model_new9999.ckpt'

reader = tf.train.load_checkpoint(OLD_CHECKPOINT_FILE)
shape_from_key = reader.get_variable_to_shape_map()
dtype_from_key = reader.get_variable_to_dtype_map()
print(shape_from_key)
print('')
print(sorted(shape_from_key.keys()))
print('')
#print(dtype_from_key)

vars_to_rename = {
    "encoder_h1/kernel": "Encoder/dense/kernel",
    "encoder_h1/bias": "Encoder/dense/bias",
    "encoder_h2/kernel": "Encoder/dense_1/kernel",
    "encoder_h2/bias": "Encoder/dense_1/bias",
    "encoder_out/bias": "Encoder/dense_2/bias",
    "encoder_out/kernel": "Encoder/dense_2/kernel",


    #"_CHECKPOINTABLE_OBJECT_GRAPH": "lstm/basic_lstm_cell/bias",
}
new_checkpoint_vars = {}
reader1 = tf.train.NewCheckpointReader(OLD_CHECKPOINT_FILE)
for old_name in reader1.get_variable_to_shape_map():
    if old_name in vars_to_rename:
        new_name = vars_to_rename[old_name]
    else:
        new_name = old_name
    new_checkpoint_vars[new_name] = tf.Variable(reader1.get_tensor(old_name))

init = tf.global_variables_initializer()
saver = tf.train.Saver(new_checkpoint_vars)

with tf.Session() as sess:
    sess.run(init)
    saver.save(sess, NEW_CHECKPOINT_FILE)

# Read New_Checkpoint
reader2 = tf.train.load_checkpoint(NEW_CHECKPOINT_FILE)
shape_from_key2 = reader2.get_variable_to_shape_map()
dtype_from_key2 = reader2.get_variable_to_dtype_map()
print(shape_from_key2)
print('')
print(sorted(shape_from_key2.keys()))
#print(dtype_from_key2)


OLD_CHECKPOINT_FILE2 = './results_gumbel_softmax/checkpoint/run10(batch-32-modes3(NOAA))' + '/decoder/decoder2000/decoder_model_e1999-i327040.ckpt'  # 1999-49999.ckpt'
NEW_CHECKPOINT_FILE2 = './results_gumbel_softmax/checkpoint/run10(batch-32-modes3(NOAA))' + '/test/decoder/decoder_model_e2000.ckpt'  # 2000_new50000.ckpt'


reader2 = tf.train.load_checkpoint(OLD_CHECKPOINT_FILE2)
shape_from_key = reader2.get_variable_to_shape_map()
dtype_from_key = reader2.get_variable_to_dtype_map()
print(shape_from_key)
print('')
print(sorted(shape_from_key.keys()))
print('')

vars_to_rename = {
    "decoder_h1/kernel": "Policy/dense_3/kernel",
    "decoder_h1/bias": "Policy/dense_3/bias",
    "decoder_h2/kernel": "Policy/dense_4/kernel",
    "decoder_h2/bias": "Policy/dense_4/bias",
    "decoder_out/bias": "Policy/dense_5/bias",
    "decoder_out/kernel": "Policy/dense_5/kernel",

}
new_checkpoint_vars = {}
reader3 = tf.train.NewCheckpointReader(OLD_CHECKPOINT_FILE2)
for old_name in reader3.get_variable_to_shape_map():
    if old_name in vars_to_rename:
        new_name = vars_to_rename[old_name]
    else:
        new_name = old_name
    new_checkpoint_vars[new_name] = tf.Variable(reader3.get_tensor(old_name))

init = tf.global_variables_initializer()
saver2 = tf.train.Saver(new_checkpoint_vars)

with tf.Session() as sess:
    sess.run(init)
    saver2.save(sess, NEW_CHECKPOINT_FILE2)

# Read New_Checkpoint
reader4 = tf.train.load_checkpoint(NEW_CHECKPOINT_FILE2)
shape_from_key2 = reader4.get_variable_to_shape_map()
dtype_from_key2 = reader4.get_variable_to_dtype_map()
print(shape_from_key2)
print('')
print(sorted(shape_from_key2.keys()))
# print(dtype_from_key2)