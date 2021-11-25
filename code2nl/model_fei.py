import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from tensorflow import keras
from loss import CrossEntropyLoss


class Seq2Seq(tf.Module):
    def __init__(self, encoder, decoder, config, beam_size=None, max_length=None, sos_id=None, eos_id=None):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.beam_size = beam_size
        self.max_length = max_length
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.bias = tf.linalg.band_part(tf.ones(shape=[2048,2048]), -1, 0)
        self.lm_head = tf.keras.layers.Dense(50265, use_bias=False)
        self.dense = tf.keras.layers.Dense(768)
        #self.embeds = tf.keras.layers.Embedding(input_dim=128, output_dim=128)


    def call(self, source_ids=None, source_mask=None, target_ids=None, target_mask=None, args=None):
        output = self.encoder(source_ids, attention_mask=source_mask)
        encoder_output = tf.transpose(output[0], perm=[1,0,2])
        if target_ids is not None:
            attn_mask = -1e4 * (1 - self.bias[:target_ids.shape[1], :target_ids.shape[1]])
            out = self.decoder(target_ids, encoder_output, training=False, look_ahead_mask=attn_mask, padding_mask=None)
            #hidden_states = tf.transpose(tf.keras.activations.tanh(self.dense(out)), perm=[1,0,2])
            hidden_states = tf.keras.activations.tanh(self.dense(out))
            lm_logits = self.lm_head(hidden_states)
            active_loss = tf.reshape(tf.not_equal(target_mask[..., 1:], 0), -1) == True
            shift_logits = lm_logits[..., :-1, :]
            shift_labels = target_ids[..., 1:]
            y_true = tf.reshape(shift_labels, -1)[active_loss]
            y_pred = tf.reshape(shift_logits, [-1, 50265])[active_loss]
            print('sum0:', y_pred.dtype)
            print('sum1:', y_true.dtype)
            #temp1 = tf.nn.log_softmax(y_pred, axis=1)
            loss_fun = CrossEntropyLoss()
            loss = loss_fun(y_pred, y_true)
           # loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y_true, y_pred=y_pred)#这里的交叉熵有很大的问题
           #  print('loss:', loss)
            output = loss#, np.sum(loss*active_loss), np.sum(active_loss)
            return output
        # else:
        #     #预测
        #     preds = []
