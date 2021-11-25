import os
import json
import logging
import random
import argparse
import pickle
from tqdm import tqdm, trange
import tensorflow as tf
import numpy as np
from tensorflow import keras
import tensorflow_addons as tfa
from decoder import Decoder
from model_fei2 import Seq2Seq
from transformers import RobertaConfig, TFRobertaModel, RobertaTokenizer

MODEL_CLASSES = {'roberta': (RobertaConfig, TFRobertaModel, RobertaTokenizer)}

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class Example(object):
    """A single training/test example."""
    def __init__(self,
                 idx,
                 source,
                 target,
                 ):
        self.idx = idx
        self.source = source
        self.target = target
def read_examples(filename):
    """Read examples from filename."""
    examples=[]
    with open(filename,encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line=line.strip()
            js=json.loads(line)
            if 'idx' not in js:
                js['idx']=idx
            code=' '.join(js['code_tokens']).replace('\n',' ')
            code=' '.join(code.strip().split())
            nl=' '.join(js['docstring_tokens']).replace('\n','')
            nl=' '.join(nl.strip().split())
            examples.append(
                Example(
                        idx = idx,
                        source=code,
                        target = nl,
                        )
            )
    return examples


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 source_mask,
                 target_mask,

                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask
def convert_examples_to_features(examples, tokenizer, args, stage=None):
    features = []
    for example_index, example in enumerate(examples):
        # source
        source_tokens = tokenizer.tokenize(example.source)[:args.max_source_length - 2]  # 就算超过了max_source_lenth，就进行截断
        source_tokens = [tokenizer.cls_token] + source_tokens + [tokenizer.sep_token]  # 每个sentence前后加上 CLS 和 SEP
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)  # 使用词典，将 word 转换为 数字
        source_mask = [1] * (len(source_tokens))  # 得到的是[1,1,1,……]，用于标记是根据数据来的，还是填充的
        padding_length = args.max_source_length - len(source_ids)  # 要填充的长度
        source_ids += [tokenizer.pad_token_id] * padding_length  # 用pad_token_id 填充，追加到source_id的后面
        source_mask += [0] * padding_length  # 标记说明是填充的

        # target
        if stage == "test":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(example.target)[:args.max_target_length - 2]  # 避免超长，进行截断
        target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]  # 添加 CLS 和SEP
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] * len(target_ids)
        padding_length = args.max_target_length - len(target_ids)
        target_ids += [tokenizer.pad_token_id] * padding_length
        target_mask += [0] * padding_length

        if example_index < 5:  # 日志相关
            if stage == 'train':
                logger.info("*** Example ***")
                logger.info("idx: {}".format(example.idx))

                logger.info("source_tokens: {}".format([x.replace('\u0120', '_') for x in source_tokens]))
                logger.info("source_ids: {}".format(' '.join(map(str, source_ids))))
                logger.info("source_mask: {}".format(' '.join(map(str, source_mask))))

                logger.info("target_tokens: {}".format([x.replace('\u0120', '_') for x in target_tokens]))
                logger.info("target_ids: {}".format(' '.join(map(str, target_ids))))
                logger.info("target_mask: {}".format(' '.join(map(str, target_mask))))

        features.append(
            InputFeatures(
                example_index,
                source_ids,
                target_ids,
                source_mask,
                target_mask,
            )
        )
    return features

def loss_function(real, pred):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

def set_seed(seed):
    """set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_target_length', default=128, type=int)
    parser.add_argument('--max_source_length', default=256, type=int)
    args = parser.parse_args()
    logger.info(args)

    set_seed(seed=42) #设置随机种子
    config_class, model_class, tokenizer_class = MODEL_CLASSES['roberta']
    config = config_class.from_pretrained('roberta-base')
    tokenizer = tokenizer_class.from_pretrained('roberta-base')
    encoder = model_class.from_pretrained('roberta-base', config=config)
    decoder = Decoder(num_layers=6, d_model=config.hidden_size, num_heads=config.num_attention_heads, dff=2048,
                      target_vocab_size=tokenizer.vocab_size, maximum_position_encoding=514)  # d_model是输入的维度 nhead是multi-attention的head数
    model = Seq2Seq(encoder, decoder, config=config, beam_size=10, max_length=args.max_target_length,
                  sos_id=tokenizer.cls_token_id, eos_id=tokenizer.sep_token_id)

    # 做训练
    train_file = 'E:\lab_related\CodeBERT_tf\data\CodeSearchNet\php\\train2.jsonl'
    train_examples = read_examples(train_file)
    valid_file = 'E:\lab_related\CodeBERT_tf\data\CodeSearchNet\php\\valid2.jsonl'
    valid_examples = read_examples(valid_file)
    train_features = convert_examples_to_features(train_examples, tokenizer, args, stage='train')
    valid_features = convert_examples_to_features(valid_examples, tokenizer, args, stage='valid')
    all_source_ids = tf.convert_to_tensor([f.source_ids for f in train_features])  # 将所有的source_ids整合在一个tensor中，下面的也是一样
    all_source_mask = tf.convert_to_tensor([f.source_mask for f in train_features])
    all_target_ids = tf.convert_to_tensor([f.target_ids for f in train_features])
    all_target_mask = tf.convert_to_tensor([f.target_mask for f in train_features])

    all_source_ids2 = tf.convert_to_tensor([f.source_ids for f in valid_features])  # 将所有的source_ids整合在一个tensor中，下面的也是一样
    all_source_mask2 = tf.convert_to_tensor([f.source_mask for f in valid_features])
    all_target_ids2 = tf.convert_to_tensor([f.target_ids for f in valid_features])
    all_target_mask2 = tf.convert_to_tensor([f.target_mask for f in valid_features])


    num_train_optimization_steps = 50000
    optimizer = tfa.optimizers.AdamW(learning_rate=5e-05, weight_decay=0)
    #开始训练
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", 64)
    logger.info("  Num epoch = %d", num_train_optimization_steps * 64 // len(train_examples))

    BUFFER_SIZE = len(train_features)
    BATCH_SIZE = 8
    bar = tqdm(range(num_train_optimization_steps),total=num_train_optimization_steps)
    nb_tr_examples, nb_tr_steps, tr_loss, global_step, best_bleu, best_loss = 0, 0, 0, 0, 0, 1e6

    dataset = tf.data.Dataset.from_tensor_slices((all_source_ids, all_source_mask, all_target_ids, all_target_mask))
    data_batch = dataset.batch(batch_size=BATCH_SIZE)
    dataset2 = tf.data.Dataset.from_tensor_slices((all_source_ids2, all_source_mask2, all_target_ids2, all_target_mask2))
    data_batch2 = dataset2.batch(batch_size=BATCH_SIZE)
    sample = next(iter(data_batch))
    model.compile(optimizer=optimizer, loss=loss_function, metrics=['loss', 'accuracy'])
    model.fit(data_batch, epochs=50000, validation_data=data_batch2, validation_freq=500)
    model.evaluate(data_batch2)
    sample = next(iter(data_batch2))


if __name__ == "__main__":
    main()