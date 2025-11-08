import os
import json
import torch
import gzip
import random
import numpy as np
from torch.utils.data import IterableDataset

class RedPajamaDataset(IterableDataset):
    def __init__(self, split=None, 
                 root="./data/redpajama", 
                 tokenizer='gpt2', 
                 tokenizer_path=None,
                 min_length=256,
                 max_length=768,
                 num_tokens=768,
                 end_special_token="",
                 languages=['en'],
                 snapshots=["2023-14", "2023-06", "2022-49", "2022-40"],
                 partition='head_middle',
                 shuffle=True,
                 target_is_context=False,
                 shuffle_first_context=False,
                 overlap_contexts=False,
                 ):
        '''
        split: ['sentence', 'random']
        '''
        self.root = root
        self.snapshots = snapshots
        self.partition = partition
        self.languages = languages
        self.max_length = max_length
        self.min_length = min_length
        self.num_tokens = num_tokens
        self.end_special_token = end_special_token
        if tokenizer == 'gpt2':
            from transformers import GPT2Tokenizer
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        elif tokenizer == 'llama':
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        else:
            self.tokenizer = tokenizer
        self.target_is_context = target_is_context
        self.shuffle_first_context = shuffle_first_context
        self.overlap_contexts = overlap_contexts

        all_paths = []

        for snapshot in self.snapshots:

            for language in self.languages:
                
                # load all paths from txt file f"{language}-{snapshot}-{self.partition}.txt"
                with open(f"{self.root}/listings/{language}-{snapshot}-{self.partition}.txt", "r") as f:
                    paths = f.readlines()
                
                paths = [path.strip() for path in paths]

                all_paths.extend(paths)

        if shuffle:
            # permute the paths:
            random.shuffle(all_paths)

        self.all_paths = all_paths

        # take a pass over all the data to get the meta information

    def get_context_and_sentence(self, doc, return_doc=False):
        """
        将单个文档分割成上下文(contexts)和目标句子(sentence)，为 MemoryLLM 的记忆训练准备数据

          输入文档: "The quick brown fox jumps over the lazy dog. Python is a programming language..."

        正常模式输出:
        - contexts: [tokens_1, tokens_2, ...] (前面的部分)
        - sentence: [tokens_last] (最后部分)
        - label: 1 (相关上下文)

        上下文目标模式输出:
        - contexts: [tokens_1, tokens_2, ...] (组合上下文)
        - sentence: tokens_selected (随机选择)
        - label: 0 或 2 (根据选择策略)
        """
        # 1. 方法核心功能
        
        # 2. 输入处理
        # tokenizer doc:
        doc = self.tokenizer(doc + self.end_special_token, return_tensors='pt', truncation=False, add_special_tokens=False).input_ids[0]

        # 3. 特殊返回模式
        # 用于验证场景，直接返回整个文档作为单个上下文
        if return_doc:
            doc = doc[:self.max_length]
            attention_masks = torch.ones(self.num_tokens + len(doc))
            return [doc], [attention_masks], doc, attention_masks

        # get the context and sentence:

        # 4. 两种核心数据生成模式

        # 4.1 上下文目标模式 (target_is_context=True) - Sanity Check
        # 用途: 用于验证模型性能的对照实验
        if self.target_is_context:
            # For Sanity Check Experiments
            # 短文档处理：简单二分
            if len(doc) < self.max_length / 2 + self.min_length / 2:
                index = len(doc) // 2
                contexts = [doc[:index], doc[index:]]
            # 长文档处理：滑动窗口分割
            else:
                contexts = []
                while len(doc) >= self.max_length / 2:
                    context_length = random.randint(self.min_length / 2, self.max_length / 2)
                    contexts.append(doc[:context_length])
                    doc = doc[context_length:] 
                if len(doc) >= self.min_length / 2:
                    contexts.append(doc)

            if self.overlap_contexts:
                # # 创建重叠上下文: [0:1], [1:2], [2:3]...
                contexts = [torch.cat([contexts[i], contexts[i+1]]) for i in range(len(contexts) - 1)]
            else:
                if len(contexts) == 1:
                    pass
                else:
                # # 创建配对上下文: [0:1], [2:3], [4:5]...
                    contexts = [torch.cat([contexts[i*2], contexts[i*2+1]]) for i in range(len(contexts) // 2)]

            # 随机选择目标句子
            if self.shuffle_first_context:
                if np.random.rand() > 0.5:
                    sentence = contexts[0]
                    label = 0
                else:
                    sentence = contexts[1] if len(contexts) > 1 else contexts[0]
                    label = 2
            else:
                sentence = contexts[0]
                label = 0

        # 4.2 正常训练模式 (target_is_context=False) - Standard Training
        # 用途: 标准的记忆增强训练
        else:
            label = 1
            # Normal Training Path
            if len(doc) < self.max_length + self.min_length:
                index = len(doc) // 2
                contexts = [doc[:index], doc[index:]]
            else:
                contexts = []
                while len(doc) >= self.max_length:
                    context_length = random.randint(self.min_length, self.max_length)
                    contexts.append(doc[:context_length])
                    doc = doc[context_length:] 
                if len(doc) >= self.min_length:
                    contexts.append(doc)
            # 最后一个片段作为目标句子
            sentence = contexts[-1]
            # 前面的所有片段作为上下文
            contexts = contexts[:-1]
        
        assert len(contexts) > 0

        base_attention_masks = torch.ones(self.num_tokens)

        # Shape: [[num_tokens + context_length]]
        contexts_masks = [
            torch.cat([base_attention_masks, torch.ones(len(context_ids))]) for context_ids in contexts
        ]
        # Shape: [num_tokens + sentence_length]
        sentence_mask = torch.cat([base_attention_masks, torch.ones(len(sentence))])
        if self.target_is_context:
            return contexts, contexts_masks, sentence, sentence_mask, label
        else:
            return contexts, contexts_masks, sentence, sentence_mask, label


    def __iter__(self):

        for path in self.all_paths:

            if not os.path.exists(f"{self.root}/documents/{path}.json.gz"):
                print("WARNING!!" * 100)
                print(f"Path {path} does not exist!")
                continue

            with gzip.open(f"{self.root}/documents/{path}.json.gz") as f:
                for line in f:
                    data = json.loads(line)
                    doc = data['raw_content']
                    output = self.get_context_and_sentence(doc)
                    if output is not None:
                        contexts, contexts_masks, sentence, sentence_mask, label = output
                        yield contexts, contexts_masks, sentence, sentence_mask, label
                        # yield contexts, sentence


if __name__ == '__main__':

    def worker_init_fn(worker_id):
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset # the dataset copy in this worker process
        all_dataset = len(dataset.all_paths)
        per_worker = all_dataset // worker_info.num_workers
        dataset.all_paths = dataset.all_paths[worker_id * per_worker: (worker_id + 1) * per_worker]

    dataset = RedPajamaDataset(root="../../../data/redpajama", 
                               snapshots=['2022-40'],
                               end_special_token='</s>',
                               tokenizer='llama',
                               tokenizer_path='openlm-research/open_llama_3b_v2')

    import time

    count = 1000

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)

    start = time.time()
    for i, data in enumerate(dataset):
        contexts, contexts_masks, sentence, sentence_mask = data
        import ipdb; ipdb.set_trace()
        if i > count:
            break
    end = time.time()
    print("Time taken: ", end - start)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=16, worker_init_fn=worker_init_fn)
    start = time.time()
    for i, data in enumerate(dataloader):
        contexts, contexts_masks, sentence, sentence_mask = data
        if i > count:
            break
    end = time.time()
    print("Time taken: ", end - start)
