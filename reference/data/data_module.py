import pytorch_lightning as pl
from transformers import DataCollatorForLanguageModeling, AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset

class LLMDataModule(pl.LightningDataModule):
    def __init__(self, model_name, batch_size, max_seq_len, num_workers):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_workers = num_workers
    
    def prepare_data(self):
        load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir='./data')

    def setup(self, stage=None):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False, return_tensors='pt')

        raw_datasets = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir='./data')

        def tokenize(element):
            outputs = self.tokenizer(
                element["text"],
                truncation=True,
                max_length=self.max_seq_len,
                return_overflowing_tokens=True,
                return_length=True,
            )
            input_batch = []
            for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
                if length == self.max_seq_len:
                    input_batch.append(input_ids)
            return {"input_ids": input_batch}


        tokenized_datasets = raw_datasets.map(
            tokenize, batched=True, remove_columns=raw_datasets["train"].column_names
        )

        self.train_dataset = tokenized_datasets['train']
        self.val_dataset = tokenized_datasets['validation']
        self.test_dataset = tokenized_datasets['test']

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=self.data_collator, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.data_collator, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self.data_collator, num_workers=self.num_workers, pin_memory=True)
