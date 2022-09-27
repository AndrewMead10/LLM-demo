from data.data_module import LLMDataModule
from datasets import load_dataset


from pytorch_lightning import Trainer
from pytorch_lightning.loggers.wandb import WandbLogger

from model.model import SmallTextGenerator
from model.pl_model import SmallTextGeneratorModule


def main():
    # Data
    data_module = LLMDataModule(
        model_name="gpt2",
        batch_size=8,
        max_seq_len=128,
        num_workers=0
    )

    # Model
    model = SmallTextGenerator(
        d_model=128,
        vocab_size=50257,
        num_layers=1,
        num_heads=4,
        dim_feedforward=512,
        dropout=0.1,
    )
    print(sum([p.numel() for p in model.parameters()]))

    model_module = SmallTextGeneratorModule(model)

    # Training
    trainer = Trainer(
        logger=WandbLogger(project="LLM-demo", name="test"),
        max_epochs=3,
        accelerator='gpu',
        
    )
    trainer.fit(model_module, data_module)

if __name__=="__main__":
    main()