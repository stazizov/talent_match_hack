import json
import sys

import pandas as pd
import transformers
from transformers import AdamW
from xztrainer import XZTrainer, XZTrainerConfig, SchedulerType, CheckpointType, ContextType
from xztrainer.logger.tensorboard import TensorboardLoggingEngineConfig
from xztrainer.setup_helper import set_seeds, enable_tf32

from dataset import Collator, QFormerDataset
from qformer import QFormer
from trainer import QTrainer

if __name__ == '__main__':
    set_seeds(0x1337AF)
    enable_tf32()

    model = QFormer()
    trainer = XZTrainer(XZTrainerConfig(
        batch_size=1,  # currently dont change
        accumulation_batches=4,
        batch_size_eval=1,
        epochs=1,
        experiment_name='master',
        optimizer=lambda x: AdamW(
            x.parameters(), 5e-5,
            # weight_decay=0.01,
            # use_triton=True
        ),
        scheduler=lambda opt, steps: transformers.get_linear_schedule_with_warmup(opt, int(steps * 0.1), steps),
        scheduler_type=SchedulerType.STEP,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        dataloader_shuffle_train_dataset=True,
        print_steps=10,
        eval_steps=20,
        skip_nan_loss=True,
        save_steps=20,
        save_keep_n=20,
        collate_fn=Collator(),
        gradient_clipping=3,
        logger=TensorboardLoggingEngineConfig()
    ), model, QTrainer())
    mode = sys.argv[1]
    if mode == 'train':
        with open('data/resumes.json', 'r') as f:
            data = json.load(f)
        data_train = data[:-2]
        data_test = data[-2:]
        train_ds = QFormerDataset(data_train, ContextType.TRAIN)
        val_ds = QFormerDataset(data_test, ContextType.EVAL)
        print(len(train_ds), len(val_ds))
        trainer.train(train_ds, val_ds)
    elif mode == 'eval':
        with open('data/resumes.json', 'r') as f:
            data = json.load(f)
        inf_ds = QFormerDataset(data, ContextType.EVAL)
        trainer.load_model_checkpoint('checkpoint/working.pt',  CheckpointType.XZTRAINER)
        outs, _ = trainer.infer(inf_ds)
        outs = [{'id': a, 'logits': b.numpy().tolist(), 'pred': c.item()} for a, b, c in zip(outs['id'], outs['logits'], outs['pred'])]
        outs = pd.DataFrame(outs)
        outs.to_pickle('logits.pkl')
        print(outs)
    elif mode == 'infer':
        with open('data/case_2_reference_without_resume_sorted.json', 'r') as f:
            data = json.load(f)
            data = [data]
        inf_ds = QFormerDataset(data, ContextType.INFERENCE)
        trainer.load_model_checkpoint(
            'checkpoint/working.pt',
            CheckpointType.XZTRAINER)
        outs, _ = trainer.infer(inf_ds)
        outs = [{'id': a, 'logits': b.numpy().tolist(), 'pred': c.item()} for a, b, c in
                zip(outs['id'], outs['logits'], outs['pred'])]
        outs = pd.DataFrame(outs)
        outs.to_pickle('logits_test.pkl')
        print(outs)
    else:
        print('Unknown mode')