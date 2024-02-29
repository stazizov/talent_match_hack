import streamlit as st
import json
from utils import read_json, transform_data
import pandas as pd
from typing import List
import random
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

import torch 



def process_item(features):
    print(features)
    return random.uniform(0, 1)

def get_probas(inputs:pd.DataFrame) -> List[float]: 
    return [process_item(x) for x in inputs.iterrows()]
    
def arange_resumes(inputs:pd.DataFrame, probas:List[float], threshold=0.5): 
    inputs["pred_score"] = probas
    inputs = inputs.sort_values(by="pred_score", ascending=False)
    inputs['status'] = inputs.pred_score.apply(lambda a: "success" if a > threshold else "fail")
    inputs = inputs[["status", "pred_score"]+list(inputs.columns)[:-2]]
    return inputs

def main():
    st.title("MISIS 4️⃣2️⃣")

    uploaded_file = st.file_uploader("Choose a JSON file", type=["json"])


    set_seeds(0x1337AF)
    enable_tf32()


    # checkpoint = torch.load("./checkpoint/working.pt")
    model = QFormer()
    # model.load_state_dict(checkpoint['model_state_dict'])
    
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
    
    THRESHOLD = 0.12399999797344208
    
    if uploaded_file is not None:
        st.sidebar.info("File uploaded successfully!")
        
        try:
            data = json.load(uploaded_file)
            inf_ds = QFormerDataset(data, ContextType.EVAL)
            trainer.load_model_checkpoint('checkpoint/working.pt',  CheckpointType.XZTRAINER)
            outs, _ = trainer.infer(inf_ds)
            outs = [
                {
                    'confirmed': c.item() > THRESHOLD, 
                    'id': a, 
                    'score': c.item()
                } for a, b, c in zip(outs['id'], outs['logits'], outs['pred'])
                ]
            
            outs = pd.DataFrame(outs)
            outs = outs.sort_values(by="score", ascending=False)
            st.write(outs)
        except:
            st.warning("wrong input data format")

if __name__ == "__main__":
    main()
