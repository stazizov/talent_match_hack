from collections import defaultdict
from typing import Dict, List, Tuple, Any

import torch
import torchmetrics
import torch.nn.functional as F
from torch import Tensor
from torchmetrics import Metric, MeanMetric, AUROC, Accuracy
from xztrainer import XZTrainable, ContextType, BaseContext, DataType, ModelOutputsType


class ClassifiedRocAUCMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("pred", default=[], dist_reduce_fx="sum")
        self.add_state("target", default=[], dist_reduce_fx="sum")
        self.add_state("classes", default=[], dist_reduce_fx="sum")

    def update(self, model_outputs) -> None:
        self.classes += model_outputs['class']
        self.target += model_outputs['target']
        self.pred += model_outputs['pred']

    def compute(self) -> Any:
        packs = defaultdict(lambda: defaultdict(list))
        for clazz, pred, tgt in zip(self.classes, self.pred, self.target):
            packs[clazz.item()]['pred'].append(pred)
            packs[clazz.item()]['tgt'].append(tgt)
        packs = {k: {'pred': torch.stack(v['pred']), 'tgt': torch.stack(v['tgt']).to(torch.long)} for k, v in packs.items()}
        acc_scores = {}
        for thres in torch.arange(0.001, 0.999, 0.001):
            acc_score = ((torch.cat([x['pred'] for x in packs.values()]) >= thres) == torch.cat([x['tgt'] for x in packs.values()]).bool()).sum()/torch.cat([x['pred'] for x in packs.values()]).shape[0]
            acc_scores[thres.item()] = acc_score.item()
        print(sorted(acc_scores.items(), key=lambda x: x[1], reverse=True)[0])
         # {k: torch.stack([((v['pred'] >= thres) == v['tgt']).sum() / v['tgt'].shape[0] for thres in torch.arange(0, 1, 0.001)]).max() for k, v in packs.items()}
        packs = {k: torchmetrics.functional.auroc(v['pred'], v['tgt'], task='binary') for k, v in packs.items()}
        return torch.stack(list(packs.values())).mean()


class QTrainer(XZTrainable):
    def step(self, context: BaseContext, data: DataType) -> Tuple[Tensor, ModelOutputsType]:
        if context.context_type == ContextType.TRAIN:
            sims = [context.model(x) for x in data]
            sims = torch.stack(sims, dim=1)
            tgt = torch.tensor([0], dtype=torch.long, device=sims[0].device)
            loss_ct = F.cross_entropy(sims, tgt)
            loss_bce = F.binary_cross_entropy_with_logits(sims, torch.tensor([1] + [0] * (sims.shape[1] - 1), dtype=torch.float32, device=sims[0].device).unsqueeze(0))
            loss = loss_ct + loss_bce
            return loss, {
                'loss': loss,
                'target': tgt,
                'pred': torch.argmax(sims, dim=-1)
            }
        elif context.context_type == ContextType.EVAL:
            sim, logits = context.model(data, out_logits=True)
            return None, {
                'class': data['class'],
                'target': data['target'],
                'pred': F.sigmoid(sim),
                'id': data['id'],
                'logits': logits
            }
        else:
            sim, logits = context.model(data, out_logits=True)
            return None, {
                'class': data['class'],
                'pred': F.sigmoid(sim),
                'id': data['id'],
                'logits': logits
            }

    def create_metrics(self, context_type: ContextType) -> Dict[str, Metric]:
        if context_type == ContextType.TRAIN:
            return {
                'loss': MeanMetric(),
                'accuracy': Accuracy('binary'),
            }
        else:
            return {
                'auc': ClassifiedRocAUCMetric()
            }

    def update_metrics(self, context_type: ContextType, model_outputs: Dict[str, List], metrics: Dict[str, Metric]):
        if context_type == ContextType.TRAIN:
            metrics['loss'].update(model_outputs['loss'])
            metrics['accuracy'].update((model_outputs['pred'] == model_outputs['target']).to(torch.float32), (model_outputs['target'] + 1).to(torch.float32))
        else:
            metrics['auc'].update(model_outputs)