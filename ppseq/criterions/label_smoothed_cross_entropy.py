from paddlenlp.transformers import CrossEntropyCriterion
from ppseq.criterions import PaddleseqCriterion


import math
from dataclasses import dataclass, field

# import torch
# from fairseq import metrics, utils
from ppseq.criterions import PaddleseqCriterion, register_criterion
from ppseq.dataclass import PaddleseqDataclass
from omegaconf import II


@dataclass
class LabelSmoothedCrossEntropyCriterionConfig(PaddleseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    # report_accuracy: bool = field(
    #     default=False,
    #     metadata={"help": "report accuracy metric"},
    # )
    # ignore_prefix_size: int = field(
    #     default=0,
    #     metadata={"help": "Ignore first N tokens"},
    # )
    # sentence_avg: bool = II("optimization.sentence_avg")


@register_criterion("label_smoothed_cross_entropy", dataclass=LabelSmoothedCrossEntropyCriterionConfig)
class LabelSmoothedCrossEntropyCriterion(CrossEntropyCriterion,PaddleseqCriterion):
    def __init__(self,
                 label_smooth_eps=None,
                 pad_idx=1):
        super(LabelSmoothedCrossEntropyCriterion,self).__init__(label_smooth_eps,pad_idx)

    def forward(self, model, sample ,need_attn=False):
        logits, attn = model(sample["src_tokens"], sample["prev_tokens"])
        sum_cost, avg_cost, token_num = super().forward(logits, sample["tgt_tokens"])
        if not need_attn:
            return logits, sum_cost, avg_cost, token_num
        else:
            return logits, sum_cost, avg_cost, token_num, attn