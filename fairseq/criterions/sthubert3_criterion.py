# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import re
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.logging import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
import logging
logger = logging.getLogger(__name__)

@dataclass
class STHubertCriterionConfig3(FairseqDataclass):
    pred_masked_weight: float = field(
        default=1.0,
        metadata={"help": "weight for predictive loss for masked frames"},
    )
    pred_nomask_weight: float = field(
        default=0.0,
        metadata={"help": "weight for predictive loss for unmasked frames"},
    )
    loss_weights: Optional[List[float]] = field(
        default=None,
        metadata={"help": "weights for additional loss terms (not first one)"},
    )
    log_keys: List[str] = field(
        default_factory=lambda: [],
        metadata={"help": "output keys to log"},
    )
    text_ctc_weight: float = field(
        default=0.1,
        metadata={"help": "weights for text CTC Loss, loss will be (hubert_loss + text_weight * CTC_loss))"},
    )
    text_mum_weight: float = field(
        default=0.0,
        metadata={"help": "masked unit modeling weight from the text end"},
    )
    d2v_loss_weight:  float = field(
        default=0.001,
        metadata={"help": "contextualized loss (idea from data2vec) weight from the speech end"},
    )

@register_criterion("sthubert3_criterion", dataclass=STHubertCriterionConfig3)
class STHubertCriterion3(FairseqCriterion):
    def __init__(
        self,
        task,
        pred_masked_weight,
        pred_nomask_weight,
        loss_weights=None,
        log_keys=None,
        text_ctc_weight=0.1,
        text_mum_weight=0,
        d2v_loss_weight=0.001,
        no_ctc_blank=False,
        
    ):
        super().__init__(task)
        self.pred_masked_weight = pred_masked_weight
        self.pred_nomask_weight = pred_nomask_weight
        self.loss_weights = loss_weights
        self.log_keys = [] if log_keys is None else log_keys
        self.text_ctc_weight = text_ctc_weight
        self.text_mum_weight = text_mum_weight
        self.d2v_loss_weight = d2v_loss_weight
        self.no_ctc_blank= no_ctc_blank
        self.padding_idx = task.dictionaries[0].pad()
        self.eos_idx = task.dictionaries[0].eos()
        self.blank_idx = task.dictionaries[0].bos()
    def compute_hubert_loss(
        self, model, logp_m_list, targ_m_list, logp_u_list,targ_u_list, reduce=True, log_pred=False, suffix=""
    ):
        """Compute the mask lm style loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        loss = 0.0
        sample_size = 0
        logging_output = {}
        reduction = "sum" if reduce else "none"

        loss_m_list = []
        #logp_m_list = model.get_logits(net_output, True)
        #targ_m_list = model.get_targets(net_output, True)
        assert self.pred_masked_weight == 0 or len(logp_m_list) > 0, f"len(logp_m_list): {len(logp_m_list)}, logp_m_list: {logp_m_list}"
        for i, (logp_m, targ_m) in enumerate(zip(logp_m_list, targ_m_list)):
            loss_m = F.cross_entropy(logp_m, targ_m, reduction=reduction)
            loss_m_list.append(loss_m)
            logging_output[f"loss_m_{i}{suffix}"] = loss_m.detach().item()
        if self.pred_masked_weight > 0:
            loss += self.pred_masked_weight * sum(loss_m_list)
            sample_size += targ_m_list[0].numel()
        loss_u_list = []
        #logp_u_list = model.get_logits(net_output, False)
        #targ_u_list = model.get_targets(net_output, False)
        assert self.pred_nomask_weight == 0 or len(logp_u_list) > 0
        for i, (logp_u, targ_u) in enumerate(zip(logp_u_list, targ_u_list)):
            loss_u = F.cross_entropy(logp_u, targ_u, reduction=reduction)
            loss_u_list.append(loss_u)
            logging_output[f"loss_u_{i}{suffix}"] = loss_u.detach().item()
        if self.pred_nomask_weight > 0:
            loss += self.pred_nomask_weight * sum(loss_u_list)
            sample_size += targ_u_list[0].numel()


        def compute_correct(logits):
            if logits.numel() == 0:
                return 0, 0
            else:
                assert logits.dim() > 1, logits.shape
                max = logits.argmax(-1) == 0
                min = logits.argmin(-1) == 0
                both = max & min
                corr = max.long().sum().item() - both.long().sum().item()
                count = max.numel()
                return corr, count
        with torch.no_grad():
            for i, logp_m in enumerate(logp_m_list):
                corr_m, count_m = compute_correct(logp_m)
                logging_output[f"correct_m_{i}{suffix}"] = corr_m
                logging_output[f"count_m_{i}{suffix}"] = count_m

            for i, logp_u in enumerate(logp_u_list):
                corr_u, count_u = compute_correct(logp_u)
                logging_output[f"correct_u_{i}{suffix}"] = corr_u
                logging_output[f"count_u_{i}{suffix}"] = count_u

        return loss, sample_size, logging_output
 
    def forward(self, model, sample, reduce=True, log_pred=False):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        reduction = "sum" if reduce else "none"
        loss_speech=0
        # 1.1. mask lm loss in speech part, in other words, do hubert forward and compute loss
        #sample["net_input"][
        #    "source_text"
        #] = None  ## drop text sample do hubert forward and compute loss
        net_output = model(target_list=sample["target_list"], **sample["net_input"])
        logp_m_list = model.get_logits(net_output, True) ## it has three elements, 
                                                         #first two elements for speech,
                                                         # last elements for text
        targ_m_list = model.get_targets(net_output, True)
        logp_u_list = model.get_logits(net_output, False)
        targ_u_list = model.get_targets(net_output, False)
        
        loss_speech, sample_size, logging_output = self.compute_hubert_loss(
            model,
            logp_m_list[:2],
            targ_m_list[:2],
            logp_u_list[:2],
            targ_u_list[:2],
            reduce,
            suffix="",
        )
        # 1.2. extra_loss for speech part.
        if self.loss_weights is not None:
            assert hasattr(model, "get_extra_losses")
            extra_losses, names = model.get_extra_losses(net_output)
            if torch.is_tensor(extra_losses):
                extra_losses = [extra_losses]
                names = [names]
            if len(self.loss_weights) == 1 and len(extra_losses) != 1:
                self.loss_weights = [self.loss_weights[0]] * len(extra_losses)
            assert len(extra_losses) == len(
                self.loss_weights
            ), f"{len(extra_losses)}, {len(self.loss_weights)}"
            for p, n, coef in zip(extra_losses, names, self.loss_weights):
                if coef != 0 and p is not None:
                    p = coef * p.float() * sample_size
                    loss_speech += p
                    logging_output[f"loss_{n}"] = p.item()
        #### add net_output other loss, for example: embedding_l2_loss
        ### add some display information into logging system
        for lk in self.log_keys:
            #logger.info(f"self.log_keys contain : {lk}")
            if lk in net_output:
                logging_output[lk] = float((net_output[lk]))
            #elif lk in net_output["result_speech"]:
            #    logging_output[lk] = float((net_output[lk])) 
        #loss_speech  =  loss_speech/sample_size
        ##1.3 d2v loss
        scaled_losses = {}
        losses = {}
        if isinstance(net_output, dict) and "losses" in net_output["result_speech"]:
            losses = net_output["result_speech"]["losses"]
        for lk, p in losses.items():
            #logger.info(f"lk: {lk}, p: {p}")
            #logging_output[lk] = p.float().sum()/p.size(0)
            #scaled_losses[lk] = p.float().sum()/p.size(0)
            logging_output[lk] = p.float().sum()/sample_size  * self.d2v_loss_weight ## for display
            scaled_losses[lk] = p.float().sum()/sample_size  * self.d2v_loss_weight
        if "ema_decay" in net_output["result_speech"]:
            logging_output["ema_decay"] = net_output["result_speech"]["ema_decay"]

        '''        
        for lk in  net_output["result_speech"].keys():
            if lk.startswith("target_var_"):
                logging_output[lk] = net_output["result_speech"][lk] 
            elif lk.startswith("pred_var_"):  
                logging_output[lk] = net_output["result_speech"][lk] 
        '''
        loss_d2v = sum(scaled_losses.values())
        loss_speech  =  loss_speech/sample_size + loss_d2v
       

        # 2.1. do text part forward and loss computation
        loss_text=0
        ## mask lm loss
        if self.text_mum_weight > 0:
            loss_u2u, sample_size_u2u, logging_output_u2u = self.compute_hubert_loss(
                model,
                logp_m_list[2:],
                targ_m_list[2:],
                logp_u_list[2:],
                targ_u_list[2:],
                reduce,
                suffix="_u2u",
            )
            loss_text = loss_u2u
            loss_text = self.text_mum_weight * loss_text / sample_size_u2u
            logging_output.update(logging_output_u2u)
        #(FIXME) Can I get text token number from dataset at text part
        #text_sample_size = sample_size_u2u
        if self.text_ctc_weight > 0:
            #logger.info("I am here , it is ctc loss")
            text_sample_size=sample["text_ntokens_list"][0]
            #logger.info(f"text_sample_size: {text_sample_size}")
            text_ctc_loss = self.compute_ctc_loss(
                model, net_output, sample["net_input"]["source_text"][0], reduction=reduction
            )
            loss_text += (
                self.text_ctc_weight * text_ctc_loss / text_sample_size
            )
            logging_output["text_ctc_loss"] = utils.item(loss_text)
            logging_output["text_sample_size"] = text_sample_size
            #logger.info(f'logging_output["text_sample_size"] : {logging_output["text_sample_size"]}')
        loss = loss_speech + loss_text
        logging_output = {
            "loss": utils.item(loss) if reduce else loss,
            "ntokens": sample_size,
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
            **logging_output,
        }

        return loss, sample_size, logging_output
    def compute_ctc_loss(self, model, net_output, target, reduction):
        logits = net_output["result_text"]["shared_encoder_out_ctc"][0].permute(1,0,2)  # (T, B, C) from the coshared_transformer encoder
        #logger.info(f"logits shape: {logits.shape}, logits : {logits}")
        if self.no_ctc_blank:
            ## set prob of <blank> to -inf
            logits = logits.float()
            logits[:, :, self.blank_idx] = -1000000.0
        
        lprobs = F.log_softmax(logits.float(), dim=-1)
        #logger.info(f"lprobs shape: {lprobs.shape}, lprobs : {lprobs}")
        encoder_padding_mask = net_output["result_text"]["shared_encoder_padding_mask_ctc"][0]
        #logger.info(f"encoder_padding_mask: {encoder_padding_mask}, its shape:{encoder_padding_mask.shape}")
        non_padding_mask = ~encoder_padding_mask
        input_lengths = non_padding_mask.long().sum(-1)
        #logger.info(f"input_lengths: {input_lengths}")
        pad_mask = (target != self.padding_idx) & (target != self.eos_idx)
        targets_flat = target.masked_select(pad_mask)
        target_lengths = pad_mask.sum(-1)

        with torch.backends.cudnn.flags(enabled=False):
            loss = F.ctc_loss(
                lprobs,
                targets_flat,
                input_lengths,
                target_lengths,
                blank=self.blank_idx,
                reduction=reduction,
                zero_infinity=True,
            )
        return loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training (copied from normal cross entropy)."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        #text_sample_size = sum(log.get("text_sample_size", 0) for log in logging_outputs)


        metrics.log_scalar(
            "loss", loss_sum / math.log(2), sample_size, round=3
        )
        #if sample_size != ntokens:
        #    metrics.log_scalar(
        #        "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
        #    )
        #    metrics.log_derived(
        #        "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        #    )
        
        #metrics.log_derived(
        #    "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
        #)

        counts = {}
        for lk in logging_outputs[0].keys():
            if lk.startswith("count_"):
                val = sum(log[lk] for log in logging_outputs)
                metrics.log_scalar(lk, val)
                counts[lk] = val

        for lk in logging_outputs[0].keys():
            if lk.startswith("loss_"):
                val = sum(log[lk] for log in logging_outputs)
                metrics.log_scalar(lk, val / sample_size / math.log(2), round=3)
            elif lk.startswith("correct_"):
                val = sum(log[lk] for log in logging_outputs)
                metrics.log_scalar(lk, val / counts[re.sub("correct", "count", lk)])
            elif lk.startswith("text_") and lk.endswith("_loss"):
                val = sum(log[lk] for log in logging_outputs)
                metrics.log_scalar(lk, val / math.log(2), round=3)
            elif lk.startswith("ema_"):
                val = sum(log[lk] for log in logging_outputs)
                metrics.log_scalar(lk, val , round=3)
            elif lk.endswith("_cls_loss"):
                val = sum(log[lk] for log in logging_outputs)
                metrics.log_scalar(lk, val, round=3)
            #elif lk.startswith("target_var_") or lk.startswith("pred_var_"):
                #logger.info(f"_var_ is running, lk: {lk}" )
            #    val = sum(log[lk] for log in logging_outputs)
            #    metrics.log_scalar(lk, val , round=3)


    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        raise NotImplementedError()

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False
