import transformers
import os
from typing import Any, Dict, List, Optional, Union
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers.trainer import *

class DistillationTrainingArguments(transformers.TrainingArguments):
    def __init__(self, *args, distill_lambda=0.001, llama_decoder_nums_student=8, llama_decoder_nums_teacher=32, distill_block=4, distill_type="other", distill_leave_layers=0, distill_type_standard="offline",
                is_cls_multiple=False,
                cls_multiple_lambda=1.0,
                kd_loss_type="cosine",
                distill_temperature=2.0,
                is_cls_multiple_teacher=False,
                is_cls_multiple_student=False,
                cls_multiple_lambda_teacher=1.0,
                cls_multiple_lambda_student=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.distill_lambda = distill_lambda
        self.llama_decoder_nums_student = llama_decoder_nums_student
        self.llama_decoder_nums_teacher = llama_decoder_nums_teacher
        self.distill_block = distill_block
        self.distill_type = distill_type
        self.distill_leave_layers = distill_leave_layers
        self.distill_type_standard = distill_type_standard
        self.is_cls_multiple=is_cls_multiple
        self.cls_multiple_lambda=cls_multiple_lambda
        self.kd_loss_type=kd_loss_type
        self.distill_temperature=distill_temperature
        self.is_cls_multiple_teacher=is_cls_multiple_teacher
        self.is_cls_multiple_student=is_cls_multiple_student
        self.cls_multiple_lambda_teacher=cls_multiple_lambda_teacher
        self.cls_multiple_lambda_student=cls_multiple_lambda_student

class SLMTrainer(transformers.Trainer):

    def log(self, logs: Dict[str, float],*args,**kwargs) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        # save output
        with open(os.path.join(self.args.output_dir,"log.txt"), 'a') as file:
            # print("logger output:{}".format(output))
            json.dump(output, file)
            file.write('\n')

        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)
        return super().log(logs, *args, **kwargs)

class DistillationTrainer(transformers.Trainer):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        # place teacher on same device as student
        if teacher_model is not None:
            self._move_model_to_device(self.teacher, self.model.llama_model.device)
            self.teacher.eval()
        # self.logger = logger

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):

        # compute student output
        outputs_student = model(**inputs)
        student_loss = outputs_student.loss
        # compute teacher output
        with torch.no_grad():
            outputs_teacher = self.teacher(**inputs)

        assert outputs_student.logits.size() == outputs_teacher.logits.size()

        temperature = getattr(self.args, "distill_temperature", 2.0)
        student_log_probs = F.log_softmax(outputs_student.logits.float() / temperature, dim=-1)
        teacher_log_probs = F.log_softmax(outputs_teacher.logits.detach().float() / temperature, dim=-1)
        student_probs = student_log_probs.exp()
        loss_logits = torch.sum(student_probs * (student_log_probs - teacher_log_probs), dim=-1).mean() * (temperature ** 2)
        loss = student_loss + getattr(self.args, "distill_lambda", 1.0) * loss_logits
        return (loss, outputs_student) if return_outputs else loss


class RecDistillationTrainer(DistillationTrainer,SLMTrainer):
    def _kl_student_teacher(self, logits_student, logits_teacher):
        temperature = self.args.distill_temperature
        student_log_probs = F.log_softmax(logits_student.float() / temperature, dim=-1)
        teacher_log_probs = F.log_softmax(logits_teacher.detach().float() / temperature, dim=-1)
        student_probs = student_log_probs.exp()
        return torch.sum(student_probs * (student_log_probs - teacher_log_probs), dim=-1).mean() * (temperature ** 2)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if self.args.distill_type_standard == "offline":
            # compute student output
            if self.label_smoother is not None and "labels" in inputs:
                labels = inputs.pop("labels")
            else:
                labels = None
            outputs_student = model(**inputs)
            student_loss = outputs_student['loss']
            if torch.max(outputs_student['data_type']).item() == 0:
                with torch.no_grad():
                    outputs_teacher = self.teacher(**inputs)
                loss_distill = self._kl_student_teacher(outputs_student['logits'], outputs_teacher['logits'])
                self.log({"loss_distill_kl": loss_distill.item()})
                if self.args.is_cls_multiple and outputs_student['loss_cls_multiple'] is not None:
                    loss_multiple = outputs_student['loss_cls_multiple'] * self.args.cls_multiple_lambda
                    student_loss = student_loss + loss_multiple
                    self.log({"loss_multiple": loss_multiple.item()})
                loss = student_loss + self.args.distill_lambda * loss_distill
            else:
                loss = student_loss
        elif self.args.distill_type_standard=="online":
            # compute student output
            if self.label_smoother is not None and "labels" in inputs:
                labels = inputs.pop("labels")
            else:
                labels = None
            outputs_student = model(**inputs)
            student_loss = outputs_student['loss']
            if torch.max(inputs['data_type']).item() == 0:
                loss_distill = self._kl_student_teacher(outputs_student['logits_student'][-1], outputs_student['logits_teacher'][-1])
                self.log({"loss_distill_kl": loss_distill.item()})

                if self.args.is_cls_multiple_teacher and outputs_student['loss_cls_multiple_teacher'] is not None:
                    loss_multiple_teacher = outputs_student['loss_cls_multiple_teacher'] * self.args.cls_multiple_lambda_teacher
                    student_loss = student_loss + loss_multiple_teacher
                    self.log({"loss_multiple_teacher": loss_multiple_teacher.item()})

                if self.args.is_cls_multiple_student and outputs_student['loss_cls_multiple_student'] is not None:
                    loss_multiple_student = outputs_student['loss_cls_multiple_student'] * self.args.cls_multiple_lambda_student
                    student_loss = student_loss + loss_multiple_student
                    self.log({"loss_multiple_student": loss_multiple_student.item()})
                
                loss = student_loss + self.args.distill_lambda * loss_distill
            else:
                loss = student_loss
        return (loss, outputs_student) if return_outputs else loss
