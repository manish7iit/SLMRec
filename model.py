import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
import numpy as np
from peft import (
    LoraConfig,
    get_peft_model
)
import math
import os

DEFAULT_OPEN_BASE_MODEL = "Qwen/Qwen2.5-3B"

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

class Log2feats(torch.nn.Module):
    def __init__(self, user_emb_dim, item_emb_dim, seq_len):
        super(Log2feats, self).__init__()
        self.pos_emb = torch.nn.Embedding(seq_len, item_emb_dim) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=0.5)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(user_emb_dim, eps=1e-8)

        for _ in range(2):
            new_attn_layernorm = torch.nn.LayerNorm(user_emb_dim, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(user_emb_dim,
                                                            8,
                                                            0.5)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(user_emb_dim, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(user_emb_dim, 0.5)
            self.forward_layers.append(new_fwd_layer)

    def forward(self, log_seqs):
        seqs = log_seqs
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).cuda())
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(log_seqs.cpu() == 0).cuda()
        seqs *= ~timeline_mask # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device="cuda"))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats

class SASRec(nn.Module):
    def __init__(self, args, device, dataset):
        super(SASRec, self).__init__()
        self.device = device
        self.m_item = dataset.m_item
        self.dim = args.emb_dim
        self.hid_dim = args.hid_dim
        self.embedding = nn.Embedding(self.m_item, self.dim)
        self.up_emb = nn.Linear(self.dim,self.hid_dim)
        self.down_emb = nn.Linear(self.hid_dim,self.dim)

        self.pos_embedding = nn.Embedding(args.max_seq_length, self.hid_dim)
        self.emb_dropout = nn.Dropout(p=0.5)
        self.dropout = 0.5
        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        self.last_layernorm = nn.LayerNorm(self.hid_dim, eps=1e-8)

        for _ in range(args.layers):
            new_attn_layernorm = nn.LayerNorm(self.hid_dim, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = nn.MultiheadAttention(self.hid_dim, 1, self.dropout, batch_first=True)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = nn.LayerNorm(self.hid_dim, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = nn.Sequential(nn.Conv1d(self.hid_dim, self.hid_dim, kernel_size=1),
                                          nn.Dropout(p=self.dropout),
                                          nn.ReLU(),
                                          nn.Conv1d(self.hid_dim, self.hid_dim, kernel_size=1),
                                          nn.Dropout(p=self.dropout))
            self.forward_layers.append(new_fwd_layer)

        self.loss = torch.nn.CrossEntropyLoss()
        self.act = nn.Sigmoid()

    def log2feats(self, log_seq):
        seqs = log_seq
        seqs *= self.dim ** 0.5
        positions = np.tile(np.array(range(log_seq.shape[1])), [log_seq.shape[0], 1])
        seqs += self.pos_embedding(torch.LongTensor(positions).to(self.device))
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(log_seq.cpu() == 0).to(self.device)
        seqs *= ~timeline_mask
        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.device))

        for i in range(len(self.attention_layers)):
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, attn_mask=attention_mask)
            seqs = Q + mha_outputs

            seqs = self.forward_layernorms[i](seqs)
            residuals = self.forward_layers[i](seqs.transpose(-1, -2)).transpose(-1, -2)
            seqs = seqs + residuals
            seqs *= ~timeline_mask

        log_feats = self.last_layernorm(seqs)

        return log_feats
    
    def predict_sample(self, seq, pos, neg):
        seq = self.up_emb(self.embedding(seq))
        log_feats = self.log2feats(seq)
        seq_output = self.down_emb(log_feats[:, -1, :]).unsqueeze(1)
        pos_embs = self.embedding(pos)
        neg_embs = self.embedding(neg)
        pos_logits = torch.matmul(seq_output, pos_embs.permute(0, 2, 1))#(log_feats * pos_embs).sum(dim=-1)
        neg_logits = torch.matmul(seq_output, neg_embs.permute(0, 2, 1))#(log_feats * neg_embs).sum(dim=-1)
        # print(pos_logits.shape,neg_logits.shape)
        return pos_logits, neg_logits
    
    def predict_all(self,seq):
        seq = self.up_emb(self.embedding(seq))
        log_feats = self.log2feats(seq)
        seq_output = self.down_emb(log_feats[:, -1, :]) # [b,dim,1]
        test_item_emb = self.embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        loss = self.loss(logits, pos.squeeze(-1))

        return logits,loss #[bs,item_size]

    def forward(self, seq, pos):
    # ce loss 
        seq = self.up_emb(self.embedding(seq))
        log_feats = self.log2feats(seq)
        seq_output = self.down_emb(log_feats[:, -1, :]) # [b,dim]
        test_item_emb = self.embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        loss = self.loss(logits, pos.squeeze(-1))
        return loss

class LLM4Rec(nn.Module):
    def __init__(self, **args):
        super(LLM4Rec, self).__init__()
        self._keys_to_ignore_on_save = []
        self.args = args
        self.input_dim, self.output_dim = args['input_dim'], args['output_dim']
        self.base_model = self.args.get('base_model') or DEFAULT_OPEN_BASE_MODEL

        print(f'Initializing language decoder from {self.base_model} ...')
        # add the lora module
        peft_config = LoraConfig(
            task_type='FEATURE_EXTRACTION',
            r=self.args['lora_r'],
            lora_alpha=self.args['lora_alpha'],
            lora_dropout=self.args['lora_dropout'],
            target_modules=self.args['lora_target_modules'],
            bias='none',
        )

        self.llama_model = AutoModel.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16,
            cache_dir=args['cache_dir'],
            device_map=self.args['device_map']
        )
        if self.args['drop_type'] == "trune":
            decoder_layers = self._get_decoder_layers(self.llama_model)
            self._set_decoder_layers(
                self.llama_model,
                nn.ModuleList(decoder_layers[:self.args['llama_decoder_nums']])
            )
        elif self.args['drop_type'] == "interval":
            # Interval for layer dropping
            interval_nums = self.args['interval_nums']

            # Keep layers with interval-based dropping
            decoder_layers = self._get_decoder_layers(self.llama_model)
            kept_layers = nn.ModuleList([layer for i, layer in enumerate(decoder_layers) if (i + 1) % (interval_nums + 1) != 0])
            self._set_decoder_layers(self.llama_model, kept_layers)
            num_layers = len(kept_layers)
            print(f'Number of layers in the model: {num_layers}')

        # Remove prepare_model_for_int8_training call
        if self.args['train_stargy'] == "lora":
            self.llama_model = get_peft_model(self.llama_model, peft_config)
            self.llama_model.print_trainable_parameters()
        self.llama_model.config.use_cache = False
        # self.llama_model.config.num_hidden_layers = 10
        self.llama_tokenizer = AutoTokenizer.from_pretrained(self.base_model, use_fast=True, cache_dir=args['cache_dir'])
        if self.llama_tokenizer.pad_token is None:
            self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        self.llama_tokenizer.padding_side = "right"
        self.instruct_ids, self.instruct_mask = self.llama_tokenizer(self.args['instruction_text'][0],
                                                                     truncation=True, padding=False,
                                                                     return_tensors='pt', add_special_tokens=False).values()
        self.response_ids, self.response_mask = self.llama_tokenizer(self.args['instruction_text'][1],
                                                                     truncation=True, padding=False,
                                                                     return_tensors='pt', add_special_tokens=False).values()
        print('Language decoder initialized.')

        self.task_type = args['task_type']
        self.input_embeds = nn.Embedding.from_pretrained(self.args['input_embeds'], freeze=True) # official true - loss 0
        self.input_proj = nn.Linear(self.input_dim, self.llama_model.config.hidden_size)
        self.score = nn.Linear(self.llama_model.config.hidden_size, self.input_dim, bias=False)
        # self.score_up = nn.Linear(self.input_dim, self.llama_model.config.hidden_size, bias=False)
        self.loss = torch.nn.CrossEntropyLoss()

    def _get_nested_attr(self, obj, path):
        for name in path.split("."):
            obj = getattr(obj, name)
        return obj

    def _set_nested_attr(self, obj, path, value):
        names = path.split(".")
        for name in names[:-1]:
            obj = getattr(obj, name)
        setattr(obj, names[-1], value)

    def _get_decoder_layers(self, model):
        for path in ("layers", "model.layers", "decoder.layers", "gpt_neox.layers", "transformer.h"):
            try:
                layers = self._get_nested_attr(model, path)
                self._decoder_layers_path = path
                return layers
            except AttributeError:
                continue
        raise AttributeError(f"Could not find decoder layers for {self.base_model}.")

    def _set_decoder_layers(self, model, layers):
        self._set_nested_attr(model, self._decoder_layers_path, layers)

    def _embed_tokens(self, token_ids):
        if hasattr(self.llama_model, "get_input_embeddings"):
            embeddings = self.llama_model.get_input_embeddings()
            if embeddings is not None:
                return embeddings(token_ids)
        if hasattr(self.llama_model, "model") and hasattr(self.llama_model.model, "get_input_embeddings"):
            embeddings = self.llama_model.model.get_input_embeddings()
            if embeddings is not None:
                return embeddings(token_ids)
        raise AttributeError(f"Could not find token embeddings for {self.base_model}.")

    def predict(self, inputs, inputs_mask, output_hidden_states=False, output_logits=True):
        bs = inputs.shape[0]
        instruct_embeds = self._embed_tokens(self.instruct_ids.cuda()).expand(bs, -1, -1)
        response_embeds = self._embed_tokens(self.response_ids.cuda()).expand(bs, -1, -1)
        instruct_mask = self.instruct_mask.cuda().expand(bs, -1)
        response_mask = self.response_mask.cuda().expand(bs, -1)

        if self.task_type == 'general':
            users = self.user_proj(self.user_embeds(inputs[:, 0].unsqueeze(1)))
            items = self.input_proj(self.input_embeds(inputs[:, 1:]))
            inputs = torch.cat([users, items], dim=1)
        else:
            inputs = self.input_proj(self.input_embeds(inputs))
        # print(instruct_embeds.shape,inputs.shape,response_embeds.shape)
        inputs = torch.cat([instruct_embeds, inputs, response_embeds], dim=1)
        attention_mask = torch.cat([instruct_mask, inputs_mask, response_mask], dim=1)
        assert attention_mask.size()[0] == inputs.size()[0] and attention_mask.size()[1] == inputs.size()[1]

        outputs = self.llama_model(inputs_embeds=inputs, attention_mask=attention_mask, return_dict=True, output_hidden_states=output_hidden_states)
        # print("outputs len:{}".format(len(outputs.hidden_states))) # 33
        if output_logits:
            pooled_output = self._mean_pool(outputs.last_hidden_state, attention_mask)
            pooled_logits = self.score(pooled_output)
        if not output_hidden_states:
            return pooled_logits
        else:
            if output_logits:
                return pooled_logits,outputs.hidden_states
            else:
                return outputs.hidden_states

    def _mean_pool(self, hidden_states, attention_mask):
        mask = attention_mask.to(hidden_states.device).unsqueeze(-1).type_as(hidden_states)
        return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    def _all_item_logits(self, pooled_logits):
        pooled_logits = F.normalize(pooled_logits, dim=-1)
        item_emb = F.normalize(self.input_embeds.weight, dim=-1)
        return torch.matmul(pooled_logits, item_emb.transpose(0, 1))

    def multiple_predict(self, inputs, inputs_mask):
        bs = inputs.shape[0]
        instruct_embeds = self._embed_tokens(self.instruct_ids.cuda()).expand(bs, -1, -1)
        response_embeds = self._embed_tokens(self.response_ids.cuda()).expand(bs, -1, -1)
        instruct_mask = self.instruct_mask.cuda().expand(bs, -1)
        response_mask = self.response_mask.cuda().expand(bs, -1)

        if self.task_type == 'general':
            users = self.user_proj(self.user_embeds(inputs[:, 0].unsqueeze(1)))
            items = self.input_proj(self.input_embeds(inputs[:, 1:]))
            inputs = torch.cat([users, items], dim=1)
        else:
            inputs = self.input_proj(self.input_embeds(inputs))
        inputs = torch.cat([instruct_embeds, inputs, response_embeds], dim=1)
        attention_mask = torch.cat([instruct_mask, inputs_mask, response_mask], dim=1)
        assert attention_mask.size()[0] == inputs.size()[0] and attention_mask.size()[1] == inputs.size()[1]

        outputs = self.llama_model(inputs_embeds=inputs, attention_mask=attention_mask, return_dict=True, output_hidden_states=True)
        outputs_all = outputs.hidden_states
        pooled_logits = list()
        for output_tmp in outputs_all: # downsample
            pooled_tmp = self.score(self._mean_pool(output_tmp, attention_mask))
            pooled_logits.append(pooled_tmp.unsqueeze(1))
        return pooled_logits

    def forward(self, input_ids, labels, inputs, inputs_mask, answers, neg_samples, data_type):
        # pooled_logits = self.multiple_predict(inputs, inputs_mask)#self.predict(inputs, inputs_mask) #[b,emb_dim]

        loss = None
        if torch.max(data_type).item() in {0, 1}: # all item ce loss / full-ranking eval
            pooled_logits = self.predict(inputs, inputs_mask)
            logits = self._all_item_logits(pooled_logits)
            loss = self.loss(logits, answers.squeeze(-1))
            predict = logits
        else:
            raise ValueError(f"Unsupported data_type for LLM4Rec: {torch.max(data_type).item()}")
        return { 'loss':loss,
            'logits': predict,
        }

class LLM4RecTeacher(LLM4Rec):
    def forward(self, input_ids, labels, inputs, inputs_mask, answers, neg_samples, data_type):
        loss = None
        if torch.max(data_type).item() in {0, 1}: # all item ce loss / full-ranking eval
            pooled_logits, teacher_output_states = self.predict(inputs, inputs_mask, output_hidden_states=True, output_logits=True)
            logits = self._all_item_logits(pooled_logits)
            loss = self.loss(logits, answers.squeeze(-1))
            predict = logits
            return { 'loss':loss,
                'logits': predict,
                'teacher_output_states':teacher_output_states,
            }
        else:
            raise ValueError(f"Unsupported data_type for LLM4RecTeacher: {torch.max(data_type).item()}")
        return { 'loss':loss,
            'logits': predict,
        }

class LLM4RecStudent(LLM4Rec):
    def __init__(self, **args):
        super().__init__(**args)
        self.distill_block = args['distill_block']
        self.is_cls_multiple = args['is_cls_multiple']
        self.down_layer_list = nn.ModuleList()
        if self.is_cls_multiple:
            for _ in range(self.distill_block-1):
                self.down_layer_list.append(nn.Linear(self.llama_model.config.hidden_size, self.input_dim, bias=False))


    def forward(self, input_ids, labels, inputs, inputs_mask, answers, neg_samples, data_type):
        loss = None
        if torch.max(data_type).item() in {0, 1}: # all item ce loss / full-ranking eval
            pooled_logits,student_output_states = self.predict(inputs, inputs_mask,output_hidden_states=True,output_logits=True)
            logits = self._all_item_logits(pooled_logits)
            loss = self.loss(logits, answers.squeeze(-1))
            predict = logits
            loss_cls_multiple = 0
            if self.is_cls_multiple:
                test_item_emb = F.normalize(self.input_embeds.weight, dim=-1)
                for i in range(0,self.distill_block-1): 
                    pooled_logits_tmp = self.down_layer_list[i](student_output_states[(len(student_output_states)//self.distill_block)*(i+1)].mean(dim=1)) 
                    pooled_logits_tmp = F.normalize(pooled_logits_tmp, dim=-1)
                    logits_tmp = torch.matmul(pooled_logits_tmp, test_item_emb.transpose(0, 1))
                    loss_tmp = self.loss(logits_tmp, answers.squeeze(-1))
                    loss_cls_multiple = loss_cls_multiple + loss_tmp
            return { 'loss':loss,
                'logits': predict,
                'student_output_states':student_output_states,
                'data_type':data_type,
                'loss_cls_multiple':loss_cls_multiple,
            }
        else:
            raise ValueError(f"Unsupported data_type for LLM4RecStudent: {torch.max(data_type).item()}")

class LLM4RecDistill(nn.Module):
    def __init__(self, **args):
        super(LLM4RecDistill, self).__init__()
        self._keys_to_ignore_on_save = []
        self.args = args
        self.model_teacher = LLM4Rec(
            base_model=self.args['base_model'],
            task_type=self.args['task_type'],
            cache_dir=self.args['cache_dir'],
            input_dim=128,
            output_dim=0,
            interval_nums=self.args['interval_nums'],
            drop_type=self.args['drop_type'],
            lora_r=self.args['lora_r'],
            lora_alpha=self.args['lora_alpha'],
            lora_dropout=self.args['lora_dropout'],
            lora_target_modules=self.args['lora_target_modules'],
            device_map=self.args['device_map'],
            instruction_text=self.args['instruction_text'],
            train_stargy = self.args['train_stargy'],
            user_embeds=None,
            input_embeds=self.args['item_embed'],
            seq_len=30,
            llama_decoder_nums=self.args['llama_decoder_nums_teacher'],
        )
        self.model_student = LLM4Rec(
            base_model=self.args['base_model'],
            task_type=self.args['task_type'],
            cache_dir=self.args['cache_dir'],
            input_dim=128,
            output_dim=0,
            interval_nums=self.args['interval_nums'],
            drop_type=self.args['drop_type'],
            lora_r=self.args['lora_r'],
            lora_alpha=self.args['lora_alpha'],
            lora_dropout=self.args['lora_dropout'],
            lora_target_modules=self.args['lora_target_modules'],
            device_map=self.args['device_map'],
            instruction_text=self.args['instruction_text'],
            train_stargy = self.args['train_stargy'],
            user_embeds=None,
            input_embeds=self.args['item_embed'],
            seq_len=30,
            llama_decoder_nums=self.args['llama_decoder_nums_student'],
        )
        self.teacher_block = self.args['llama_decoder_nums_teacher']//4
        self.student_block = self.args['llama_decoder_nums_student']//4
        self.distill_lambda = self.args['distill_lambda']
        # self.ratio = self.args['llama_decoder_nums_teacher']//self.args['llama_decoder_nums_student']
        # Total number of blocks for knowledge distillation
        self.loss_cls = torch.nn.CrossEntropyLoss()
        self.down_layer_teacher_list = nn.ModuleList()
        self.down_layer_student_list = nn.ModuleList()
        self.distill_block = self.args['distill_block']
        if self.args['is_cls_multiple_teacher']:
            for _ in range(self.distill_block-1):
                self.down_layer_teacher_list.append(nn.Linear(self.model_teacher.llama_model.config.hidden_size, self.model_teacher.input_dim, bias=False))
        if self.args['is_cls_multiple_student']:
            for _ in range(self.distill_block-1):
                self.down_layer_student_list.append(nn.Linear(self.model_teacher.llama_model.config.hidden_size, self.model_teacher.input_dim, bias=False))

    def predict(self, inputs, inputs_mask):
        pooled_logits_teacher,teacher_output_states = self.model_teacher.predict(inputs, inputs_mask,output_hidden_states=True,output_logits=True)
        pooled_logits_student,student_hidden_states = self.model_student.predict(inputs, inputs_mask,output_hidden_states=True,output_logits=True)
        return teacher_output_states,student_hidden_states,pooled_logits_teacher,pooled_logits_student

    def predict_student(self, inputs, inputs_mask):
        pooled_logits = self.model_student.predict(inputs, inputs_mask,output_hidden_states=False,output_logits=True)
        return pooled_logits

    def forward(self, input_ids, labels, inputs, inputs_mask, answers, neg_samples, data_type):
        loss = None
        if torch.max(data_type).item() ==0: # all item ce loss 
            # self.model_student._set_static_graph(True)
            teacher_output_states,student_hidden_states,pooled_logits_teacher,pooled_logits_student = self.predict(inputs, inputs_mask)
            test_item_emb = F.normalize(self.model_student.input_embeds.weight, dim=-1)
            logits_teacher = torch.matmul(F.normalize(pooled_logits_teacher, dim=-1), test_item_emb.transpose(0, 1))
            logits_student = torch.matmul(F.normalize(pooled_logits_student, dim=-1), test_item_emb.transpose(0, 1))
            loss = self.loss_cls(logits_student, answers.squeeze(-1))
            predict = logits_student
            loss_cls_multiple_teacher, loss_cls_multiple_student = 0, 0
            logits_teacher_concat, logits_student_concat = list(), list()
            if self.args['is_cls_multiple_teacher']:
                for i in range(1,self.distill_block):  
                    pooled_logits_tmp = self.down_layer_teacher_list[i-1](teacher_output_states[(len(teacher_output_states)//self.distill_block)*i].mean(dim=1)) 
                    pooled_logits_tmp = F.normalize(pooled_logits_tmp, dim=-1)
                    logits_tmp = torch.matmul(pooled_logits_tmp, test_item_emb.transpose(0, 1))
                    loss_tmp = self.loss_cls(logits_tmp, answers.squeeze(-1))
                    loss_cls_multiple_teacher = loss_cls_multiple_teacher + loss_tmp
                    logits_teacher_concat.append(logits_tmp)
                logits_teacher_concat.append(logits_teacher)
                # logits_teacher_concat = torch.stack(logits_teacher_concat)
            else:
                logits_teacher_concat.append(logits_teacher)
            if self.args['is_cls_multiple_student']:
                for i in range(1,self.distill_block): 
                    pooled_logits_tmp = self.down_layer_student_list[i-1](student_hidden_states[(len(student_hidden_states)//self.distill_block)*i].mean(dim=1)) 
                    pooled_logits_tmp = F.normalize(pooled_logits_tmp, dim=-1)
                    logits_tmp = torch.matmul(pooled_logits_tmp, test_item_emb.transpose(0, 1))
                    loss_tmp = self.loss_cls(logits_tmp, answers.squeeze(-1))
                    loss_cls_multiple_student = loss_cls_multiple_student + loss_tmp
                    logits_student_concat.append(logits_tmp)
                logits_student_concat.append(logits_student)
                # logits_student_concat = torch.stack(logits_student_concat)
            else:
                logits_student_concat.append(logits_student)
            return { 'loss':loss,
                'logits': predict,
                'teacher_output_states':teacher_output_states,
                'student_output_states':student_hidden_states,
                'loss_cls_multiple_teacher':loss_cls_multiple_teacher,
                'loss_cls_multiple_student':loss_cls_multiple_student,
                'logits_teacher':logits_teacher_concat,
                'logits_student':logits_student_concat,
                'data_type':data_type,
            }
        elif torch.max(data_type).item() ==1: # full-ranking eval
            pooled_logits = self.predict_student(inputs, inputs_mask)
            test_item_emb = F.normalize(self.model_student.input_embeds.weight, dim=-1)
            predict = torch.matmul(F.normalize(pooled_logits, dim=-1), test_item_emb.transpose(0, 1))
            loss = self.loss_cls(predict, answers.squeeze(-1))

            return { 'loss':loss,
                'logits': predict,
            }
        else:
            raise ValueError(f"Unsupported data_type for LLM4RecDistill: {torch.max(data_type).item()}")
