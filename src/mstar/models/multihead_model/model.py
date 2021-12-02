from transformers import BertPreTrainedModel, XLMRobertaConfig, XLMRobertaModel
import torch.nn as nn
import torch
import pdb

class MultiheadModel(BertPreTrainedModel):
    def __init__(
            self, config, mtl_args
            ):

        super().__init__(config)
        self.num_tasks = len(mtl_args["tasks"])
        self.model_args = mtl_args

        self.roberta = XLMRobertaModel(config, add_pooling_layer=False)
        self.heads = nn.ModuleList()
        for i, task in enumerate(mtl_args["tasks"]):
            if mtl_args["task_kind"][i] == "seq":
                self.heads.append(TokenClassificationDecoder(config.hidden_size, task, mtl_args["task_label_map"]))
            elif mtl_args["task_kind"][i] == "glue":
                self.heads.append(GlueDecoder(config.hidden_size, task, mtl_args["task_label_map"]))
            elif mtl_args["task_kind"][i] == "qa":
                self.heads.append(QADecoder(config.hidden_size, task, mtl_args["task_label_map"]))
        self.init_weights()
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        task_ids=None,
        glue_labels=None,
        return_logits = False,
        start_positions = None,
        end_positions = None
    ):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        # Find out which task ids are present in this batch
        unique_task_ids = torch.unique(task_ids)
        unique_task_ids_list = (
            unique_task_ids.cpu().numpy()
            if unique_task_ids.is_cuda
            else unique_task_ids.numpy()
        )
        all_logits = []
        all_losses = []
        all_labels = []
        all_glue_labels = []
        all_start_positions = []
        all_end_positions = []
        all_task_ids = []
        for unique_task_id in unique_task_ids_list:

            # Create the filter for all inputs in this batch which have that particular task ID
            current_task_filter = task_ids == unique_task_id
            decoder_id = unique_task_id

            # Access that particular decoder and run a forward pass
            # print(unique_task_id)
            # pdb.set_trace()

            if type(self.heads[decoder_id]) == TokenClassificationDecoder:
                logits, loss = self.heads[decoder_id](
                        sequence_output[current_task_filter],
                        attention_mask=attention_mask[current_task_filter],
                        labels=None if labels is None else labels[current_task_filter],
                    )
            elif type(self.heads[decoder_id]) == GlueDecoder:
                logits, loss = self.heads[decoder_id](
                        sequence_output[current_task_filter],
                        attention_mask=attention_mask[current_task_filter],
                        labels=None if labels is None else glue_labels[current_task_filter],
                    )
            elif type(self.heads[decoder_id]) == QADecoder:
                assert start_positions is not None # , print(qa_spans)
                # pdb.set_trace()
                logits, loss = self.heads[decoder_id](
                        sequence_output[current_task_filter],
                        attention_mask=attention_mask[current_task_filter],
                        start_positions = start_positions[current_task_filter], end_positions = end_positions[current_task_filter]
                        # labels=None if labels is None else glue_labels[current_task_filter],
                    )

            # Logits are of shape (# examples in batch of said task) x (sequence length) x (# labels in that decoder)
            # So the third dimension does not match for each decoder. How do we handle this?
            all_labels.append(labels[current_task_filter])
            all_glue_labels.append(glue_labels[current_task_filter])
            all_start_positions.append(start_positions[current_task_filter])
            all_end_positions.append(end_positions[current_task_filter])
            all_logits.append(logits)
            all_losses.append(loss)
            all_task_ids.append(unique_task_id)
        
        loss = torch.stack(all_losses)

        # For each batch we return one loss value and two lists - the logits and corresponding labels
        # The number of items in both lists are the same, which is the number of different tasks present in the batch

        return {'loss':loss.mean(), 'logits':all_logits, 'labels':all_labels, 'task_ids':all_task_ids, 'glue_labels':all_glue_labels, 'all_losses':all_losses, 'start_positions':all_start_positions, 'end_positions':all_end_positions} 


class TokenClassificationDecoder(nn.Module):
    def __init__(self, hidden_size, task_name, task_label_map):
        super().__init__()
        self.num_labels = task_label_map[task_name]
        self.dropout = nn.Dropout(0.1)
        self.model = nn.Linear(hidden_size, self.num_labels)

    def forward(self, sequence_output, attention_mask, labels=None, **kwargs):
        loss = None
        sequence_output = self.dropout(sequence_output)
        logits = self.model(sequence_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                active_labels = active_labels.long()
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # Here loss is returned which is a mean across all tokens in a sequence across all sequences in the batch. Here we'd potentially need some form of loss scaling. 
        # For now just scaling by the number of labels but this need not be the best way to do it
        return logits, loss#/self.num_labels


class GlueDecoder(torch.nn.Module):
    def __init__(self, hidden_size, task_name, task_label_map):
        super().__init__()
        self.num_labels = task_label_map[task_name]
        self.dropout = nn.Dropout(0.1)
        self.model = nn.Linear(hidden_size, self.num_labels)

    def forward(self, sequence_output, attention_mask, labels=None, **kwargs):
        loss = None
        pooled_output = sequence_output[:, 0, :] # Output of the <s> token
        pooled_output = self.dropout(pooled_output)
        logits = self.model(pooled_output)

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.float().view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.long().view(-1))

        return logits, loss


class QADecoder(nn.Module):
    def __init__(self, hidden_size, task_name, task_label_map):
        super().__init__()
        self.num_labels = task_label_map[task_name] # config.num_labels
        self.qa_outputs = nn.Linear(hidden_size, self.num_labels)

    def forward(self, sequence_output, attention_mask, start_positions=None, end_positions = None, **kwargs):
        loss = None
        logits = self.qa_outputs(sequence_output)
        # Logits is (batch size) x (seq length) x (num_labels = 2)
        start_logits, end_logits = logits.split(1, dim=-1)
        # Each is (batch size) x (seq length) x (1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        # Each is (batch size) x (seq length)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            # In the loss: (batch size) x (seq length) vs (batch size)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        return logits, total_loss
