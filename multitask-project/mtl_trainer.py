from transformers import Trainer
from transformers.trainer_utils import EvalLoopOutput, EvalPrediction
from torch.utils.data.dataloader import DataLoader
from typing import Optional, List
from sklearn.metrics import f1_score
import numpy as np
import collections
import torch
from torchmetrics import F1, Accuracy, Metric
import time
import json
import pdb
# data_args = json.load(open("large_args", "r"))


class TaskwiseCrossEntropyLoss(Metric):
    def __init__(self, dist_sync_on_step=False):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, losses: torch.Tensor):
        # update metric states
        self.sum += torch.sum(losses)
        self.count += losses.shape[0]

    def compute(self):
        # compute final result
        return self.sum.float() / self.count

class MultitaskTrainer(Trainer):
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        
        model = self._wrap_model(self.model, training=False)
        batch_size = dataloader.batch_size

        model.eval()
        device = self.model.device
        # print(device)

        self.callback_handler.eval_dataloader = dataloader
        eval_dataset = dataloader.dataset

        f1s = []
        task_losses = []
        for i, task in enumerate(self.model.model_args['tasks']):
            f1s.append(F1(num_classes = self.model.model_args['task_label_map'][task], average='none').to('cuda'))
            task_losses.append(TaskwiseCrossEntropyLoss().to('cuda'))

        all_preds = [[] for i in range(len(f1s))]
        all_labels = [[] for i in range(len(f1s))]
        all_glue_labels = [[] for i in range(len(f1s))]
        all_qa_labels = [[] for i in range(len(f1s))]
        all_task_losses = [[] for i in range(len(f1s))]
        all_losses = []
        all_task_ids = []

        for steps, inputs in enumerate(dataloader):
            if "description_input_ids" in inputs.keys():
                outputs = model.forward(
                    inputs['input_ids'].to('cuda'),
                    attention_mask = inputs['attention_mask'].to('cuda'),
                    labels = inputs['labels'].to('cuda'),
                    task_ids = inputs['task_ids'].to('cuda'),
                    description_input_ids = inputs['description_input_ids'].to('cuda'),
                    description_attention_mask = inputs['description_attention_mask'].to('cuda'),
                    return_logits = True)
            elif "glue_labels" in inputs.keys():
                outputs = model.forward(
                    inputs['input_ids'].to('cuda'),
                    attention_mask = inputs['attention_mask'].to('cuda'),
                    labels = inputs['labels'].to('cuda'),
                    glue_labels = inputs['glue_labels'].to('cuda'),
                    start_positions = inputs['start_positions'].to('cuda'),
                    end_positions = inputs['end_positions'].to('cuda'),
                    task_ids = inputs['task_ids'].to('cuda'),
                    return_logits = True)
            else:
                outputs = model.forward(
                    inputs['input_ids'].to('cuda'),
                    attention_mask = inputs['attention_mask'].to('cuda'),
                    labels = inputs['labels'].to('cuda'),
                    task_ids = inputs['task_ids'].to('cuda'),
                    return_logits = True)

            all_losses.append(outputs['loss'].cpu().data)
            assert len(outputs['logits']) == len(outputs['labels']) and len(outputs['task_ids']) == len(outputs['labels'])
            # Logits and labels are list where each item in the list corresponds to the elements in the batch from one task
            # Each outputs['logits'][i] is a tensor size (# examples of that task) x (seq length of batch) x (# labels that are present in that task)
            # Each outputs['labels'][i] is a tensor of (# examples of that task) x (seq length of batch)

            for i in range(len(outputs['logits'])):
                temp = outputs['logits'][i].cpu().data # .numpy()

                # This argmax finds the prediction from the logits i.e. which label is correct in the last dimension
                preds = np.argmax(temp, axis=len(outputs['logits'][i].shape)-1)
                # print(preds.shape)
                #pdb.set_trace()
                if self.model.model_args['task_kind'][outputs['task_ids'][i]] == "seq":
                    for j in range(len(preds)):
                        all_preds[outputs['task_ids'][i]].append(preds[j])
                        all_labels[outputs['task_ids'][i]].append(outputs['labels'][i][j])
                        all_task_ids.append(outputs['task_ids'][i])
                elif self.model.model_args['task_kind'][outputs['task_ids'][i]] == "glue":
                    for j in range(len(preds)):
                        all_preds[outputs['task_ids'][i]].append(preds[j])
                        all_glue_labels[outputs['task_ids'][i]].append(outputs['glue_labels'][i][j])
                elif self.model.model_args['task_kind'][outputs['task_ids'][i]] == "qa":
                    qa_preds = np.argmax(temp, axis = 1)
                    # qa_preds is (batch size) x 2 where each is the predicted span start and end position
                    for j in range(len(qa_preds)):
                        pred_start, pred_end = qa_preds[j][0], qa_preds[j][1]
                        gold_start, gold_end = outputs['start_positions'][i][j][0].cpu().item(), outputs['end_positions'][i][j][0].cpu().item()
                        min_pos = min(pred_start, gold_start)
                        max_pos = max(pred_end, gold_end)
                        try:
                            pred = np.zeros(max_pos - min_pos + 1)
                            gold = np.zeros(max_pos - min_pos + 1)
                        except:
                            print(max_pos, min_pos, max_pos - min_pos + 1)
                            continue
                        for k in range(len(gold)):
                            if k + min_pos >= gold_start and k + min_pos <= gold_end:
                                gold[k] = 1
                                
                        for k in range(len(pred)):
                            if k + min_pos >= pred_start and k + min_pos <= pred_end:
                                pred[k] = 1

                        #pdb.set_trace()
                        all_preds[outputs['task_ids'][i]].append(pred)
                        all_qa_labels[outputs['task_ids'][i]].append(gold)
                    # Here we have the start and end gold positions
                    # And the start and end oredicted positions
                    # Compute F1 over those
                # pdb.set_trace()
                for j in range(outputs['logits'][i].shape[0]):
                    all_task_losses[outputs['task_ids'][i]].append(outputs['all_losses'][i].item())

        # all_Preds and all_logits are the same shape, num_examples_in_process x sequence_len  
        for i in range(len(f1s)):
            if self.model.model_args['task_kind'][i] == "seq":
                process_predictions = [
                    [p.item() for (p, l) in zip(prediction, label) if l != -100]
                    for prediction, label in zip(all_preds[i], all_labels[i])
                ]
                process_predictions = [item for sublist in process_predictions for item in sublist]
                # process_predictions = torch.tensor([item for sublist in process_predictions for item in sublist])

                process_labels = [
                    [l.item() for (p, l) in zip(prediction, label) if l != -100]
                    for prediction, label in zip(all_preds[i], all_labels[i])
                ]
                process_labels = [item for sublist in process_labels for item in sublist]

                if len(process_labels) > 0:
                    ip_preds = torch.tensor(process_predictions).to('cuda')
                    ip_labels = torch.tensor(process_labels).to('cuda')
                    # print(ip_preds, ip_labels)
                    # print(ip_preds.is_cuda, ip_labels.is_cuda)
                    f1s[i].update(ip_preds, ip_labels)
            elif self.model.model_args['task_kind'][i] == "glue":
                process_predictions = [prediction.item() for prediction in all_preds[i]]
                process_labels = [label.item() for label in all_glue_labels[i]]
                if len(process_labels) > 0:
                    ip_preds = torch.tensor(process_predictions).to('cuda')
                    ip_labels = torch.tensor(process_labels).to('cuda')
                    # print(ip_preds, ip_labels)
                    # print(ip_preds.is_cuda, ip_labels.is_cuda)
                    f1s[i].update(ip_preds, ip_labels)
            elif self.model.model_args['task_kind'][i] == "qa":
                qa_preds = [int(item) for sublist in all_preds[i] for item in sublist]
                qa_preds = torch.tensor(qa_preds).to('cuda')
                qa_labels = [int(item) for sublist in all_qa_labels[i] for item in sublist]
                qa_labels = torch.tensor(qa_labels).to('cuda')
                # pdb.set_trace()
                f1s[i].update(qa_preds, qa_labels)
            task_losses[i].update(torch.tensor(all_task_losses[i]).to('cuda'))

        metrics = {}

        f1_scores = []
        for i in range(len(f1s)):
            classwise_scores = f1s[i].compute()
            reshaped_scores = classwise_scores.reshape(classwise_scores.shape[0], -1)
            filtered_scores = reshaped_scores[~torch.any(reshaped_scores.isnan(),dim=1)]
            f1_scores.append(torch.mean(filtered_scores))

        # print("Process ", torch.distributed.get_rank(), f1_scores)
        # ch = input()
        for i in range(len(f1s)):
            metrics[self.model.model_args['tasks'][i]] = f1_scores[i].cpu().item()
            metrics[self.model.model_args['tasks'][i]+"-loss"] =  task_losses[i].compute().cpu().item() #np.mean(all_task_losses[i])

        metrics['eval_f1'] = np.mean([score.cpu().data for score in f1_scores]).item()

        metrics['eval loss'] = np.mean(all_losses).item()
        # print(metrics)
        # pdb.set_trace() 
        """

        # Uncomment this block if you also want to log train loss during pre-finetuning

        train_dataloader = self.get_train_dataloader()
        all_task_losses = [[] for i in range(len(f1s))]
        train_task_losses = []
        for i, task in enumerate(self.model.model_args['tasks']):
            train_task_losses.append(TaskwiseCrossEntropyLoss().to('cuda'))

        for steps, inputs in enumerate(train_dataloader):
            if steps == 2500:
                break
            if "description_input_ids" in inputs.keys():
                outputs = model.forward(
                    inputs['input_ids'].to('cuda'),
                    attention_mask = inputs['attention_mask'].to('cuda'),
                    labels = inputs['labels'].to('cuda'),
                    task_ids = inputs['task_ids'].to('cuda'),
                    description_input_ids = inputs['description_input_ids'].to('cuda'),
                    description_attention_mask = inputs['description_attention_mask'].to('cuda'),
                    return_logits = True)
            elif "glue_labels" in inputs.keys():
                outputs = model.forward(
                    inputs['input_ids'].to('cuda'),
                    attention_mask = inputs['attention_mask'].to('cuda'),
                    labels = inputs['labels'].to('cuda'),
                    glue_labels = inputs['glue_labels'].to('cuda'),
                    task_ids = inputs['task_ids'].to('cuda'),
                    return_logits = True)
            else:
                outputs = model.forward(
                    inputs['input_ids'].to('cuda'),
                    attention_mask = inputs['attention_mask'].to('cuda'),
                    labels = inputs['labels'].to('cuda'),
                    task_ids = inputs['task_ids'].to('cuda'),
                    return_logits = True)

            assert len(outputs['logits']) == len(outputs['labels']) and len(outputs['task_ids']) == len(outputs['labels'])
            # Logits and labels are list where each item in the list corresponds to the elements in the batch from one task
            # Each outputs['logits'][i] is a tensor size (# examples of that task) x (seq length of batch) x (# labels that are present in that task)
            # Each outputs['labels'][i] is a tensor of (# examples of that task) x (seq length of batch)

            for i in range(len(outputs['logits'])):
                for j in range(outputs['logits'][i].shape[0]):
                    all_task_losses[outputs['task_ids'][i]].append(outputs['all_losses'][i].item())

        for i in range(len(f1s)):
            train_task_losses[i].update(torch.tensor(all_task_losses[i]).to('cuda'))

        for i in range(len(f1s)):
            metrics[self.model.model_args['tasks'][i]+"-train-loss"] =  train_task_losses[i].compute().cpu().item() #np.mean(all_task_losses[i])

        """
        # pdb.set_trace()

        num_samples = len(all_preds) # self.numexamples(dataloader)
        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)
