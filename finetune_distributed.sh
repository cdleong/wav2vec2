# https://discuss.huggingface.co/t/weights-biases-supporting-wave2vec2-finetuning/4839/2
python -m torch.distributed.launch \
--nproc_per_node 4 run_finetuning.py \
--model_name_or_path="facebook/wav2vec2-large-xlsr-53" \
--dataset_config_name="rw" \
--output_dir=/workspace/output_models/rw/wav2vec2-large-xlsr-kinyarwanda \
--preprocessing_num_workers="52" \
--overwrite_output_dir \
--num_train_epochs="5" \
--per_device_train_batch_size="2" \
--per_device_eval_batch_size="2" \
--learning_rate="2e-4" \
--warmup_steps="500" \
--evaluation_strategy="steps" \
--save_steps="10000" \
--eval_steps="10000" \
--logging_steps="1000" \
--save_total_limit="3" \
--freeze_feature_extractor \
--activation_dropout="0.055" \
--attention_dropout="0.094" \
--feat_proj_dropout="0.04" \
--layerdrop="0.04" \
--mask_time_prob="0.08" \
--gradient_checkpointing="0" \
--fp16 \
--do_train \
--do_eval \
--dataloader_num_workers="52" \
--group_by_length \
--report_to="wandb" \
--run_name = 'rw-distributed-baseline', \
--load_best_model_at_end = True, \
--metric_for_best_model='wer', \
--greater_is_better=False, \
