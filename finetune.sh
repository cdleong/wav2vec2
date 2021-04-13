python run_finetuning.py \
--model_name_or_path="facebook/wav2vec2-large-xlsr-53" \
--dataset_config_name="rw" \
--output_dir=/workspace/output_models/rw/wav2vec2-large-xlsr-kinyarwanda/rw/wav2vec2-large-xlsr-kinyarwanda-fixed-vocab/ \
--overwrite_output_dir \
--num_train_epochs="20" \
--per_device_train_batch_size="4" \
--per_device_eval_batch_size="4" \
--learning_rate="2.34e-4" \
--evaluation_strategy="steps" \
--save_steps="1000" \
--eval_steps="1000" \
--logging_steps="1000" \
--save_total_limit="5" \
--freeze_feature_extractor \
--hidden_dropout="0.047" \
--activation_dropout="0.055" \
--attention_dropout="0.094" \
--feat_proj_dropout="0.04" \
--layerdrop="0.041" \
--mask_time_prob="0.082" \
--gradient_checkpointing="1" \
--fp16 \
--do_train \
--do_eval \
--group_by_length \
--report_to="wandb" \
--load_best_model_at_end=True \
--metric_for_best_model='wer' \
--greater_is_better=False \
--warmup_steps="500" \
--dataloader_num_workers="1" \
--max_val_samples="1024" \
# --warmup_ratio="0.1" \
# --preprocessing_num_workers="16" \
