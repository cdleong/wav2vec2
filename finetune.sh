python run_finetuning.py \
--model_name_or_path="facebook/wav2vec2-large-xlsr-53" \
--dataset_config_name="rw" \
--output_dir=./wav2vec2-large-xlsr-kinyrwanda \
--preprocessing_num_workers="16" \
--overwrite_output_dir \
--num_train_epochs="5" \
--per_device_train_batch_size="50" \
--per_device_eval_batch_size="50" \
--learning_rate="2e-5" \
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
--gradient_checkpointing="1" \
--fp16 \
--do_train \
--do_eval \
--dataloader_num_workers="16" \
--group_by_length
