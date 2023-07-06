python main.py \
--dataset_name "src/data_utils/wizard_of_wikipedia.py" \
--preproc_dir ./data/processed_wow_datasets \
--use_less_samples True \
--dataset_config_name posterior \
--learning_rate 3e-5 \
--num_train_epochs 7 \
--source_prefix "" \
--output_dir "save/path_to_model" \
--model_name_or_path "save/path_to_model" \
--max_source_length 512 \
--max_target_length 128 \
--preprocessing_num_workers 1 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--pad_to_max_length \
--weight_decay 0.005 \
--do_predict True \
--save_strategy "epoch" \
--metric_name "ppl&bleu&rouge&dist&f1&acc&recall" \
--report_to "tensorboard" \
--overwrite_cache False \
--gradient_accumulation_steps 16 \
--logging_steps 100 \
--eval_selection True \
--num_beams 1 \
--attend_latent True \
--use_feature kn \

# add the following command line for categorical prior
# --categorical_prior True \