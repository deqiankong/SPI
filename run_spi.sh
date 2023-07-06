# uniform prior
python main.py \
--dataset_name "src/data_utils/wizard_of_wikipedia.py" \
--preproc_dir ./data/processed_wow_datasets \
--use_less_samples True \
--dataset_config_name posterior \
--learning_rate 3e-5 \
--num_train_epochs 15 \
--source_prefix "" \
--output_dir "save/langevin_s_5_ss_01_bart-base_gp_1_kn_e15_top5_pad" \
--model_name_or_path "facebook/bart-base" \
--max_source_length 512 \
--max_target_length 128 \
--preprocessing_num_workers 1 \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 1 \
--pad_to_max_length \
--weight_decay 0.005 \
--do_train True \
--do_eval True \
--save_strategy "epoch" \
--metric_name "ppl&acc&recall" \
--report_to "tensorboard" \
--overwrite_cache False \
--gradient_accumulation_steps 16 \
--logging_steps 100 \
--merge_eval \
--eval_selection True \
--with_tracking True \
--max_patience 3 \
--attend_latent True \
--cls_ratio 1.0 \
--g_l_steps 1 \
--g_l_step_size 0.1 \
--use_feature kn \
--add_z_mse True \
--vae_kl_weights 1.0 \
--top_k_kn 5 \
--pad_knowledge True \
--pseudo_confidence 1 \

# Categorical prior
python main.py \
--dataset_name "src/data_utils/wizard_of_wikipedia.py" \
--preproc_dir ./data/processed_wow_datasets \
--use_less_samples True \
--dataset_config_name posterior \
--learning_rate 3e-5 \
--num_train_epochs 15 \
--source_prefix "" \
--output_dir "save/langevin_s_5_ss_01_bart-base_gp_1_kn_e15_top-all_pad_cat" \
--model_name_or_path "facebook/bart-base" \
--max_source_length 512 \
--max_target_length 128 \
--preprocessing_num_workers 1 \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 1 \
--pad_to_max_length \
--weight_decay 0.005 \
--do_train True \
--do_eval True \
--save_strategy "epoch" \
--metric_name "ppl&acc&recall" \
--report_to "tensorboard" \
--overwrite_cache False \
--gradient_accumulation_steps 16 \
--logging_steps 100 \
--merge_eval \
--eval_selection True \
--with_tracking True \
--max_patience 3 \
--attend_latent True \
--cls_ratio 1.0 \
--g_l_steps 1 \
--g_l_step_size 0.1 \
--use_feature kn \
--add_z_mse True \
--vae_kl_weights 1.0 \
--top_k_kn 1000 \
--pad_knowledge True \
--pseudo_confidence 1 \
--categorical_prior True \