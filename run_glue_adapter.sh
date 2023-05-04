export TASK_NAME=cola # ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"] 

python run_glue_adapter.py \
	--model_name_or_path bert-base-uncased \
	--task_name $TASK_NAME \
	--do_train \
	--do_eval \
	--max_seq_length 128 \
	--per_device_train_batch_size 32 \
	--learning_rate 2e-5 \
	--num_train_epochs 20 \
	--save_total_limit 5 \
    --metric_for_best_model eval_loss \
    --use_mps_device True\
	--overwrite_output_dir True\
	--output_dir ../result/$TASK_NAME/