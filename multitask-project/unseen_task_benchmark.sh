for seed in 1 #2 3 4 5
do
	for task in wikigold #wnut17 mit-movie-en ncbi-diseease 
	do
		python3 finetune_model_sl.py --target-task-args ./data/metadata-with-kind/$task.json --num-epochs 40 > results/$seed/$task

		# python3 finetune_model_sl.py --target-task-args ./data/metadata-with-kind/$task.json --num-epochs 40 --use-encoder-weights ./model_dir --prefinetuning-args arguments_file  > results/$seed/$task
	done
done


for seed in 1 #2 3 4 5
do
	for task in cb #boolq copa wic
	do
		python3 finetune_model_superglue.py --target-task-args ./data/metadata-with-kind/$task.json --num-epochs 40 > results/$seed/$task

		# python3 finetune_model_superglue.py --target-task-args ./data/metadata-with-kind/$task.json --num-epochs 40 --use-encoder-weights ./model_dir --prefinetuning-args arguments_file  > results/$seed/$task
	done
done

for seed in 1 #2 3 4 5
do
	for task in squad_adv #qed xquad_r subjqa
	do
		python3 finetune_model_qa.py --target-task-args ./data/metadata-with-kind/$task.json --num-epochs 40 > results/$seed/$task

		# python3 finetune_model_superglue.py --target-task-args ./data/metadata-with-kind/$task.json --num-epochs 40 --use-encoder-weights ./model_dir --prefinetuning-args arguments_file  > results/$seed/$task
	done
done
