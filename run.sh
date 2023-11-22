testdata=dataset/pde_heat/test_102400_pde_heat.prefix

# base run
torchrun --standalone --nnodes 1 --nproc_per_node 2 train.py --exp_id base_pde_heat_512_5 --data_loss_weight 5.0 --max_epoch 80 &&
CUDA_VISIBLE_DEVICES=0 python train.py --use_wandb True --eval_only --exp_name operator_eval --exp_id base_pde_heat_512_5 --eval_data $testdata --eval_size 102400 --eval_from_exp checkpoint/jingmins/dumped/pde_exp/base_pde_heat_512_5  --batch_size_eval 512 --eval_amp -1 &&

#torchrun --standalone --nnodes 1 --nproc_per_node 2 train.py --exp_id base_pde_heat_512_4 --data_loss_weight 4.0 --max_epoch 40 &&
#CUDA_VISIBLE_DEVICES=0 python train.py --use_wandb True --eval_only --exp_name operator_eval --exp_id base_pde_heat_512_4 --eval_data $testdata --eval_size 102400 --eval_from_exp checkpoint/jingmins/dumped/pde_exp/base_pde_heat_512_4  --batch_size_eval 512 --eval_amp -1 &&

#torchrun --standalone --nnodes 1 --nproc_per_node 2 train.py --exp_id base_pde_heat_512_3 --data_loss_weight 3.0 --max_epoch 30 &&
#CUDA_VISIBLE_DEVICES=0 python train.py --use_wandb True --eval_only --exp_name operator_eval --exp_id base_pde_heat_512_3 --eval_data $testdata --eval_size 102400 --eval_from_exp checkpoint/jingmins/dumped/pde_exp/base_pde_heat_512_3  --batch_size_eval 512 --eval_amp -1 &&


#torchrun --standalone --nnodes 1 --nproc_per_node 2 train.py --exp_id base_pde_heat_512_2 --data_loss_weight 2.0 --max_epoch 30 &&
#CUDA_VISIBLE_DEVICES=0 python train.py --use_wandb True --eval_only --exp_name operator_eval --exp_id base_pde_heat_512_2 --eval_data $testdata --eval_size 102400 --eval_from_exp checkpoint/jingmins/dumped/pde_exp/base_pde_heat_512_2  --batch_size_eval 512 --eval_amp -1 &&


## no text
#CUDA_VISIBLE_DEVICES=0 python train.py --no_text --exp_id no_text_t6e_5 --batch_size 512 --batch_size_eval 1024 --max_epoch 80 &&
#CUDA_VISIBLE_DEVICES=0 python train.py --use_wandb False --eval_only --exp_name operator_eval --exp_id no_text_t6e_10 --eval_data $testdata --eval_size 102400 --eval_from_exp checkpoint/yuxuan/dumped/ode_exp/no_text_t6e_5 --batch_size_eval 1024 --eval_amp -1 --no_text &&
#
## data only
#torchrun --standalone --nnodes 1 --nproc_per_node 2 train.py --exp_id no_text_loss_t6e --max_epoch 80 --data_only &&
#CUDA_VISIBLE_DEVICES=0 python train.py --use_wandb False --eval_only --exp_name operator_eval --exp_id no_text_loss_t6e --eval_data $testdata --eval_size 102400 --eval_from_exp checkpoint/yuxuan/dumped/ode_exp/no_text_loss_t6e --batch_size_eval 512 --eval_amp -1 --data_only &&
#
## text only
#torchrun --standalone --nnodes 1 --nproc_per_node 2 train.py --exp_id no_data_loss_t6e --max_epoch 80 --text_only &&
#CUDA_VISIBLE_DEVICES=0 python train.py --use_wandb False --eval_only --exp_name operator_eval --exp_id no_data_loss_t6e --eval_data $testdata --eval_size 102400 --eval_from_exp checkpoint/yuxuan/dumped/ode_exp/no_data_loss_t6e --batch_size_eval 512 --eval_amp -1 --text_only &&
#
## dimension mismatch
#datapathh=functions,../dataset/ode_5d_6e/train_512000_ode_5d_6e.prefix,../dataset/ode_5d_6/val_25600_ode_5d_6e.prefix,
#testdataa=../dataset/ode_5d_6e/test_102400_ode_5d_6e.prefix
#torchrun --standalone --nnodes 1 --nproc_per_node 2 train.py --exp_id dim_mismatch_t6e --max_output_dimension 5 --reload_data $datapathh --batch_size 192;
#CUDA_VISIBLE_DEVICES=0 python train.py --use_wandb False --eval_only --exp_name operator_eval --exp_id dim_mismatch_t6e --eval_data $testdataa --eval_size 102400 --eval_from_exp checkpoint/yuxuan/dumped/ode_exp/dim_mismatch_t6e --batch_size_eval 512 --eval_amp -1 --max_output_dimension 5 &&
#
## noiseless text
#torchrun --standalone --nnodes 1 --nproc_per_node 2 train.py --exp_id noiseless_text_t6e --max_epoch 80 --noisy_text_input False &&
#CUDA_VISIBLE_DEVICES=0 python train.py --use_wandb False --eval_only --exp_name operator_eval --exp_id noiseless_text_t6e --eval_data $testdata --eval_size 102400 --eval_from_exp checkpoint/yuxuan/dumped/ode_exp/noiseless_text_t6e --batch_size_eval 512 --eval_amp -1 --noisy_text_input False &&
#
## clean text
#torchrun --standalone --nnodes 1 --nproc_per_node 2 train.py --exp_id clean_text_t6e --max_epoch 80 --use_skeleton False &&
#CUDA_VISIBLE_DEVICES=0 python train.py --use_wandb False --eval_only --exp_name operator_eval --exp_id clean_text_t6e --eval_data $testdata --eval_size 102400 --eval_from_exp checkpoint/yuxuan/dumped/ode_exp/clean_text_t6e --batch_size_eval 512 --eval_amp -1 --use_skeleton False &&
#
## compare gap (64)
#torchrun --standalone --nnodes 1 --nproc_per_node 2 train.py --exp_id base_input_64 --train_noise_gamma 0 --eval_noise_gamma 0 --max_epoch 60 &&
#CUDA_VISIBLE_DEVICES=0 python train.py --use_wandb False --eval_only --exp_name operator_eval --exp_id base_input_64 --train_noise_gamma 0 --eval_noise_gamma 0 --eval_data $testdata --eval_size 102400 --eval_from_exp checkpoint/yuxuan/dumped/ode_exp/base_input_64 --batch_size_eval 512 --eval_amp -1 &&
#CUDA_VISIBLE_DEVICES=0 python train.py --no_text --exp_id no_text_input_64 --batch_size 512 --batch_size_eval 1024 --train_noise_gamma 0 --eval_noise_gamma 0 --max_epoch 60 &&
#CUDA_VISIBLE_DEVICES=0 python train.py --no_text --use_wandb False --eval_only --exp_name operator_eval --exp_id no_text_input_64 --train_noise_gamma 0 --eval_noise_gamma 0 --eval_data $testdata --eval_size 102400 --eval_from_exp checkpoint/yuxuan/dumped/ode_exp/no_text_input_64 --batch_size_eval 512 --eval_amp -1 &&
#
## compare gap (16)
#torchrun --standalone --nnodes 1 --nproc_per_node 2 train.py --exp_id base_input_16 --train_noise_gamma 0 --eval_noise_gamma 0 --input_step 4 --max_epoch 40 &&
#CUDA_VISIBLE_DEVICES=0 python train.py --use_wandb False --eval_only --exp_name operator_eval --exp_id base_input_16 --train_noise_gamma 0 --eval_noise_gamma 0 --input_step 4 --eval_data $testdata --eval_size 102400 --eval_from_exp checkpoint/yuxuan/dumped/ode_exp/base_input_16 --batch_size_eval 512 --eval_amp -1 &&
#CUDA_VISIBLE_DEVICES=0 python train.py --no_text --exp_id no_text_input_16 --batch_size 512 --batch_size_eval 1024 --train_noise_gamma 0 --eval_noise_gamma 0 --input_step 4 --max_epoch 40 &&
#CUDA_VISIBLE_DEVICES=0 python train.py --no_text --use_wandb False --eval_only --exp_name operator_eval --exp_id no_text_input_16 --train_noise_gamma 0 --eval_noise_gamma 0 --input_step 4 --eval_data $testdata --eval_size 102400 --eval_from_exp checkpoint/yuxuan/dumped/ode_exp/no_text_input_16 --batch_size_eval 512 --eval_amp -1 &&
#
## compare gap (32)
#torchrun --standalone --nnodes 1 --nproc_per_node 2 train.py --exp_id base_input_32 --train_noise_gamma 0 --eval_noise_gamma 0 --input_step 2 --max_epoch 40 &&
#CUDA_VISIBLE_DEVICES=0 python train.py --use_wandb False --eval_only --exp_name operator_eval --exp_id base_input_32 --train_noise_gamma 0 --eval_noise_gamma 0 --input_step 2 --eval_data $testdata --eval_size 102400 --eval_from_exp checkpoint/yuxuan/dumped/ode_exp/base_input_32 --batch_size_eval 512 --eval_amp -1 &&
#CUDA_VISIBLE_DEVICES=0 python train.py --no_text --exp_id no_text_input_32 --batch_size 512 --batch_size_eval 1024 --train_noise_gamma 0 --eval_noise_gamma 0 --input_step 2 --max_epoch 40 &&
#CUDA_VISIBLE_DEVICES=0 python train.py --no_text --use_wandb False --eval_only --exp_name operator_eval --exp_id no_text_input_32 --train_noise_gamma 0 --eval_noise_gamma 0 --input_step 2 --eval_data $testdata --eval_size 102400 --eval_from_exp checkpoint/yuxuan/dumped/ode_exp/no_text_input_32 --batch_size_eval 512 --eval_amp -1 &&
#
## ood evaluation
#CUDA_VISIBLE_DEVICES=0 python train.py --use_wandb False --eval_only --exp_name operator_eval --exp_id base_t6e_ood_15 --eval_data $testood --eval_size 102400 --eval_from_exp checkpoint/yuxuan/dumped/ode_exp/base_t6e_10 --batch_size_eval 512 --eval_amp -1 &&
#CUDA_VISIBLE_DEVICES=0 python train.py --use_wandb False --eval_only --exp_name operator_eval --exp_id base_t6e_ood_20 --eval_data $testoodd --eval_size 102400 --eval_from_exp checkpoint/yuxuan/dumped/ode_exp/base_t6e_10 --batch_size_eval 512 --eval_amp -1 &&

echo "Done"
