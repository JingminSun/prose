GPU=1

# heat
CUDA_VISIBLE_DEVICES=$GPU python3 train.py --export_data --exp_name pde_heat --exp_id gen_val_data --max_epoch 1 --n_steps_per_epoch 100 --log_periodic 10 --ICs_per_equation 4 --t_range 6.0 --t_num 192 --ode_param_range_gamma 0.1 &&
cp checkpoint/jingmins/dumped/pde_heat/gen_val_data/data.prefix ~/prose/dataset/pde_heat/val_25600_pde_heat.prefix &&
CUDA_VISIBLE_DEVICES=$GPU python3 train.py --export_data --exp_name pde_heat --exp_id gen_train_data --max_epoch 1 --n_steps_per_epoch 2000 --log_periodic 50 --ICs_per_equation 20 --t_range 6.0 --t_num 192 --ode_param_range_gamma 0.1 &&
cp checkpoint/jingmins/dumped/pde_heat/gen_train_data/data.prefix ~/prose/dataset/pde_heat/train_512000_pde_heat.prefix &&
CUDA_VISIBLE_DEVICES=$GPU python3 train.py --export_data --exp_name pde_heat --exp_id gen_test_data --max_epoch 1 --n_steps_per_epoch 400 --log_periodic 50 --ICs_per_equation 4 --t_range 6.0 --t_num 192 --ode_param_range_gamma 0.1 &&
cp checkpoint/jingmins/dumped/pde_heat/gen_test_data/data.prefix ~/prose/dataset/pde_heat/test_102400_pde_heat.prefix &&


echo "Done"
