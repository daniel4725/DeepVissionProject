# -------------------------------------- 6.6.22 --------------------------------------
# baseline mixed models with less data (half the data to even the amount for the gender baseline)
# * no L2 here
python3.6 train_models.py --experiment_name "baseline mixed 0.5data 1" --GPU 1 --model "AgePredModel" --data_type "mixed" --partial_data 0.5 --epochs 50 --wandb_callback True
python3.6 train_models.py --experiment_name "baseline mixed 0.5data 2" --GPU 1 --model "AgePredModel" --data_type "mixed" --partial_data 0.5 --epochs 50 --wandb_callback True

# adding L2 to baseline gender models (L2 of the ADAM optimization)
python3.6 train_models.py --experiment_name "baseline_L2-0.1 M 1" --GPU 2 --model "AgePredModel" --data_type M --L2 0.1 --epochs 50 --wandb_callback True

# adding L2 to baseline gender models "M" (0.05 and 0.1)
python3.6 train_models.py --experiment_name "baseline_my_L2-0.1 M 1" --GPU 2 --model "AgePredModel" --data_type M --L2 0.1 --epochs 50 --wandb_callback True
python3.6 train_models.py --experiment_name "baseline_my_L2-0.1 M 2" --GPU 2 --model "AgePredModel" --data_type M --L2 0.1 --epochs 50 --wandb_callback True
python3.6 train_models.py --experiment_name "baseline_my_L2-0.05 M 1" --GPU 0 --model "AgePredModel" --data_type M --L2 0.05 --epochs 50 --wandb_callback True
python3.6 train_models.py --experiment_name "baseline_my_L2-0.05 M 2" --GPU 1 --model "AgePredModel" --data_type M --L2 0.05 --epochs 50 --wandb_callback True

# * with L2 here: (my L2)
python3.6 train_models.py --experiment_name "baseline_L2-0.1 mixed 0.5data 1" --GPU 0 --model "AgePredModel" --data_type "mixed" --partial_data 0.5 --L2 0.1 --epochs 50 --wandb_callback True
python3.6 train_models.py --experiment_name "baseline_L2-0.1 mixed 0.5data 2" --GPU 2 --model "AgePredModel" --data_type "mixed" --partial_data 0.5 --L2 0.1 --epochs 50 --wandb_callback True


# -------------------------------------- 7.6.22 --------------------------------------
# adding L2 to baseline gender models "F" (0.05 and 0.1)
python3.6 train_models.py --experiment_name "baseline_my_L2-0.1 F 1" --GPU 0 --model "AgePredModel" --data_type "F" --L2 0.1 --epochs 52 --wandb_callback "True"
python3.6 train_models.py --experiment_name "baseline_my_L2-0.1 F 2" --GPU 0 --model "AgePredModel" --data_type "F" --L2 0.1 --epochs 52 --wandb_callback "True"
python3.6 train_models.py --experiment_name "baseline_my_L2-0.05 F 1" --GPU 1 --model "AgePredModel" --data_type "F" --L2 0.05 --epochs 52 --wandb_callback "True"
python3.6 train_models.py --experiment_name "baseline_my_L2-0.05 F 2" --GPU 1 --model "AgePredModel" --data_type "F" --L2 0.05 --epochs 52 --wandb_callback "True"

# -------------------------------------- 9.6.22 --------------------------------------
# baseline 0.5data with small L2
python3.6 train_models.py --experiment_name "baseline mixed 0.5data L2-0.001 1" --GPU 0 --model "AgePredModel" --data_type "mixed" --partial_data "0.5" --L2 "0.001" --epochs 60 --wandb_callback "True"
python3.6 train_models.py --experiment_name "baseline mixed 0.5data L2-0.005 1" --GPU 1 --model "AgePredModel" --data_type "mixed" --partial_data "0.5" --L2 "0.005" --epochs 60 --wandb_callback "True"
python3.6 train_models.py --experiment_name "baseline mixed 0.5data L2-0.01 1" --GPU 2 --model "AgePredModel" --data_type "mixed" --partial_data "0.5" --L2 "0.01" --epochs 60 --wandb_callback "True"

# baseline 0.5data with dropouts
python3.6 train_models.py --experiment_name "baseline mixed 0.5data drop-0.5 1" --GPU 0 --model "AgePredModel" --data_type "mixed" --partial_data "0.5" --dropout "0.5" --epochs 60 --wandb_callback "True"
python3.6 train_models.py --experiment_name "baseline mixed 0.5data drop-0.5 2" --GPU 0 --model "AgePredModel" --data_type "mixed" --partial_data "0.5" --dropout "0.5" --epochs 60 --wandb_callback "True"
python3.6 train_models.py --experiment_name "baseline mixed 0.5data drop-0.4 1" --GPU 1 --model "AgePredModel" --data_type "mixed" --partial_data "0.5" --dropout "0.4" --epochs 60 --wandb_callback "True"
python3.6 train_models.py --experiment_name "baseline mixed 0.5data drop-0.4 2" --GPU 1 --model "AgePredModel" --data_type "mixed" --partial_data "0.5" --dropout "0.4" --epochs 60 --wandb_callback "True"
python3.6 train_models.py --experiment_name "baseline mixed 0.5data drop-0.3 1" --GPU 2 --model "AgePredModel" --data_type "mixed" --partial_data "0.5" --dropout "0.3" --epochs 60 --wandb_callback "True"
python3.6 train_models.py --experiment_name "baseline mixed 0.5data drop-0.3 2" --GPU 2 --model "AgePredModel" --data_type "mixed" --partial_data "0.5" --dropout "0.3" --epochs 60 --wandb_callback "True"

# baseline 0.5data with 32 batch_size
python3.6 train_models.py --experiment_name "baseline mixed 0.5data 32batch 1" --GPU 3 --model "AgePredModel" --data_type "mixed" --partial_data "0.5" --batch_size "32" --epochs 50 --wandb_callback "True"
python3.6 train_models.py --experiment_name "baseline mixed 0.5data 32batch 2" --GPU 3 --model "AgePredModel" --data_type "mixed" --partial_data "0.5" --batch_size "32" --epochs 50 --wandb_callback "True"

# -------------------------------------- 11.6.22 --------------------------------------
# baseline 0.5data with std-mean and min-max normalization
python3.6 train_models.py --experiment_name "baseline mixed 0.5data std-mean norm 1" --GPU 0 --model "AgePredModel" --data_type "mixed" --partial_data "0.5" --epochs 50 --transform "normalize_stdmean" --wandb_callback "True"
python3.6 train_models.py --experiment_name "baseline mixed 0.5data std-mean norm 2" --GPU 0 --model "AgePredModel" --data_type "mixed" --partial_data "0.5" --epochs 50 --transform "normalize_stdmean" --wandb_callback "True"
python3.6 train_models.py --experiment_name "baseline mixed 0.5data minmax norm 1" --GPU 1 --model "AgePredModel" --data_type "mixed" --partial_data "0.5" --epochs 50 --transform "normalize_minmax" --wandb_callback "True"
python3.6 train_models.py --experiment_name "baseline mixed 0.5data minmax norm 2" --GPU 1 --model "AgePredModel" --data_type "mixed" --partial_data "0.5" --epochs 50 --transform "normalize_minmax" --wandb_callback "True"

# -------------------------------------- 12.6.22 --------------------------------------
# baseline 0.5data with std-mean and min-max normalization
python3.6 train_models.py --experiment_name "baseline mixed 0.5data std-mean norm 1" --GPU 1 --model "AgePredModel" --data_type "mixed" --partial_data "0.5" --epochs 60 --transform "normalize_stdmean" --wandb_callback "True"
python3.6 train_models.py --experiment_name "baseline mixed 0.5data std-mean norm 2" --GPU 1 --model "AgePredModel" --data_type "mixed" --partial_data "0.5" --epochs 60 --transform "normalize_stdmean" --wandb_callback "True"
python3.6 train_models.py --experiment_name "baseline mixed 0.5data minmax norm 1" --GPU 1 --model "AgePredModel" --data_type "mixed" --partial_data "0.5" --epochs 60 --transform "normalize_minmax" --wandb_callback "True"
python3.6 train_models.py --experiment_name "baseline mixed 0.5data minmax norm 2" --GPU 0 --model "AgePredModel" --data_type "mixed" --partial_data "0.5" --epochs 60 --transform "normalize_minmax" --wandb_callback "True"

# baseline 0.5 data
python3.6 train_models.py --experiment_name "baseline 0.5mixed 1" --GPU 0 --model "AgePredModel" --data_type "mixed" --partial_data "0.5" --epochs 60 --wandb_callback "True"
python3.6 train_models.py --experiment_name "baseline 0.5mixed 2" --GPU 0 --model "AgePredModel" --data_type "mixed" --partial_data "0.5" --epochs 60 --wandb_callback "True"
python3.6 train_models.py --experiment_name "baseline 0.5mixed 3" --GPU 0 --model "AgePredModel" --data_type "mixed" --partial_data "0.5" --epochs 60 --wandb_callback "True"

# baseline 0.5 data with Adam's L2
python3.6 train_models.py --experiment_name "baseline 0.5mixed A_L2-1% 1" --GPU 2 --model "AgePredModel" --data_type "mixed" --partial_data "0.5" --L2 "0.01" --epochs 60 --wandb_callback "True"
python3.6 train_models.py --experiment_name "baseline 0.5mixed A_L2-1% 2" --GPU 2 --model "AgePredModel" --data_type "mixed" --partial_data "0.5" --L2 "0.01" --epochs 60 --wandb_callback "True"
python3.6 train_models.py --experiment_name "baseline 0.5mixed A_L2-5% 1" --GPU 3 --model "AgePredModel" --data_type "mixed" --partial_data "0.5" --L2 "0.05" --epochs 60 --wandb_callback "True"
python3.6 train_models.py --experiment_name "baseline 0.5mixed A_L2-5% 2" --GPU 3 --model "AgePredModel" --data_type "mixed" --partial_data "0.5" --L2 "0.05" --epochs 60 --wandb_callback "True"
python3.6 train_models.py --experiment_name "baseline 0.5mixed A_L2-10% 1" --GPU 3 --model "AgePredModel" --data_type "mixed" --partial_data "0.5" --L2 "0.1" --epochs 60 --wandb_callback "True"
python3.6 train_models.py --experiment_name "baseline 0.5mixed A_L2-10% 2" --GPU 3 --model "AgePredModel" --data_type "mixed" --partial_data "0.5" --L2 "0.1" --epochs 60 --wandb_callback "True"
python3.6 train_models.py --experiment_name "baseline 0.5mixed A_L2-0.5% 1" --GPU 2 --model "AgePredModel" --data_type "mixed" --partial_data "0.5" --L2 "0.005" --epochs 60 --wandb_callback "True"
python3.6 train_models.py --experiment_name "baseline 0.5mixed A_L2-0.5% 2" --GPU 2 --model "AgePredModel" --data_type "mixed" --partial_data "0.5" --L2 "0.005" --epochs 60 --wandb_callback "True"

# -------------------------------------- 12.6.22 --------------------------------------
# The final baselines
# mixed:
python3.6 train_models.py --experiment_name "a baseline 1" --GPU 0 --model "AgePredModel" --data_type "mixed" --partial_data "0.5" --L2 "0.05" --epochs 60 --transform "normalize_stdmean" --wandb_callback "True"
python3.6 train_models.py --experiment_name "a baseline 2" --GPU 0 --model "AgePredModel" --data_type "mixed" --partial_data "0.5" --L2 "0.05" --epochs 60 --transform "normalize_stdmean" --wandb_callback "True"
python3.6 train_models.py --experiment_name "a baseline 3" --GPU 0 --model "AgePredModel" --data_type "mixed" --partial_data "0.5" --L2 "0.05" --epochs 60 --transform "normalize_stdmean" --wandb_callback "True"

# males:
python3.6 train_models.py --experiment_name "a baseline F 1" --GPU 1 --model "AgePredModel" --data_type "F" --L2 "0.05" --epochs 60 --transform "normalize_stdmean" --wandb_callback "True"
python3.6 train_models.py --experiment_name "a baseline F 2" --GPU 1 --model "AgePredModel" --data_type "F" --L2 "0.05" --epochs 60 --transform "normalize_stdmean" --wandb_callback "True"
python3.6 train_models.py --experiment_name "a baseline F 3" --GPU 1 --model "AgePredModel" --data_type "F" --L2 "0.05" --epochs 60 --transform "normalize_stdmean" --wandb_callback "True"

# females:
python3.6 train_models.py --experiment_name "a baseline M 1" --GPU 2 --model "AgePredModel" --data_type "M" --L2 "0.05" --epochs 60 --transform "normalize_stdmean" --wandb_callback "True"
python3.6 train_models.py --experiment_name "a baseline M 2" --GPU 2 --model "AgePredModel" --data_type "M" --L2 "0.05" --epochs 60 --transform "normalize_stdmean" --wandb_callback "True"
python3.6 train_models.py --experiment_name "a baseline M 3" --GPU 2 --model "AgePredModel" --data_type "M" --L2 "0.05" --epochs 60 --transform "normalize_stdmean" --wandb_callback "True"
