python3 src/main.py --categorical_encoding=count --dataset=fram --type_dataset=binary
python3 src/main.py --categorical_encoding=count --dataset=steno --type_dataset=binary

python3 src/tabular.py --categorical_encoding=count --dataset=steno --classifier=dt
python3 src/tabular.py --categorical_encoding=count --dataset=steno --classifier=tabpfn
python3 src/tabular.py --categorical_encoding=count --dataset=steno --classifier=knn
python3 src/tabular.py --categorical_encoding=count --dataset=steno --classifier=svm
python3 src/tabular.py --categorical_encoding=count --dataset=steno --classifier=lasso
python3 src/tabular.py --categorical_encoding=count --dataset=fram --classifier=dt
python3 src/tabular.py --categorical_encoding=count --dataset=fram --classifier=tabpfn
python3 src/tabular.py --categorical_encoding=count --dataset=fram --classifier=knn
python3 src/tabular.py --categorical_encoding=count --dataset=fram --classifier=svm
python3 src/tabular.py --categorical_encoding=count --dataset=fram --classifier=lasso

python3 src/noise_generator.py --categorical_encoding=count --dataset=fram --n_new_vars=3 --dataset2=steno --noise_type=homogeneous
python3 src/noise_generator.py --categorical_encoding=count --dataset=steno --n_new_vars=3 --noise_type=homogeneous
python3 src/noise_generator.py --categorical_encoding=count --dataset=fram --n_new_vars=3 --dataset2=steno --noise_type=heterogeneous
python3 src/noise_generator.py --categorical_encoding=count --dataset=steno --n_new_vars=3 --noise_type=heterogeneous

python3 src/tab2img.py --noise_type=homogeneous --dataset=fram
python3 src/tab2img.py --noise_type=homogeneous --dataset=steno_second
python3 src/tab2img.py --noise_type=heterogeneous --dataset=fram
python3 src/tab2img.py --noise_type=heterogeneous --dataset=steno_second
python3 src/tab2img.py --noise_type=heterogeneous --dataset=steno
python3 src/tab2img.py --noise_type=homogeneous --dataset=steno

python3 src/cnn_pytorch.py --noise_type=homogeneous --n_cpus=50 --n_gpus=1 --dataset=fram
python3 src/cnn_pytorch.py --noise_type=heterogeneous --n_cpus=50 --n_gpus=1 --dataset=fram
python3 src/cnn_pytorch.py --noise_type=homogeneous --n_cpus=50 --n_gpus=1 --dataset=steno
python3 src/cnn_pytorch.py --noise_type=heterogeneous --n_cpus=50 --n_gpus=1 --dataset=steno

python3 src/fine_tuning.py --noise_type=homogeneous --n_cpus=50 --n_gpus=1 --parent_dataset=fram --child_dataset=steno_second
python3 src/fine_tuning.py --noise_type=heterogeneous --n_cpus=50 --n_gpus=1 --parent_dataset=fram --child_dataset=steno_second

python3 src/gradcam_torch.py --noise_type=heterogeneous --n_jobs=50  --dataset=fram
python3 src/gradcam_torch.py --noise_type=homogeneous --n_jobs=50 --dataset=fram