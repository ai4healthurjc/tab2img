#python3 src/main.py --categorical_encoding=count --dataset=hepatitis --type_dataset=binary
#python3 src/main.py --categorical_encoding=count --dataset=steno --type_dataset=binary
#
#python3 src/tabular.py --categorical_encoding=count --dataset=dermat --classifier=dt
#python3 src/tabular.py --categorical_encoding=count --dataset=dermat --classifier=tabpfn
#python3 src/tabular.py --categorical_encoding=count --dataset=dermat --classifier=knn
#python3 src/tabular.py --categorical_encoding=count --dataset=dermat --classifier=svm
#python3 src/tabular.py --categorical_encoding=count --dataset=dermat --classifier=lasso
#
#python3 src/noise_generator.py --categorical_encoding=count --dataset=dermat --n_new_vars=3 --noise_type=homogeneous
#python3 src/noise_generator.py --categorical_encoding=count --dataset=dermat --n_new_vars=3 --noise_type=heterogeneous
##
#python3 src/tab2img.py --noise_type=homogeneous --dataset=dermat
#python3 src/tab2img.py --noise_type=heterogeneous --dataset=dermat
#python3 src/tab2img.py --noise_type=preprocessed --dataset=dermat

#python3 src/cnn_pytorch.py --noise_type=homogeneous --n_cpus=50 --n_gpus=1 --dataset=dermat
python3 src/cnn_pytorch.py --noise_type=heterogeneous --n_cpus=50 --n_gpus=1 --dataset=dermat
python3 src/cnn_pytorch.py --noise_type=preprocessed --n_cpus=50 --n_gpus=1 --dataset=dermat

python3 src/gradcam_torch.py --noise_type=heterogeneous --n_jobs=50  --dataset=hepatitis
python3 src/gradcam_torch.py --noise_type=homogeneous --n_jobs=50 --dataset=hepatitis