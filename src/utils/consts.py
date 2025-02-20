from pathlib import Path

SEEDS = [24,12,87,50]

LIST_CONVENTIONAL_ESTIMATORS = ['dt', 'knn', 'lasso', 'rf']
LIST_DYNAMIC_ENSEMBLE_SELECTION_ESTIMATORS = ['knora-u', 'knora-e', 'des-p', 'ola', 'mcb', 'knop', 'meta-des']
LIST_DYNAMIC_CLASSIFIER_SELECTION_ESTIMATORS = ['rank', 'ola', 'lca', 'mla']

PATH_PROJECT_DIR = Path(__file__).resolve().parents[2]
PATH_PROJECT_DATA_RAW = Path.joinpath(PATH_PROJECT_DIR, 'data', 'raw')
PATH_PROJECT_REPORTS_TAB_TO_IMAGE = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image')
PATH_PROJECT_FIGURES = Path.joinpath(PATH_PROJECT_DIR, 'reports','figures')
PATH_PROJECT_REPORTS_NOISY_DATASETS = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'noisy_dataset')
PATH_PROJECT_REPORTS_NOISY_FIGURES = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'noisy_figures')

PATH_PROJECT_DATA_CRX_DIR = Path.joinpath(PATH_PROJECT_DIR, 'data', 'raw', 'bbdd_binary', 'crx.data.csv')
PATH_PROJECT_DATA_DIABETES_DIR = Path.joinpath(PATH_PROJECT_DIR, 'data', 'raw', 'bbdd_binary', 'bbdd_diabetes.csv')
PATH_PROJECT_DATA_GERMAN_DIR = Path.joinpath(PATH_PROJECT_DIR, 'data', 'raw', 'bbdd_binary', 'german.csv')
PATH_PROJECT_DATA_HEPATITIS_DIR = Path.joinpath(PATH_PROJECT_DIR, 'data', 'raw', 'bbdd_binary', 'hepatitis_data.csv')
PATH_PROJECT_DATA_IONOS_DIR = Path.joinpath(PATH_PROJECT_DIR, 'data', 'raw', 'bbdd_binary', 'ionosphere.data.csv')
PATH_PROJECT_DATA_SAHEART_DIR = Path.joinpath(PATH_PROJECT_DIR, 'data', 'raw', 'bbdd_binary', 'saheart.csv')
PATH_PROJECT_DATA_AUSTRALIAN_DIR = Path.joinpath(PATH_PROJECT_DIR, 'data', 'raw', 'bbdd_binary', 'australian.csv')
PATH_PROJECT_DATA_HORSE_DIR = Path.joinpath(PATH_PROJECT_DIR, 'data', 'raw', 'bbdd_binary', 'horse-colic.csv')
PATH_PROJECT_DATA_CYLINDER_DIR = Path.joinpath(PATH_PROJECT_DIR, 'data', 'raw', 'bbdd_binary', 'cylinder.csv')
PATH_PROJECT_DATA_DRESSES_DIR = Path.joinpath(PATH_PROJECT_DIR, 'data', 'raw', 'bbdd_binary', 'dresses.csv')
PATH_PROJECT_DATA_LOAN_DIR = Path.joinpath(PATH_PROJECT_DIR, 'data', 'raw', 'bbdd_binary', 'loan.csv')
PATH_PROJECT_DATA_AUTISM_DIR = Path.joinpath(PATH_PROJECT_DIR, 'data', 'raw', 'bbdd_binary', 'autism.csv')
PATH_PROJECT_DATA_THORACIC_DIR = Path.joinpath(PATH_PROJECT_DIR, 'data', 'raw', 'bbdd_binary', 'thoracic.csv')
PATH_PROJECT_DATA_FRAM_DIR = Path.joinpath(PATH_PROJECT_DIR, 'data', 'raw', 'bbdd_binary', 'fram.csv')
PATH_PROJECT_DATA_STENO_DIR = Path.joinpath(PATH_PROJECT_DIR, 'data', 'raw', 'bbdd_binary', 'steno.csv')


PATH_PROJECT_DATA_ANNEALING_DIR = Path.joinpath(PATH_PROJECT_DIR, 'data', 'raw', 'bbdd_multiclass', 'anneal.csv')
PATH_PROJECT_DATA_BRIDGES_DIR = Path.joinpath(PATH_PROJECT_DIR, 'data', 'raw', 'bbdd_multiclass', 'bridges.data.version1.csv')
PATH_PROJECT_DATA_CMC_DIR = Path.joinpath(PATH_PROJECT_DIR, 'data', 'raw', 'bbdd_multiclass', 'cmc.csv')
PATH_PROJECT_DATA_DERMAT_DIR = Path.joinpath(PATH_PROJECT_DIR, 'data', 'raw', 'bbdd_multiclass', 'dermatology.csv')
PATH_PROJECT_DATA_HEART_DIR = Path.joinpath(PATH_PROJECT_DIR, 'data', 'raw', 'bbdd_multiclass', 'heart.csv')
PATH_PROJECT_DATA_TAE_DIR = Path.joinpath(PATH_PROJECT_DIR, 'data', 'raw', 'bbdd_multiclass', 'tae.csv')
PATH_PROJECT_DATA_POST_DIR = Path.joinpath(PATH_PROJECT_DIR, 'data', 'raw', 'bbdd_multiclass', 'post-operative.csv')
PATH_PROJECT_DATA_HYPO_DIR = Path.joinpath(PATH_PROJECT_DIR, 'data', 'raw', 'bbdd_multiclass', 'hypothyroid.csv')
PATH_PROJECT_DATA_AUTOS_DIR = Path.joinpath(PATH_PROJECT_DIR, 'data', 'raw', 'bbdd_multiclass', 'autos.csv')

PATH_PROJECT_DATA_DIABETES_NUM_DIR = Path.joinpath(PATH_PROJECT_DIR, 'data', 'raw', 'diabetes_num.csv')

PATH_PROJECT_DATA_PREPROCESSED = Path.joinpath(PATH_PROJECT_DIR, 'data', 'processed')

# Datasets for binary classification
PATH_PROJECT_DATA_CRX_DIR_PREPROCESSED = Path.joinpath(PATH_PROJECT_DIR, 'data', 'processed', 'crx_preprocessed.csv')
PATH_PROJECT_DATA_DIABETES_DIR_PREPROCESSED = Path.joinpath(PATH_PROJECT_DIR, 'data', 'processed', 'diabetes_preprocessed.csv')
PATH_PROJECT_DATA_GERMAN_DIR_PREPROCESSED = Path.joinpath(PATH_PROJECT_DIR, 'data', 'processed', 'german_preprocessed.csv')
PATH_PROJECT_DATA_HEPATITIS_DIR_PREPROCESSED = Path.joinpath(PATH_PROJECT_DIR, 'data', 'processed', 'hepatitis_preprocessed.csv')
PATH_PROJECT_DATA_IONOS_DIR_PREPROCESSED = Path.joinpath(PATH_PROJECT_DIR, 'data', 'processed', 'ionos_preprocessed.csv')
PATH_PROJECT_DATA_SAHEART_DIR_PREPROCESSED = Path.joinpath(PATH_PROJECT_DIR, 'data', 'processed', 'saheart_preprocessed.csv')
PATH_PROJECT_DATA_AUSTRALIAN_DIR_PREPROCESSED = Path.joinpath(PATH_PROJECT_DIR, 'data', 'processed', 'australian_preprocessed.csv')
PATH_PROJECT_DATA_HORSE_DIR_PREPROCESSED = Path.joinpath(PATH_PROJECT_DIR, 'data', 'processed', 'horse_preprocessed.csv')
PATH_PROJECT_DATA_CYLINDER_DIR_PREPROCESSED = Path.joinpath(PATH_PROJECT_DIR, 'data', 'processed', 'cylinder_preprocessed.csv')
PATH_PROJECT_DATA_DRESSES_DIR_PREPROCESSED = Path.joinpath(PATH_PROJECT_DIR, 'data', 'processed', 'dresses_preprocessed.csv')
PATH_PROJECT_DATA_LOAN_DIR_PREPROCESSED = Path.joinpath(PATH_PROJECT_DIR, 'data', 'processed', 'loan_preprocessed.csv')
PATH_PROJECT_DATA_AUTISM_DIR_PREPROCESSED = Path.joinpath(PATH_PROJECT_DIR, 'data', 'processed', 'autism_preprocessed.csv')
PATH_PROJECT_DATA_THORACIC_DIR_PREPROCESSED = Path.joinpath(PATH_PROJECT_DIR, 'data', 'processed', 'thoracic_preprocessed.csv')


PATH_PROJECT_DATA_ANNEALING_DIR_PREPROCESSED = Path.joinpath(PATH_PROJECT_DIR, 'data', 'processed', 'anneal_preprocessed.csv')
PATH_PROJECT_DATA_BRIDGES_DIR_PREPROCESSED = Path.joinpath(PATH_PROJECT_DIR, 'data', 'processed', 'bridges_preprocessed.csv')
PATH_PROJECT_DATA_CMC_DIR_PREPROCESSED = Path.joinpath(PATH_PROJECT_DIR, 'data', 'processed', 'cmc_preprocessed.csv')
PATH_PROJECT_DATA_DERMAT_DIR_PREPROCESSED = Path.joinpath(PATH_PROJECT_DIR, 'data', 'processed', 'dermat_preprocessed.csv')
PATH_PROJECT_DATA_HEART_DIR_PREPROCESSED = Path.joinpath(PATH_PROJECT_DIR, 'data', 'processed', 'heart_preprocessed.csv')
PATH_PROJECT_DATA_TAE_DIR_PREPROCESSED = Path.joinpath(PATH_PROJECT_DIR, 'data', 'processed', 'tae_preprocessed.csv')
PATH_PROJECT_DATA_POST_DIR_PREPROCESSED = Path.joinpath(PATH_PROJECT_DIR, 'data', 'processed', 'postop_preprocessed.csv')
PATH_PROJECT_DATA_HYPO_DIR_PREPROCESSED = Path.joinpath(PATH_PROJECT_DIR, 'data', 'processed', 'hypo_preprocessed.csv')
PATH_PROJECT_DATA_AUTOS_DIR_PREPROCESSED = Path.joinpath(PATH_PROJECT_DIR, 'data', 'processed', 'autos_preprocessed.csv')

PATH_PROJECT_DATA_DIABETES_NUM_DIR_PREPROCESSED = Path.joinpath(PATH_PROJECT_DIR, 'data', 'processed', 'diabetes_num_preprocessed.csv')

PATH_PROJECT_METRICS = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'metrics')
PATH_PROJECT_NOISY_DATASET = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'noisy_dataset')

PATH_PROJECT_DATA_CRX_DIR_NOISE_HOMO = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'noisy_dataset', 'noisy_dataset_homogeneous_crx.csv')
PATH_PROJECT_DATA_DIABETES_DIR_NOISE_HOMO = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'noisy_dataset', 'noisy_dataset_homogeneous_diabetes.csv')
PATH_PROJECT_DATA_GERMAN_DIR_NOISE_HOMO = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'noisy_dataset', 'noisy_dataset_homogeneous_german.csv')
PATH_PROJECT_DATA_HEPATITIS_DIR_NOISE_HOMO = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'noisy_dataset', 'noisy_dataset_homogeneous_hepatitis.csv')
PATH_PROJECT_DATA_IONOS_DIR_NOISE_HOMO = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'noisy_dataset', 'noisy_dataset_homogeneous_ionos.csv')
PATH_PROJECT_DATA_SAHEART_DIR_NOISE_HOMO = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'noisy_dataset', 'noisy_dataset_homogeneous_saheart.csv')
PATH_PROJECT_DATA_AUSTRALIAN_DIR_NOISE_HOMO = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'noisy_dataset', 'noisy_dataset_homogeneous_australian.csv')
PATH_PROJECT_DATA_HORSE_DIR_NOISE_HOMO = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'noisy_dataset', 'noisy_dataset_homogeneous_horse.csv')
PATH_PROJECT_DATA_CYLINDER_DIR_NOISE_HOMO = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'noisy_dataset', 'noisy_dataset_homogeneous_cylinder.csv')
PATH_PROJECT_DATA_DRESSES_DIR_NOISE_HOMO = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'noisy_dataset', 'noisy_dataset_homogeneous_dresses.csv')
PATH_PROJECT_DATA_LOAN_DIR_NOISE_HOMO = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'noisy_dataset', 'noisy_dataset_homogeneous_loan.csv')
PATH_PROJECT_DATA_AUTISM_DIR_NOISE_HOMO = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'noisy_dataset', 'noisy_dataset_homogeneous_autism.csv')
PATH_PROJECT_DATA_THORACIC_DIR_NOISE_HOMO = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'noisy_dataset', 'noisy_dataset_homogeneous_thoracic.csv')
PATH_PROJECT_DATA_FRAM_DIR_NOISE_HOMO = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'noisy_dataset', 'noisy_dataset_homogeneous_fram.csv')
PATH_PROJECT_DATA_STENO_DIR_NOISE_HOMO = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'noisy_dataset', 'noisy_dataset_homogeneous_steno.csv')
PATH_PROJECT_DATA_FRAM_SECOND_DIR_NOISE_HOMO = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'noisy_dataset', 'noisy_dataset_homogeneous_fram_second.csv')
PATH_PROJECT_DATA_STENO_SECOND_DIR_NOISE_HOMO = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'noisy_dataset', 'noisy_dataset_homogeneous_steno_second.csv')

PATH_PROJECT_DATA_ANNEALING_DIR_NOISE_HOMO = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'noisy_dataset', 'noisy_dataset_homogeneous_anneal.csv')
PATH_PROJECT_DATA_BRIDGES_DIR_NOISE_HOMO = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'noisy_dataset', 'noisy_dataset_homogeneous_bridges.csv')
PATH_PROJECT_DATA_CMC_DIR_NOISE_HOMO = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'noisy_dataset', 'noisy_dataset_homogeneous_cmc.csv')
PATH_PROJECT_DATA_DERMAT_DIR_NOISE_HOMO = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'noisy_dataset', 'noisy_dataset_homogeneous_dermat.csv')
PATH_PROJECT_DATA_HEART_DIR_NOISE_HOMO = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'noisy_dataset', 'noisy_dataset_homogeneous_heart.csv')
PATH_PROJECT_DATA_TAE_DIR_NOISE_HOMO = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'noisy_dataset', 'noisy_dataset_homogeneous_tae.csv')
PATH_PROJECT_DATA_POST_DIR_NOISE_HOMO = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'noisy_dataset', 'noisy_dataset_homogeneous_postop.csv')
PATH_PROJECT_DATA_HYPO_DIR_NOISE_HOMO = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'noisy_dataset', 'noisy_dataset_homogeneous_hypo.csv')
PATH_PROJECT_DATA_AUTOS_DIR_NOISE_HOMO = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'noisy_dataset', 'noisy_dataset_homogeneous_autos.csv')

PATH_PROJECT_DATA_CRX_DIR_NOISE_HETE = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'noisy_dataset', 'noisy_dataset_heterogeneous_crx.csv')
PATH_PROJECT_DATA_DIABETES_DIR_NOISE_HETE = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'noisy_dataset', 'noisy_dataset_heterogeneous_diabetes.csv')
PATH_PROJECT_DATA_GERMAN_DIR_NOISE_HETE = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'noisy_dataset', 'noisy_dataset_heterogeneous_german.csv')
PATH_PROJECT_DATA_HEPATITIS_DIR_NOISE_HETE = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'noisy_dataset', 'noisy_dataset_heterogeneous_hepatitis.csv')
PATH_PROJECT_DATA_IONOS_DIR_NOISE_HETE = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'noisy_dataset', 'noisy_dataset_heterogeneous_ionos.csv')
PATH_PROJECT_DATA_SAHEART_DIR_NOISE_HETE = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'noisy_dataset', 'noisy_dataset_heterogeneous_saheart.csv')
PATH_PROJECT_DATA_AUSTRALIAN_DIR_NOISE_HETE = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'noisy_dataset', 'noisy_dataset_heterogeneous_australian.csv')
PATH_PROJECT_DATA_HORSE_DIR_NOISE_HETE = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'noisy_dataset', 'noisy_dataset_heterogeneous_horse.csv')
PATH_PROJECT_DATA_CYLINDER_DIR_NOISE_HETE = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'noisy_dataset', 'noisy_dataset_heterogeneous_cylinder.csv')
PATH_PROJECT_DATA_DRESSES_DIR_NOISE_HETE = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'noisy_dataset', 'noisy_dataset_heterogeneous_dresses.csv')
PATH_PROJECT_DATA_LOAN_DIR_NOISE_HETE = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'noisy_dataset', 'noisy_dataset_heterogeneous_loan.csv')
PATH_PROJECT_DATA_AUTISM_DIR_NOISE_HETE = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'noisy_dataset', 'noisy_dataset_heterogeneous_autism.csv')
PATH_PROJECT_DATA_THORACIC_DIR_NOISE_HETE = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'noisy_dataset', 'noisy_dataset_heterogeneous_thoracic.csv')
PATH_PROJECT_DATA_FRAM_DIR_NOISE_HETE = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'noisy_dataset', 'noisy_dataset_heterogeneous_fram.csv')
PATH_PROJECT_DATA_STENO_DIR_NOISE_HETE = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'noisy_dataset', 'noisy_dataset_heterogeneous_steno.csv')
PATH_PROJECT_DATA_FRAM_SECOND_DIR_NOISE_HETE = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'noisy_dataset', 'noisy_dataset_heterogeneous_fram_second.csv')
PATH_PROJECT_DATA_STENO_SECOND_DIR_NOISE_HETE = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'noisy_dataset', 'noisy_dataset_heterogeneous_steno_second.csv')
PATH_PROJECT_DATA_ANNEALING_DIR_NOISE_HETE = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'noisy_dataset', 'noisy_dataset_heterogeneous_anneal.csv')
PATH_PROJECT_DATA_BRIDGES_DIR_NOISE_HETE = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'noisy_dataset', 'noisy_dataset_heterogeneous_bridges.csv')
PATH_PROJECT_DATA_CMC_DIR_NOISE_HETE = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'noisy_dataset', 'noisy_dataset_heterogeneous_cmc.csv')
PATH_PROJECT_DATA_DERMAT_DIR_NOISE_HETE = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'noisy_dataset', 'noisy_dataset_heterogeneous_dermat.csv')
PATH_PROJECT_DATA_HEART_DIR_NOISE_HETE = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'noisy_dataset', 'noisy_dataset_heterogeneous_heart.csv')
PATH_PROJECT_DATA_TAE_DIR_NOISE_HETE = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'noisy_dataset', 'noisy_dataset_heterogeneous_tae.csv')
PATH_PROJECT_DATA_POST_DIR_NOISE_HETE = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'noisy_dataset', 'noisy_dataset_heterogeneous_postop.csv')
PATH_PROJECT_DATA_HYPO_DIR_NOISE_HETE = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'noisy_dataset', 'noisy_dataset_heterogeneous_hypo.csv')
PATH_PROJECT_DATA_AUTOS_DIR_NOISE_HETE = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'noisy_dataset', 'noisy_dataset_heterogeneous_autos.csv')

# PATH_PROJECT_DATA_DIABETES_NUM_DIR_NOISE = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'noisy_dataset', 'noisy_dataset_diabetes_num.csv')

PATH_PROJECT_TAB_TO_IMAGE = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image')

PATH_PROJECT_DATA_CRX_DIR_IMAGES = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_preprocessed_crx', 'data')
PATH_PROJECT_DATA_DIABETES_DIR_IMAGES = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_preprocessed_diabetes', 'data')
PATH_PROJECT_DATA_GERMAN_DIR_IMAGES = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_preprocessed_german', 'data')
PATH_PROJECT_DATA_HEPATITIS_DIR_IMAGES = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_preprocessed_hepatitis', 'data')
PATH_PROJECT_DATA_IONOS_DIR_IMAGES = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_preprocessed_ionos', 'data')
PATH_PROJECT_DATA_SAHEART_DIR_IMAGES = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_preprocessed_saheart', 'data')
PATH_PROJECT_DATA_AUSTRALIAN_DIR_IMAGES = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_preprocessed_australian', 'data')
PATH_PROJECT_DATA_HORSE_DIR_IMAGES = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_preprocessed_horse', 'data')
PATH_PROJECT_DATA_CYLINDER_DIR_IMAGES = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_preprocessed_cylinder', 'data')
PATH_PROJECT_DATA_DRESSES_DIR_IMAGES = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_preprocessed_dresses', 'data')
PATH_PROJECT_DATA_LOAN_DIR_IMAGES = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_preprocessed_loan', 'data')
PATH_PROJECT_DATA_AUTISM_DIR_IMAGES = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_preprocessed_autism', 'data')
PATH_PROJECT_DATA_THORACIC_DIR_IMAGES = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_preprocessed_thoracic', 'data')

PATH_PROJECT_DATA_ANNEALING_DIR_IMAGES = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_preprocessed_anneal', 'data')
PATH_PROJECT_DATA_BRIDGES_DIR_IMAGES = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_preprocessed_bridges', 'data')
PATH_PROJECT_DATA_CMC_DIR_IMAGES = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_preprocessed_cmc', 'data')
PATH_PROJECT_DATA_DERMAT_DIR_IMAGES = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_preprocessed_dermat', 'data')
PATH_PROJECT_DATA_HEART_DIR_IMAGES = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_preprocessed_heart', 'data')
PATH_PROJECT_DATA_TAE_DIR_IMAGES = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_preprocessed_tae', 'data')
PATH_PROJECT_DATA_POST_DIR_IMAGES = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_preprocessed_postop', 'data')
PATH_PROJECT_DATA_HYPO_DIR_IMAGES = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_preprocessed_hypo', 'data')
PATH_PROJECT_DATA_AUTOS_DIR_IMAGES = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_preprocessed_autos', 'data')


PATH_PROJECT_DATA_CRX_DIR_IMAGES_INTERPRETABILITY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_preprocessed_crx', 'data_interpretability')
PATH_PROJECT_DATA_DIABETES_DIR_IMAGES_INTERPRETABILITY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_preprocessed_diabetes', 'data_interpretability')
PATH_PROJECT_DATA_GERMAN_DIR_IMAGES_INTERPRETABILITY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_preprocessed_german', 'data_interpretability')
PATH_PROJECT_DATA_HEPATITIS_DIR_IMAGES_INTERPRETABILITY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_preprocessed_hepatitis', 'data_interpretability')
PATH_PROJECT_DATA_IONOS_DIR_IMAGES_INTERPRETABILITY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_preprocessed_ionos', 'data_interpretability')
PATH_PROJECT_DATA_SAHEART_DIR_IMAGES_INTERPRETABILITY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_preprocessed_saheart', 'data_interpretability')
PATH_PROJECT_DATA_AUSTRALIAN_DIR_IMAGES_INTERPRETABILITY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_preprocessed_australian', 'data_interpretability')
PATH_PROJECT_DATA_HORSE_DIR_IMAGES_INTERPRETABILITY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_preprocessed_horse', 'data_interpretability')
PATH_PROJECT_DATA_CYLINDER_DIR_IMAGES_INTERPRETABILITY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_preprocessed_cylinder', 'data_interpretability')
PATH_PROJECT_DATA_DRESSES_DIR_IMAGES_INTERPRETABILITY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_preprocessed_dresses', 'data_interpretability')
PATH_PROJECT_DATA_LOAN_DIR_IMAGES_INTERPRETABILITY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_preprocessed_loan', 'data_interpretability')
PATH_PROJECT_DATA_AUTISM_DIR_IMAGES_INTERPRETABILITY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_preprocessed_autism', 'data_interpretability')
PATH_PROJECT_DATA_THORACIC_DIR_IMAGES_INTERPRETABILITY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_preprocessed_thoracic', 'data_interpretability')

PATH_PROJECT_DATA_FRAM_DIR_IMAGES_INTERPRETABILITY= Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_heterogeneous_fram', 'data_interpretability')
PATH_PROJECT_DATA_STENO_DIR_IMAGES_INTERPRETABILITY= Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_heterogeneous_steno', 'data_interpretability')
PATH_PROJECT_DATA_FRAM_SECOND_DIR_IMAGES_INTERPRETABILITY= Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_heterogeneous_fram_second', 'data_interpretability')
PATH_PROJECT_DATA_STENO_SECOND_DIR_IMAGES_INTERPRETABILITY= Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_heterogeneous_steno_second', 'data_interpretability')

PATH_PROJECT_DATA_FRAM_DIR_IMAGES_HOMO_INTERPRETABILITY= Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_homogeneous_fram', 'data_interpretability')
PATH_PROJECT_DATA_STENO_DIR_IMAGES_HOMO_INTERPRETABILITY= Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_homogeneous_steno', 'data_interpretability')
PATH_PROJECT_DATA_FRAM_SECOND_DIR_IMAGES_HOMO_INTERPRETABILITY= Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_homogeneous_fram_second', 'data_interpretability')
PATH_PROJECT_DATA_STENO_SECOND_DIR_IMAGES_HOMO_INTERPRETABILITY= Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_homogeneous_steno_second', 'data_interpretability')

PATH_PROJECT_DATA_FRAM_DIR_IMAGES_HETE_INTERPRETABILITY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_heterogeneous_fram', 'data_interpretability')
PATH_PROJECT_DATA_STENO_DIR_IMAGES_HETE_INTERPRETABILITY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_heterogeneous_steno', 'data_interpretability')
PATH_PROJECT_DATA_STENO_SECOND_DIR_IMAGES_HETE_INTERPRETABILITY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_heterogeneous_steno_second', 'data_interpretability')


PATH_PROJECT_DATA_ANNEALING_DIR_IMAGES_INTERPRETABILITY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_preprocessed_anneal', 'data_interpretability')
PATH_PROJECT_DATA_BRIDGES_DIR_IMAGES_INTERPRETABILITY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_preprocessed_bridges', 'data_interpretability')
PATH_PROJECT_DATA_CMC_DIR_IMAGES_INTERPRETABILITY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_preprocessed_cmc', 'data_interpretability')
PATH_PROJECT_DATA_DERMAT_DIR_IMAGES_INTERPRETABILITY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_preprocessed_dermat', 'data_interpretability')
PATH_PROJECT_DATA_HEART_DIR_IMAGES_INTERPRETABILITY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_preprocessed_heart', 'data_interpretability')
PATH_PROJECT_DATA_TAE_DIR_IMAGES_INTERPRETABILITY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_preprocessed_tae', 'data_interpretability')
PATH_PROJECT_DATA_POST_DIR_IMAGES_INTERPRETABILITY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_preprocessed_postop', 'data_interpretability')
PATH_PROJECT_DATA_HYPO_DIR_IMAGES_INTERPRETABILITY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_preprocessed_hypo', 'data_interpretability')
PATH_PROJECT_DATA_AUTOS_DIR_IMAGES_INTERPRETABILITY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_preprocessed_autos', 'data_interpretability')

PATH_PROJECT_DATA_IONOS_DIR_IMAGES_CLUSTER = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_cluster_ionos', 'data')
PATH_PROJECT_DATA_CRX_DIR_IMAGES_CLUSTER = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_cluster_crx', 'data')
PATH_PROJECT_DATA_GERMAN_DIR_IMAGES_CLUSTER = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_cluster_german', 'data')
PATH_PROJECT_DATA_SPAMBASE_DIR_IMAGES_CLUSTER = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_cluster_spambase', 'data')

PATH_PROJECT_DATA_CRX_DIR_IMAGES_HOMO = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_homogeneous_crx', 'data')
PATH_PROJECT_DATA_DIABETES_DIR_IMAGES_HOMO = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_homogeneous_diabetes', 'data')
PATH_PROJECT_DATA_GERMAN_DIR_IMAGES_HOMO = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_homogeneous_german', 'data')
PATH_PROJECT_DATA_HEPATITIS_DIR_IMAGES_HOMO = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_homogeneous_hepatitis', 'data')
PATH_PROJECT_DATA_IONOS_DIR_IMAGES_HOMO = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_homogeneous_ionos', 'data')
PATH_PROJECT_DATA_SAHEART_DIR_IMAGES_HOMO = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_homogeneous_saheart', 'data')
PATH_PROJECT_DATA_AUSTRALIAN_DIR_IMAGES_HOMO = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_homogeneous_australian', 'data')
PATH_PROJECT_DATA_HORSE_DIR_IMAGES_HOMO = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_homogeneous_horse', 'data')
PATH_PROJECT_DATA_CYLINDER_DIR_IMAGES_HOMO = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_homogeneous_cylinder', 'data')
PATH_PROJECT_DATA_DRESSES_DIR_IMAGES_HOMO = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_homogeneous_dresses', 'data')
PATH_PROJECT_DATA_LOAN_DIR_IMAGES_HOMO = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_homogeneous_loan', 'data')
PATH_PROJECT_DATA_AUTISM_DIR_IMAGES_HOMO = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_homogeneous_autism', 'data')
PATH_PROJECT_DATA_THORACIC_DIR_IMAGES_HOMO = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_homogeneous_thoracic', 'data')
PATH_PROJECT_DATA_FRAM_DIR_IMAGES_HOMO = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_homogeneous_fram', 'data')
PATH_PROJECT_DATA_STENO_DIR_IMAGES_HOMO = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_homogeneous_steno', 'data')
PATH_PROJECT_DATA_FRAM_SECOND_DIR_IMAGES_HOMO = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_homogeneous_fram_second', 'data')
PATH_PROJECT_DATA_STENO_SECOND_DIR_IMAGES_HOMO = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_homogeneous_steno_second', 'data')

PATH_PROJECT_DATA_ANNEALING_DIR_IMAGES_HOMO = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_homogeneous_anneal', 'data')
PATH_PROJECT_DATA_BRIDGES_DIR_IMAGES_HOMO = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_homogeneous_bridges', 'data')
PATH_PROJECT_DATA_CMC_DIR_IMAGES_HOMO = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_homogeneous_cmc', 'data')
PATH_PROJECT_DATA_DERMAT_DIR_IMAGES_HOMO = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_homogeneous_dermat', 'data')
PATH_PROJECT_DATA_HEART_DIR_IMAGES_HOMO = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_homogeneous_heart', 'data')
PATH_PROJECT_DATA_TAE_DIR_IMAGES_HOMO = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_homogeneous_tae', 'data')
PATH_PROJECT_DATA_POST_DIR_IMAGES_HOMO = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_homogeneous_postop', 'data')
PATH_PROJECT_DATA_HYPO_DIR_IMAGES_HOMO = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_homogeneous_hypo', 'data')
PATH_PROJECT_DATA_AUTOS_DIR_IMAGES_HOMO = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_homogeneous_autos', 'data')


PATH_PROJECT_DATA_CRX_DIR_IMAGES_HOMO_INTERPRETABILITY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_homogeneous_crx', 'data_interpretability')
PATH_PROJECT_DATA_DIABETES_DIR_IMAGES_HOMO_INTERPRETABILITY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_homogeneous_diabetes', 'data_interpretability')
PATH_PROJECT_DATA_GERMAN_DIR_IMAGES_HOMO_INTERPRETABILITY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_homogeneous_german', 'data_interpretability')
PATH_PROJECT_DATA_HEPATITIS_DIR_IMAGES_HOMO_INTERPRETABILITY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_homogeneous_hepatitis', 'data_interpretability')
PATH_PROJECT_DATA_IONOS_DIR_IMAGES_HOMO_INTERPRETABILITY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_homogeneous_ionos', 'data_interpretability')
PATH_PROJECT_DATA_SAHEART_DIR_IMAGES_HOMO_INTERPRETABILITY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_homogeneous_saheart', 'data_interpretability')
PATH_PROJECT_DATA_AUSTRALIAN_DIR_IMAGES_HOMO_INTERPRETABILITY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_homogeneous_australian', 'data_interpretability')
PATH_PROJECT_DATA_HORSE_DIR_IMAGES_HOMO_INTERPRETABILITY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_homogeneous_horse', 'data_interpretability')
PATH_PROJECT_DATA_CYLINDER_DIR_IMAGES_HOMO_INTERPRETABILITY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_homogeneous_cylinder', 'data_interpretability')
PATH_PROJECT_DATA_DRESSES_DIR_IMAGES_HOMO_INTERPRETABILITY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_homogeneous_dresses', 'data_interpretability')
PATH_PROJECT_DATA_LOAN_DIR_IMAGES_HOMO_INTERPRETABILITY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_homogeneous_loan', 'data_interpretability')
PATH_PROJECT_DATA_AUTISM_DIR_IMAGES_HOMO_INTERPRETABILITY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_homogeneous_autism', 'data_interpretability')
PATH_PROJECT_DATA_THORACIC_DIR_IMAGES_HOMO_INTERPRETABILITY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_homogeneous_thoracic', 'data_interpretability')

PATH_PROJECT_DATA_ANNEALING_DIR_IMAGES_HOMO_INTERPRETABILITY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_homogeneous_anneal', 'data_interpretability')
PATH_PROJECT_DATA_BRIDGES_DIR_IMAGES_HOMO_INTERPRETABILITY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_homogeneous_bridges', 'data_interpretability')
PATH_PROJECT_DATA_CMC_DIR_IMAGES_HOMO_INTERPRETABILITY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_homogeneous_cmc', 'data_interpretability')
PATH_PROJECT_DATA_DERMAT_DIR_IMAGES_HOMO_INTERPRETABILITY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_homogeneous_dermat', 'data_interpretability')
PATH_PROJECT_DATA_HEART_DIR_IMAGES_HOMO_INTERPRETABILITY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_homogeneous_heart', 'data_interpretability')
PATH_PROJECT_DATA_TAE_DIR_IMAGES_HOMO_INTERPRETABILITY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_homogeneous_tae', 'data_interpretability')
PATH_PROJECT_DATA_POST_DIR_IMAGES_HOMO_INTERPRETABILITY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_homogeneous_postop', 'data_interpretability')
PATH_PROJECT_DATA_HYPO_DIR_IMAGES_HOMO_INTERPRETABILITY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_homogeneous_hypo', 'data_interpretability')
PATH_PROJECT_DATA_AUTOS_DIR_IMAGES_HOMO_INTERPRETABILITY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_homogeneous_autos', 'data_interpretability')


PATH_PROJECT_DATA_CRX_DIR_IMAGES_HETE = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_heterogeneous_crx', 'data')
PATH_PROJECT_DATA_DIABETES_DIR_IMAGES_HETE = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_heterogeneous_diabetes', 'data')
PATH_PROJECT_DATA_GERMAN_DIR_IMAGES_HETE = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_heterogeneous_german', 'data')
PATH_PROJECT_DATA_HEPATITIS_DIR_IMAGES_HETE = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_heterogeneous_hepatitis', 'data')
PATH_PROJECT_DATA_IONOS_DIR_IMAGES_HETE = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_heterogeneous_ionos', 'data')
PATH_PROJECT_DATA_SAHEART_DIR_IMAGES_HETE = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_heterogeneous_saheart', 'data')
PATH_PROJECT_DATA_AUSTRALIAN_DIR_IMAGES_HETE = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_heterogeneous_australian', 'data')
PATH_PROJECT_DATA_HORSE_DIR_IMAGES_HETE = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_heterogeneous_horse', 'data')
PATH_PROJECT_DATA_CYLINDER_DIR_IMAGES_HETE = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_heterogeneous_cylinder', 'data')
PATH_PROJECT_DATA_DRESSES_DIR_IMAGES_HETE = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_heterogeneous_dresses', 'data')
PATH_PROJECT_DATA_LOAN_DIR_IMAGES_HETE = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_heterogeneous_loan', 'data')
PATH_PROJECT_DATA_AUTISM_DIR_IMAGES_HETE = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_heterogeneous_autism', 'data')
PATH_PROJECT_DATA_THORACIC_DIR_IMAGES_HETE = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_heterogeneous_thoracic', 'data')
PATH_PROJECT_DATA_FRAM_DIR_IMAGES_HETE = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_heterogeneous_fram', 'data')
PATH_PROJECT_DATA_STENO_DIR_IMAGES_HETE = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_heterogeneous_steno', 'data')
PATH_PROJECT_DATA_FRAM_SECOND_DIR_IMAGES_HETE = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_heterogeneous_fram_second', 'data')
PATH_PROJECT_DATA_STENO_SECOND_DIR_IMAGES_HETE = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_heterogeneous_steno_second', 'data')

PATH_PROJECT_DATA_ANNEALING_DIR_IMAGES_HETE = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_heterogeneous_anneal', 'data')
PATH_PROJECT_DATA_BRIDGES_DIR_IMAGES_HETE = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_heterogeneous_bridges', 'data')
PATH_PROJECT_DATA_CMC_DIR_IMAGES_HETE = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_heterogeneous_cmc', 'data')
PATH_PROJECT_DATA_DERMAT_DIR_IMAGES_HETE = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_heterogeneous_dermat', 'data')
PATH_PROJECT_DATA_HEART_DIR_IMAGES_HETE = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_heterogeneous_heart', 'data')
PATH_PROJECT_DATA_TAE_DIR_IMAGES_HETE = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_heterogeneous_tae', 'data')
PATH_PROJECT_DATA_POST_DIR_IMAGES_HETE = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_heterogeneous_postop', 'data')
PATH_PROJECT_DATA_HYPO_DIR_IMAGES_HETE = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_heterogeneous_hypo', 'data')
PATH_PROJECT_DATA_AUTOS_DIR_IMAGES_HETE = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_heterogeneous_autos', 'data')


PATH_PROJECT_DATA_CRX_DIR_IMAGES_HETE_INTERPRETABILITY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_heterogeneous_crx', 'data_interpretability')
PATH_PROJECT_DATA_DIABETES_DIR_IMAGES_HETE_INTERPRETABILITY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_heterogeneous_diabetes', 'data_interpretability')
PATH_PROJECT_DATA_GERMAN_DIR_IMAGES_HETE_INTERPRETABILITY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_heterogeneous_german', 'data_interpretability')
PATH_PROJECT_DATA_HEPATITIS_DIR_IMAGES_HETE_INTERPRETABILITY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_heterogeneous_hepatitis', 'data_interpretability')
PATH_PROJECT_DATA_IONOS_DIR_IMAGES_HETE_INTERPRETABILITY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_heterogeneous_ionos', 'data_interpretability')
PATH_PROJECT_DATA_SAHEART_DIR_IMAGES_HETE_INTERPRETABILITY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_heterogeneous_saheart', 'data_interpretability')
PATH_PROJECT_DATA_AUSTRALIAN_DIR_IMAGES_HETE_INTERPRETABILITY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_heterogeneous_australian', 'data_interpretability')
PATH_PROJECT_DATA_HORSE_DIR_IMAGES_HETE_INTERPRETABILITY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_heterogeneous_horse', 'data_interpretability')
PATH_PROJECT_DATA_CYLINDER_DIR_IMAGES_HETE_INTERPRETABILITY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_heterogeneous_cylinder', 'data_interpretability')
PATH_PROJECT_DATA_DRESSES_DIR_IMAGES_HETE_INTERPRETABILITY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_heterogeneous_dresses', 'data_interpretability')
PATH_PROJECT_DATA_LOAN_DIR_IMAGES_HETE_INTERPRETABILITY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_heterogeneous_loan', 'data_interpretability')
PATH_PROJECT_DATA_AUTISM_DIR_IMAGES_HETE_INTERPRETABILITY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_heterogeneous_autism', 'data_interpretability')
PATH_PROJECT_DATA_THORACIC_DIR_IMAGES_HETE_INTERPRETABILITY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_heterogeneous_thoracic', 'data_interpretability')

PATH_PROJECT_DATA_ANNEALING_DIR_IMAGES_HETE_INTERPRETABILITY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_heterogeneous_anneal', 'data_interpretability')
PATH_PROJECT_DATA_BRIDGES_DIR_IMAGES_HETE_INTERPRETABILITY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_heterogeneous_bridges', 'data_interpretability')
PATH_PROJECT_DATA_CMC_DIR_IMAGES_HETE_INTERPRETABILITY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_heterogeneous_cmc', 'data_interpretability')
PATH_PROJECT_DATA_DERMAT_DIR_IMAGES_HETE_INTERPRETABILITY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_heterogeneous_dermat', 'data_interpretability')
PATH_PROJECT_DATA_HEART_DIR_IMAGES_HETE_INTERPRETABILITY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_heterogeneous_heart', 'data_interpretability')
PATH_PROJECT_DATA_TAE_DIR_IMAGES_HETE_INTERPRETABILITY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_heterogeneous_tae', 'data_interpretability')
PATH_PROJECT_DATA_POST_DIR_IMAGES_HETE_INTERPRETABILITY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_heterogeneous_postop', 'data_interpretability')
PATH_PROJECT_DATA_HYPO_DIR_IMAGES_HETE_INTERPRETABILITY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_heterogeneous_hypo', 'data_interpretability')
PATH_PROJECT_DATA_AUTOS_DIR_IMAGES_HETE_INTERPRETABILITY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_heterogeneous_autos', 'data_interpretability')


# PATH_PROJECT_DATA_DIABETES_NUM_DIR_IMAGES = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'tab_to_image', 'image_diabetes_num', 'data')

PATH_PROJECT_SAVE_MODEL = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'models')

PATH_PROJECT_SAVE_OVERSAMPLED = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'oversamplers')
PATH_PROJECT_CTGAN_NOISE = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'noisy_dataset', 'ctgan')
PATH_PROJECT_CTGAN_NO_NOISE = Path.joinpath(PATH_PROJECT_DIR, 'data', 'processed', 'ctgan')

PATH_PROJECT_PAPER_IMAGES_MODEL = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'paper_images')

PATH_PROJECT_SAVE_PLOTS = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'plots')

PATH_PROJECT_SAVE_HEATMAPS = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'heatmaps')

DICT_DATASETS_MAP = {
    'fram': 'fram',
    'steno': 'steno',
    'fram_second': 'fram_second',
    'steno_second': 'steno_second',
    'crx': 'crx',
    'diabetes': 'diabetes',
    'german': 'german',
    'hepatitis': 'hepatitis',
    'ionos': 'ionosphere.data',
    'saheart': 'saheart',
    'australian': 'australian',
    'spambase': 'spambase',
    'horse': Path(PATH_PROJECT_DATA_HORSE_DIR_PREPROCESSED),
    'cylinder': Path(PATH_PROJECT_DATA_CYLINDER_DIR_PREPROCESSED),
    'dresses': Path(PATH_PROJECT_DATA_DRESSES_DIR_PREPROCESSED),
    'loan': Path(PATH_PROJECT_DATA_LOAN_DIR_PREPROCESSED),
    'autism': Path(PATH_PROJECT_DATA_AUTISM_DIR_PREPROCESSED),
    'thoracic': Path(PATH_PROJECT_DATA_THORACIC_DIR_PREPROCESSED),

    'dermat': Path(PATH_PROJECT_DATA_DERMAT_DIR_PREPROCESSED),
    'cmc': Path(PATH_PROJECT_DATA_CMC_DIR_PREPROCESSED),
    'heart': Path(PATH_PROJECT_DATA_HEART_DIR_PREPROCESSED),
    'tae': Path(PATH_PROJECT_DATA_TAE_DIR_PREPROCESSED),
    'anneal': Path(PATH_PROJECT_DATA_ANNEALING_DIR_PREPROCESSED),
    'bridges': Path(PATH_PROJECT_DATA_BRIDGES_DIR_PREPROCESSED),
    'diabetes_num': Path(PATH_PROJECT_DATA_DIABETES_NUM_DIR_PREPROCESSED),
    'postop': Path(PATH_PROJECT_DATA_POST_DIR_PREPROCESSED),
    'hypo': Path(PATH_PROJECT_DATA_HYPO_DIR_PREPROCESSED),
    'autos': Path(PATH_PROJECT_DATA_AUTOS_DIR_PREPROCESSED)


}

DICT_DATASETS_MAP_HOMO = {
    'crx': Path(PATH_PROJECT_DATA_CRX_DIR_NOISE_HOMO),
    'diabetes': Path(PATH_PROJECT_DATA_DIABETES_DIR_NOISE_HOMO),
    'german': Path(PATH_PROJECT_DATA_GERMAN_DIR_NOISE_HOMO),
    'hepatitis': Path(PATH_PROJECT_DATA_HEPATITIS_DIR_NOISE_HOMO),
    'ionos': Path(PATH_PROJECT_DATA_IONOS_DIR_NOISE_HOMO),
    'saheart': Path(PATH_PROJECT_DATA_SAHEART_DIR_NOISE_HOMO),
    'australian': Path(PATH_PROJECT_DATA_AUSTRALIAN_DIR_NOISE_HOMO),
    'horse': Path(PATH_PROJECT_DATA_HORSE_DIR_NOISE_HOMO),
    'cylinder': Path(PATH_PROJECT_DATA_CYLINDER_DIR_NOISE_HOMO),
    'dresses': Path(PATH_PROJECT_DATA_DRESSES_DIR_NOISE_HOMO),
    'loan': Path(PATH_PROJECT_DATA_LOAN_DIR_NOISE_HOMO),
    'autism': Path(PATH_PROJECT_DATA_AUTISM_DIR_NOISE_HOMO),
    'thoracic': Path(PATH_PROJECT_DATA_THORACIC_DIR_NOISE_HOMO),
    'fram': Path(PATH_PROJECT_DATA_FRAM_DIR_NOISE_HOMO),
    'steno': Path(PATH_PROJECT_DATA_STENO_DIR_NOISE_HOMO),
    'fram_second': Path(PATH_PROJECT_DATA_FRAM_SECOND_DIR_NOISE_HOMO),
    'steno_second': Path(PATH_PROJECT_DATA_STENO_SECOND_DIR_NOISE_HOMO),
    'dermat': Path(PATH_PROJECT_DATA_DERMAT_DIR_NOISE_HOMO),
    'cmc': Path(PATH_PROJECT_DATA_CMC_DIR_NOISE_HOMO),
    'heart': Path(PATH_PROJECT_DATA_HEART_DIR_NOISE_HOMO),
    'tae': Path(PATH_PROJECT_DATA_TAE_DIR_NOISE_HOMO),
    'anneal': Path(PATH_PROJECT_DATA_ANNEALING_DIR_NOISE_HOMO),
    'bridges': Path(PATH_PROJECT_DATA_BRIDGES_DIR_NOISE_HOMO),
    'postop': Path(PATH_PROJECT_DATA_POST_DIR_NOISE_HOMO),
    'hypo': Path(PATH_PROJECT_DATA_HYPO_DIR_NOISE_HOMO),
    'autos': Path(PATH_PROJECT_DATA_AUTOS_DIR_NOISE_HOMO)

}

DICT_DATASETS_MAP_HETE = {
    'crx': Path(PATH_PROJECT_DATA_CRX_DIR_NOISE_HETE),
    'diabetes': Path(PATH_PROJECT_DATA_DIABETES_DIR_NOISE_HETE),
    'german': Path(PATH_PROJECT_DATA_GERMAN_DIR_NOISE_HETE),
    'hepatitis': Path(PATH_PROJECT_DATA_HEPATITIS_DIR_NOISE_HETE),
    'ionos': Path(PATH_PROJECT_DATA_IONOS_DIR_NOISE_HETE),
    'saheart': Path(PATH_PROJECT_DATA_SAHEART_DIR_NOISE_HETE),
    'australian': Path(PATH_PROJECT_DATA_AUSTRALIAN_DIR_NOISE_HETE),
    'horse': Path(PATH_PROJECT_DATA_HORSE_DIR_NOISE_HETE),
    'cylinder': Path(PATH_PROJECT_DATA_CYLINDER_DIR_NOISE_HETE),
    'dresses': Path(PATH_PROJECT_DATA_DRESSES_DIR_NOISE_HETE),
    'loan': Path(PATH_PROJECT_DATA_LOAN_DIR_NOISE_HETE),
    'autism': Path(PATH_PROJECT_DATA_AUTISM_DIR_NOISE_HETE),
    'thoracic': Path(PATH_PROJECT_DATA_THORACIC_DIR_NOISE_HETE),
    'fram': Path(PATH_PROJECT_DATA_FRAM_DIR_NOISE_HETE),
    'steno': Path(PATH_PROJECT_DATA_STENO_DIR_NOISE_HETE),
    'fram_second': Path(PATH_PROJECT_DATA_FRAM_SECOND_DIR_NOISE_HETE),
    'steno_second': Path(PATH_PROJECT_DATA_STENO_SECOND_DIR_NOISE_HETE),

    'dermat': Path(PATH_PROJECT_DATA_DERMAT_DIR_NOISE_HETE),
    'cmc': Path(PATH_PROJECT_DATA_CMC_DIR_NOISE_HETE),
    'heart': Path(PATH_PROJECT_DATA_HEART_DIR_NOISE_HETE),
    'tae': Path(PATH_PROJECT_DATA_TAE_DIR_NOISE_HETE),
    'anneal': Path(PATH_PROJECT_DATA_ANNEALING_DIR_NOISE_HETE),
    'bridges': Path(PATH_PROJECT_DATA_BRIDGES_DIR_NOISE_HETE),
    'postop': Path(PATH_PROJECT_DATA_POST_DIR_NOISE_HETE),
    'hypo': Path(PATH_PROJECT_DATA_HYPO_DIR_NOISE_HETE),
    'autos': Path(PATH_PROJECT_DATA_AUTOS_DIR_NOISE_HETE)

}

DICT_IMAGES_MAP = {
    'crx': Path(PATH_PROJECT_DATA_CRX_DIR_IMAGES),
    'diabetes': Path(PATH_PROJECT_DATA_DIABETES_DIR_IMAGES),
    'german': Path(PATH_PROJECT_DATA_GERMAN_DIR_IMAGES),
    'hepatitis': Path(PATH_PROJECT_DATA_HEPATITIS_DIR_IMAGES),
    'ionos': Path(PATH_PROJECT_DATA_IONOS_DIR_IMAGES),
    'saheart': Path(PATH_PROJECT_DATA_SAHEART_DIR_IMAGES),
    'australian': Path(PATH_PROJECT_DATA_AUSTRALIAN_DIR_IMAGES),
    'horse': Path(PATH_PROJECT_DATA_HORSE_DIR_IMAGES),
    'cylinder': Path(PATH_PROJECT_DATA_CYLINDER_DIR_IMAGES),
    'dresses': Path(PATH_PROJECT_DATA_DRESSES_DIR_IMAGES),
    'loan': Path(PATH_PROJECT_DATA_LOAN_DIR_IMAGES),
    'autism': Path(PATH_PROJECT_DATA_AUTISM_DIR_IMAGES),
    'thoracic': Path(PATH_PROJECT_DATA_THORACIC_DIR_IMAGES),

    'dermat': Path(PATH_PROJECT_DATA_DERMAT_DIR_IMAGES),
    'cmc': Path(PATH_PROJECT_DATA_CMC_DIR_IMAGES),
    'heart': Path(PATH_PROJECT_DATA_HEART_DIR_IMAGES),
    'tae': Path(PATH_PROJECT_DATA_TAE_DIR_IMAGES),
    'anneal': Path(PATH_PROJECT_DATA_ANNEALING_DIR_IMAGES),
    'bridges': Path(PATH_PROJECT_DATA_BRIDGES_DIR_IMAGES),
    'postop': Path(PATH_PROJECT_DATA_POST_DIR_IMAGES),
    'hypo': Path(PATH_PROJECT_DATA_HYPO_DIR_IMAGES),
    'autos': Path(PATH_PROJECT_DATA_AUTOS_DIR_IMAGES)

    }


DICT_IMAGES_MAP_CLUSTER = {
    'ionos': Path(PATH_PROJECT_DATA_IONOS_DIR_IMAGES_CLUSTER),
    'crx': Path(PATH_PROJECT_DATA_CRX_DIR_IMAGES_CLUSTER),
    'german': Path(PATH_PROJECT_DATA_GERMAN_DIR_IMAGES_CLUSTER),
    'spambase': Path(PATH_PROJECT_DATA_SPAMBASE_DIR_IMAGES_CLUSTER),
}

DICT_IMAGES_MAP_HOMO = {
    'crx': Path(PATH_PROJECT_DATA_CRX_DIR_IMAGES_HOMO),
    'diabetes': Path(PATH_PROJECT_DATA_DIABETES_DIR_IMAGES_HOMO),
    'german': Path(PATH_PROJECT_DATA_GERMAN_DIR_IMAGES_HOMO),
    'hepatitis': Path(PATH_PROJECT_DATA_HEPATITIS_DIR_IMAGES_HOMO),
    'ionos': Path(PATH_PROJECT_DATA_IONOS_DIR_IMAGES_HOMO),
    'saheart': Path(PATH_PROJECT_DATA_SAHEART_DIR_IMAGES_HOMO),
    'australian': Path(PATH_PROJECT_DATA_AUSTRALIAN_DIR_IMAGES_HOMO),
    'horse': Path(PATH_PROJECT_DATA_HORSE_DIR_IMAGES_HOMO),
    'cylinder': Path(PATH_PROJECT_DATA_CYLINDER_DIR_IMAGES_HOMO),
    'dresses': Path(PATH_PROJECT_DATA_DRESSES_DIR_IMAGES_HOMO),
    'loan': Path(PATH_PROJECT_DATA_LOAN_DIR_IMAGES_HOMO),
    'autism': Path(PATH_PROJECT_DATA_AUTISM_DIR_IMAGES_HOMO),
    'thoracic': Path(PATH_PROJECT_DATA_THORACIC_DIR_IMAGES_HOMO),
    'fram': Path(PATH_PROJECT_DATA_FRAM_DIR_IMAGES_HOMO),
    'steno': Path(PATH_PROJECT_DATA_STENO_DIR_IMAGES_HOMO),
    'fram_second': Path(PATH_PROJECT_DATA_FRAM_SECOND_DIR_IMAGES_HOMO),
    'steno_second': Path(PATH_PROJECT_DATA_STENO_SECOND_DIR_IMAGES_HOMO),
    'dermat': Path(PATH_PROJECT_DATA_DERMAT_DIR_IMAGES_HOMO),
    'cmc': Path(PATH_PROJECT_DATA_CMC_DIR_IMAGES_HOMO),
    'heart': Path(PATH_PROJECT_DATA_HEART_DIR_IMAGES_HOMO),
    'tae': Path(PATH_PROJECT_DATA_TAE_DIR_IMAGES_HOMO),
    'anneal': Path(PATH_PROJECT_DATA_ANNEALING_DIR_IMAGES_HOMO),
    'bridges': Path(PATH_PROJECT_DATA_BRIDGES_DIR_IMAGES_HOMO),
    'postop': Path(PATH_PROJECT_DATA_POST_DIR_IMAGES_HOMO),
    'hypo': Path(PATH_PROJECT_DATA_HYPO_DIR_IMAGES_HOMO),
    'autos': Path(PATH_PROJECT_DATA_AUTOS_DIR_IMAGES_HOMO)

    }


DICT_IMAGES_MAP_HETE = {
    'crx': Path(PATH_PROJECT_DATA_CRX_DIR_IMAGES_HETE),
    'diabetes': Path(PATH_PROJECT_DATA_DIABETES_DIR_IMAGES_HETE),
    'german': Path(PATH_PROJECT_DATA_GERMAN_DIR_IMAGES_HETE),
    'hepatitis': Path(PATH_PROJECT_DATA_HEPATITIS_DIR_IMAGES_HETE),
    'ionos': Path(PATH_PROJECT_DATA_IONOS_DIR_IMAGES_HETE),
    'saheart': Path(PATH_PROJECT_DATA_SAHEART_DIR_IMAGES_HETE),
    'australian': Path(PATH_PROJECT_DATA_AUSTRALIAN_DIR_IMAGES_HETE),
    'horse': Path(PATH_PROJECT_DATA_HORSE_DIR_IMAGES_HETE),
    'cylinder': Path(PATH_PROJECT_DATA_CYLINDER_DIR_IMAGES_HETE),
    'dresses': Path(PATH_PROJECT_DATA_DRESSES_DIR_IMAGES_HETE),
    'loan': Path(PATH_PROJECT_DATA_LOAN_DIR_IMAGES_HETE),
    'autism': Path(PATH_PROJECT_DATA_AUTISM_DIR_IMAGES_HETE),
    'thoracic': Path(PATH_PROJECT_DATA_THORACIC_DIR_IMAGES_HETE),
    'fram': Path(PATH_PROJECT_DATA_FRAM_DIR_IMAGES_HETE),
    'steno': Path(PATH_PROJECT_DATA_STENO_DIR_IMAGES_HETE),
    'fram_second': Path(PATH_PROJECT_DATA_FRAM_SECOND_DIR_IMAGES_HETE),
    'steno_second': Path(PATH_PROJECT_DATA_STENO_SECOND_DIR_IMAGES_HETE),

    'dermat': Path(PATH_PROJECT_DATA_DERMAT_DIR_IMAGES_HETE),
    'cmc': Path(PATH_PROJECT_DATA_CMC_DIR_IMAGES_HETE),
    'heart': Path(PATH_PROJECT_DATA_HEART_DIR_IMAGES_HETE),
    'tae': Path(PATH_PROJECT_DATA_TAE_DIR_IMAGES_HETE),
    'anneal': Path(PATH_PROJECT_DATA_ANNEALING_DIR_IMAGES_HETE),
    'bridges': Path(PATH_PROJECT_DATA_BRIDGES_DIR_IMAGES_HETE),
    'postop': Path(PATH_PROJECT_DATA_POST_DIR_IMAGES_HETE),
    'hypo': Path(PATH_PROJECT_DATA_HYPO_DIR_IMAGES_HETE),
    'autos': Path(PATH_PROJECT_DATA_AUTOS_DIR_IMAGES_HETE)


    }


DICT_IMAGES_MAP_INTERPRETABILITY = {
    'crx': Path(PATH_PROJECT_DATA_CRX_DIR_IMAGES_INTERPRETABILITY),
    'diabetes': Path(PATH_PROJECT_DATA_DIABETES_DIR_IMAGES_INTERPRETABILITY),
    'german': Path(PATH_PROJECT_DATA_GERMAN_DIR_IMAGES_INTERPRETABILITY),
    'hepatitis': Path(PATH_PROJECT_DATA_HEPATITIS_DIR_IMAGES_INTERPRETABILITY),
    'ionos': Path(PATH_PROJECT_DATA_IONOS_DIR_IMAGES_INTERPRETABILITY),
    'saheart': Path(PATH_PROJECT_DATA_SAHEART_DIR_IMAGES_INTERPRETABILITY),
    'australian': Path(PATH_PROJECT_DATA_AUSTRALIAN_DIR_IMAGES_INTERPRETABILITY),
    'horse': Path(PATH_PROJECT_DATA_HORSE_DIR_IMAGES_INTERPRETABILITY),
    'cylinder': Path(PATH_PROJECT_DATA_CYLINDER_DIR_IMAGES_INTERPRETABILITY),
    'dresses': Path(PATH_PROJECT_DATA_DRESSES_DIR_IMAGES_INTERPRETABILITY),
    'loan': Path(PATH_PROJECT_DATA_LOAN_DIR_IMAGES_INTERPRETABILITY),
    'autism': Path(PATH_PROJECT_DATA_AUTISM_DIR_IMAGES_INTERPRETABILITY),
    'thoracic': Path(PATH_PROJECT_DATA_THORACIC_DIR_IMAGES_INTERPRETABILITY),

    'fram': Path(PATH_PROJECT_DATA_FRAM_DIR_IMAGES_INTERPRETABILITY),
    'steno': Path(PATH_PROJECT_DATA_STENO_DIR_IMAGES_INTERPRETABILITY),
    'fram_second': Path(PATH_PROJECT_DATA_FRAM_SECOND_DIR_IMAGES_INTERPRETABILITY),
    'steno_second': Path(PATH_PROJECT_DATA_STENO_SECOND_DIR_IMAGES_INTERPRETABILITY),

    'dermat': Path(PATH_PROJECT_DATA_DERMAT_DIR_IMAGES_INTERPRETABILITY),
    'cmc': Path(PATH_PROJECT_DATA_CMC_DIR_IMAGES_INTERPRETABILITY),
    'heart': Path(PATH_PROJECT_DATA_HEART_DIR_IMAGES_INTERPRETABILITY),
    'tae': Path(PATH_PROJECT_DATA_TAE_DIR_IMAGES_INTERPRETABILITY),
    'anneal': Path(PATH_PROJECT_DATA_ANNEALING_DIR_IMAGES_INTERPRETABILITY),
    'bridges': Path(PATH_PROJECT_DATA_BRIDGES_DIR_IMAGES_INTERPRETABILITY),
    'postop': Path(PATH_PROJECT_DATA_POST_DIR_IMAGES_INTERPRETABILITY),
    'hypo': Path(PATH_PROJECT_DATA_HYPO_DIR_IMAGES_INTERPRETABILITY),
    'autos': Path(PATH_PROJECT_DATA_AUTOS_DIR_IMAGES_INTERPRETABILITY)
    }

DICT_IMAGES_MAP_HOMO_INTERPRETABILITY = {
    'crx': Path(PATH_PROJECT_DATA_CRX_DIR_IMAGES_HOMO_INTERPRETABILITY),
    'diabetes': Path(PATH_PROJECT_DATA_DIABETES_DIR_IMAGES_HOMO_INTERPRETABILITY),
    'german': Path(PATH_PROJECT_DATA_GERMAN_DIR_IMAGES_HOMO_INTERPRETABILITY),
    'hepatitis': Path(PATH_PROJECT_DATA_HEPATITIS_DIR_IMAGES_HOMO_INTERPRETABILITY),
    'ionos': Path(PATH_PROJECT_DATA_IONOS_DIR_IMAGES_HOMO_INTERPRETABILITY),
    'saheart': Path(PATH_PROJECT_DATA_SAHEART_DIR_IMAGES_HOMO_INTERPRETABILITY),
    'australian': Path(PATH_PROJECT_DATA_AUSTRALIAN_DIR_IMAGES_HOMO_INTERPRETABILITY),
    'horse': Path(PATH_PROJECT_DATA_HORSE_DIR_IMAGES_HOMO_INTERPRETABILITY),
    'cylinder': Path(PATH_PROJECT_DATA_CYLINDER_DIR_IMAGES_HOMO_INTERPRETABILITY),
    'dresses': Path(PATH_PROJECT_DATA_DRESSES_DIR_IMAGES_HOMO_INTERPRETABILITY),
    'loan': Path(PATH_PROJECT_DATA_LOAN_DIR_IMAGES_HOMO_INTERPRETABILITY),
    'autism': Path(PATH_PROJECT_DATA_AUTISM_DIR_IMAGES_HOMO_INTERPRETABILITY),
    'thoracic': Path(PATH_PROJECT_DATA_THORACIC_DIR_IMAGES_HOMO_INTERPRETABILITY),

    'fram': Path(PATH_PROJECT_DATA_FRAM_DIR_IMAGES_HOMO_INTERPRETABILITY),
    'steno': Path(PATH_PROJECT_DATA_STENO_DIR_IMAGES_HOMO_INTERPRETABILITY),
    'fram_second': Path(PATH_PROJECT_DATA_FRAM_SECOND_DIR_IMAGES_HOMO_INTERPRETABILITY),
    'steno_second': Path(PATH_PROJECT_DATA_STENO_SECOND_DIR_IMAGES_HOMO_INTERPRETABILITY),

    'dermat': Path(PATH_PROJECT_DATA_DERMAT_DIR_IMAGES_HOMO_INTERPRETABILITY),
    'cmc': Path(PATH_PROJECT_DATA_CMC_DIR_IMAGES_HOMO_INTERPRETABILITY),
    'heart': Path(PATH_PROJECT_DATA_HEART_DIR_IMAGES_HOMO_INTERPRETABILITY),
    'tae': Path(PATH_PROJECT_DATA_TAE_DIR_IMAGES_HOMO_INTERPRETABILITY),
    'anneal': Path(PATH_PROJECT_DATA_ANNEALING_DIR_IMAGES_HOMO_INTERPRETABILITY),
    'bridges': Path(PATH_PROJECT_DATA_BRIDGES_DIR_IMAGES_HOMO_INTERPRETABILITY),
    'postop': Path(PATH_PROJECT_DATA_POST_DIR_IMAGES_HOMO_INTERPRETABILITY),
    'hypo': Path(PATH_PROJECT_DATA_HYPO_DIR_IMAGES_HOMO_INTERPRETABILITY),
    'autos': Path(PATH_PROJECT_DATA_AUTOS_DIR_IMAGES_HOMO_INTERPRETABILITY)
    }


DICT_IMAGES_MAP_HETE_INTERPRETABILITY = {
    'crx': Path(PATH_PROJECT_DATA_CRX_DIR_IMAGES_HETE_INTERPRETABILITY),
    'diabetes': Path(PATH_PROJECT_DATA_DIABETES_DIR_IMAGES_HETE_INTERPRETABILITY),
    'german': Path(PATH_PROJECT_DATA_GERMAN_DIR_IMAGES_HETE_INTERPRETABILITY),
    'hepatitis': Path(PATH_PROJECT_DATA_HEPATITIS_DIR_IMAGES_HETE_INTERPRETABILITY),
    'ionos': Path(PATH_PROJECT_DATA_IONOS_DIR_IMAGES_HETE_INTERPRETABILITY),
    'saheart': Path(PATH_PROJECT_DATA_SAHEART_DIR_IMAGES_HETE_INTERPRETABILITY),
    'australian': Path(PATH_PROJECT_DATA_AUSTRALIAN_DIR_IMAGES_HETE_INTERPRETABILITY),
    'horse': Path(PATH_PROJECT_DATA_HORSE_DIR_IMAGES_HETE_INTERPRETABILITY),
    'cylinder': Path(PATH_PROJECT_DATA_CYLINDER_DIR_IMAGES_HETE_INTERPRETABILITY),
    'dresses': Path(PATH_PROJECT_DATA_DRESSES_DIR_IMAGES_HETE_INTERPRETABILITY),
    'loan': Path(PATH_PROJECT_DATA_LOAN_DIR_IMAGES_HETE_INTERPRETABILITY),
    'autism': Path(PATH_PROJECT_DATA_AUTISM_DIR_IMAGES_HETE_INTERPRETABILITY),
    'thoracic': Path(PATH_PROJECT_DATA_THORACIC_DIR_IMAGES_HETE_INTERPRETABILITY),
    'fram': Path(PATH_PROJECT_DATA_FRAM_DIR_IMAGES_HETE_INTERPRETABILITY),
    'steno': Path(PATH_PROJECT_DATA_STENO_DIR_IMAGES_HETE_INTERPRETABILITY),
    'steno_second': Path(PATH_PROJECT_DATA_STENO_SECOND_DIR_IMAGES_HETE_INTERPRETABILITY),
    'dermat': Path(PATH_PROJECT_DATA_DERMAT_DIR_IMAGES_HETE_INTERPRETABILITY),
    'cmc': Path(PATH_PROJECT_DATA_CMC_DIR_IMAGES_HETE_INTERPRETABILITY),
    'heart': Path(PATH_PROJECT_DATA_HEART_DIR_IMAGES_HETE_INTERPRETABILITY),
    'tae': Path(PATH_PROJECT_DATA_TAE_DIR_IMAGES_HETE_INTERPRETABILITY),
    'anneal': Path(PATH_PROJECT_DATA_ANNEALING_DIR_IMAGES_HETE_INTERPRETABILITY),
    'bridges': Path(PATH_PROJECT_DATA_BRIDGES_DIR_IMAGES_HETE_INTERPRETABILITY),
    'postop': Path(PATH_PROJECT_DATA_POST_DIR_IMAGES_HETE_INTERPRETABILITY),
    'hypo': Path(PATH_PROJECT_DATA_HYPO_DIR_IMAGES_HETE_INTERPRETABILITY),
    'autos': Path(PATH_PROJECT_DATA_AUTOS_DIR_IMAGES_HETE_INTERPRETABILITY) 
 
 
    }
