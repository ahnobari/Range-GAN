# Range-GAN
![Range-GAN Architecture](https://github.com/ahnobari/Range-GAN/blob/main/Images/Range-GAN.png?raw=true)
This repository can be used to reproduce the results presented in the paper: [Range-GAN: Range-Constrained Generative Adversarial Network for Conditioned Design Synthesis](https://arxiv.org/abs/2103.06230)

## Required packages

- tensorflow > 2.0.0
- numpy
- matplotlib
- tensorflow_addons
- tensorflow_probability
- tqdm
- Pymcubes

## Project Page
For further information and latest news on the project visit [Range-GAN](https://decode.mit.edu/projects/rangegan/)

## Usage

### 3D Shape Synthesis Background
In Range-GAN we take the approach presented in [IM-NET](https://arxiv.org/abs/1812.02822) to generate 3D shapes. In our project we use the Airplane models in ShapeNet to train an IMAE model to encode 3D shapes into 256 dimensional vectors. If you want to experiment with other 3D shapes please refer to the [IM-NET Code](https://github.com/czq142857/IM-NET). Here we provide the weights of the trained IMAE model we used for our paper (in the Weights directory).

### Data
![Data Sample](https://github.com/ahnobari/Range-GAN/blob/main/Images/data.png?raw=true)
The data used in this paper is the airplane subset of the ShapeNET dataset. However, we only use the encodings produced by IMAE to train our GAN model. The pre calculated values are also provided in the data folder.

As discussed in the [paper](https://arxiv.org/abs/1812.02822) we also augment our data to produce better results. This augmentation is arduous and involves significant manual curation. Therefore, to allow for the replication of the results in the paper we have also included the augmented dataset we used in our work. To reproduce the results in the paper please use the augmented data instead. (Augmentation Code will be provided in the near future with automated curation)

If you require to use your own data, please create hdf5 files with encodings labeled as zs and paramteres that are to be conditioned for using Range-GAN(Further details provided later).

### Training
To train using the data provied here and replicate the results of the paper one of the following can be done for each experiment:

1. Train For Aspect Ratio
   ```bash
   python run_experiment.py --param ratio --save_name ratio
   ```
   add ```--data ./augmented``` to use augmented data (paper results use this)
2. Train For Volume
   ```bash
   python run_experiment.py --param volume --save_name volume
   ```
   add ```--data ./augmented``` to use augmented data (paper results use this)
3. Multi-Objetive Training
   ```bash
   python run_experiment.py --param both --save_name MO
   ```
   add ```--data ./augmented``` to use augmented data (paper results use this)
4. Other Paramters Which Can be Adjusted
   ```
   optional arguments:

   ```
   ```
   -h, --help            show this help message and exit
  --data DATA           The path to the data. Default: ./data use ./augmented if you create an augmented dataset
  --save_name SAVE_NAME
                        The file name of the checkpoint saved in the Weights folder. Default: experiment
  --estimator_lr ESTIMATOR_LR
                        Initial estimator learning rate before decay. Default: 1e-4
  --estimator_train_steps ESTIMATOR_TRAIN_STEPS
                        Number of training steps for estimator. Default: 10000
  --estimator_batch_size ESTIMATOR_BATCH_SIZE
                        Batch size for estimator Default: 128
  --phi PHI             phi. Default: 50.0
  --lambda1 LAMBDA1     lambda1. Default: 4.0
  --lambda2 LAMBDA2     lambda2. Default: 0.1
  --disc_lr DISC_LR     Initial discriminator learning rate before decay. Default: 1e-4
  --gen_lr GEN_LR       Initial discriminator learning rate before decay. Default: 1e-4
  --train_steps TRAIN_STEPS
                        Number of training steps. Default: 50000
  --batch_size BATCH_SIZE
                        Batch size used for GAN training. Default: 32
  --custom_data CUSTOM_DATA
                        If custom data is being used then add this flag and indicate the names of parameters in
                        custom_param values.
  --param PARAM         The parameter to train for. Default: ratio. Either one of: ratio, volume, both
  --custom_dataset CUSTOM_DATASET
                        The name of the dataset in the data folder. Default: None. Depends on the custom dataset.
  --custom_param1 CUSTOM_PARAM1
                        The parameter to train for(Only if using custom dataset). Default: None. Depends on the custom
                        dataset.
  --custom_param2 CUSTOM_PARAM2
                        The parameter to train for(Only if using custom dataset). Default: None. Depends on the custom
                        dataset.
   ```

   The trained models will be saved under the specified folder under the subdirectory of Weights under the each models folder and the result plots will be saved under the results directory. Change save_name argument to control this.
   
   Note that we can set `lambda2` to zero to train without uniformity loss.

4. Using A Custom Dataset
   ```bash
   python run_experiment.py --custom_data --custom_dataset dataset_name --custom_param1 param1 --custom_param2 param2
   ```
   Here you need to add the --custom_data flag and indicate the dataset_name. The dataset_name is the name of the hdf5 files in the data folder. Note that you must have a test set and a training set in the data folder. The custom_param1 argumnets are the name of the parameters in the hdf5 files. param2 is optional if multi-objective is being investigated.