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

## Data
![Data Sample](https://github.com/ahnobari/Range-GAN/blob/main/Images/data.png?raw=true)
The data used in this paper is the airplane subset of the ShapeNET dataset. However, we only use the encodings produced by IMAE to train our GAN model. The pre calculated values are also provided in the data folder.

As discussed in the [paper](https://arxiv.org/abs/1812.02822) we also augment our data to produce better results. This augmentation is arduous and involves significant manual curation. Therefore, to allow for the replication of the results in the paper we have also included the augmented dataset we used in our work. To reproduce the results in the paper please use the augmented data instead. (Augmentation Code will be provided in the near future with automated curation)

If you require to use your own data, please create hdf5 files with encodings labeled as zs and paramteres that are to be conditioned for using Range-GAN(Further details provided later).

1. Go to example directory: 

   ```bash
   cd Synthetic
   ```

2. Train Models:

   ```bash
   python train_models.py
   ```

   positional arguments:
    
   ```
   model PcDGAN or CcGAN
   dataset	dataset name (available datasets are Uniform, Uneven, Donut2D)
   ```

   optional arguments:

   ```
   -h, --help   show this help message and exit
   --dominant_mode DOMINANT_MODE    The dominant mode for uneven dataset. Default: 1, Options: Any integet between 0 and 5
   --mode MODE  Mode of operation, either train or evaluate. Default: Train
   --vicinal_type   The type of vicinal approach. Default: soft, Options: soft, hard
   --kappa    Vicinal loss kappa. If negative automatically calculated and scaled by the absolute value of the number. Default: -1.0 for PcDGAN -2.0 for CcGAN
    --sigma   Vicinal loss sigma. If negative automatically calculated. Default: -1.0
    --lambda0   PcDGAN lambda0. Default: 3.0
    --lambda1   PcDGAN lambda1. Default: 0.5
    --lambert_cutoff    PcDGAN parameter "a". Default: 4.7
    --gen_lr    Generator learning rate. Default: 1e-4
    --disc_lr   Discriminator learning rate. Default: 1e-4
    --train_steps   Number of training steps. Default: 50000
    --batch_size    GAN training Batch size. Default: 32
    --id    experiment ID or name. Default:
    --size  Number of samples to generate at each step for evaluation. Default: 1000
   ```

   The trained models will be saved under the specified dataset folder under the subdirectory of Weights under the each models folder and the result plots will be saved under the directory of the dataset under the subdirectory Evaluation under each models folder. Change the id argument everytime to prevent over-writing previous weights.
   
   Note that we can set `lambda0` and `lambda1` to zeros to train a CcGAN with only singular vicinal loss.

3. To reproduce the results of the paper train atleast 3 versions of each model(although for the paper 10 were trained for each model) by changing the id argument during training and run the following:

    ```bash
    python evaluation.py
    ```

    positional arguments:
    
   ```
   dataset	dataset name (available datasets are Uniform, Uneven, Donut2D)
   ```

   optional arguments:


   ```
   -h, --help   show this help message and exit
   --size SIZE  Number of samples to generate at each step for evaluation. Default: 1000
   ```
   
   After this the Resulting Figures(Similar to what is presented in the paper) will be produced under the dataset directory under the subdirectory of Evaluation.

### Airfoil example

1. Install [XFOIL](https://web.mit.edu/drela/Public/web/xfoil/). Only necessary if you are on Linux. For windows the executables are provided under the XFOIL_Windows folder.

2. Go to example directory:

   ```bash
   cd Airfoil
   ```

3. Download the airfoil dataset [here](https://drive.google.com/drive/folders/1x1SrAX28ajLD0T_zbTUhcYxg2M5kudHm?usp=sharing) and extract the NPY files into `Airfoil/data/`.


4. First train an embedder estimator pair:

   ```bash
   python train_estimator_embedder.py
   ```
    optional arguments:

   ```
   -h, --help   show this help message and exit
   --data   The path to the data. Default: ./data
   --estimator_save_name    The file name of the best checkpoint saved in the weights estimator folder. Default:best_checkpoint
   --embedder_save_name     The file name of the best checkpoint saved in the embedder weights folder. Default:best_checkpoint
   --estimator_lr   Initial estimator learning rate before decay. Default: 1e-4
   --embedder_lr    Initial embedder learning rate before decay. Default: 1e-4
   --estimator_train_steps  Number of training steps for estimator. Default: 10000
   --embedder_train_steps   Number of training steps for embedder. Default: 10000
   --estimator_batch_size   Batch size for estimator Default: 256
   --embedder_batch_size    Batch size for embedder Default: 256
   ```

   The weights of both models will be saved under the Weights folder. Remember if you name the pair differently for GAN training. Also use only one pair for each experiment as cross-validation is done using one pair which both models were based on.

5. Train Models:

   ```bash
   python train_models.py
   ```

   positional arguments:
    
   ```
   model PcDGAN or CcGAN
   ```

   optional arguments:

   ```
   -h, --help   show this help message and exit
   --mode MODE  Mode of operation, either train or evaluate. Default: Train
   --vicinal_type   The type of vicinal approach. Default: soft, Options: soft, hard
   --kappa  Vicinal loss kappa. If negative automatically calculated and scaled by the absolute value of the number. Default: -1.0 for PcDGAN -2.0 for CcGAN
   --sigma  Vicinal loss sigma. If negative automatically calculated. Default: -1.0
   --estimator  Name of the estimator checkpoint saved in the weights folder. Default: best_checkpoint
   --embedder   Name of the embedder checkpoint saved in the weights folder. Default: best_checkpoint
   --lambda0    PcDGAN lambda0. Default: 3.0
   --lambda1    PcDGAN lambda1. Default: 0.4
   --lambert_cutoff PcDGAN parameter "a". Default: 4.7
   --gen_lr GEN_LR  Generator learning rate. Default: 1e-4
   --disc_lr DISC_LR    Discriminator learning rate. Default: 1e-4
   --train_steps    Number of training steps. Default: 20000
   --batch_size     GAN training Batch size. Default: 32
   --size   Number of samples to generate at each step. Default: 1000
   --id     experiment ID or name. Default:
   ```
   Change the id argument everytime to prevent over-writing previous weights. Train each model atleast 3 times to reproduce paper results (for the paper 10 models were trained). The results of each model will be saved under the Evaluation Directory.


6. To reproduce the results of the paper train atleast 3 versions of each model(although for the paper 10 were trained for each model) by changing the id argument during training and run the following:

    ```bash
    python evaluation.py
    ```

   optional arguments:


   ```
   -h, --help   show this help message and exit
   --estimator  Name of the estimator checkpoint saved in the weights folder. Default: best_checkpoint
   --embedder   Name of the embedder checkpoint saved in the weights folder. Default: best_checkpoint
   --simonly    Only evaluate based on simluation. Default: 0(False). Set to 1 for True
   --size   Number of samples to generate at each step. Default: 1000
   ```
   
   After this the Resulting Figures(Similar to what is presented in the paper) will be produced under the Evaluation directory.