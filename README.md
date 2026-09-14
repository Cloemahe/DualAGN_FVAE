# Dual AGN FVAE README

Source code for the Flow-VAE model from A&A paper aa58653-25 "Flow-VAE for Data Augmentation of the Horizon-AGN Dual AGN dataset", Mahé C., Bretonnière H. Slezak E., 2026 (Mahe et al., 2026).

The code presented here allows to train a Variational Auto-Encoder (VAE) on a set of input images, associated with a normalising flow in order to generate large amounts of images with similar properties and visual aspects as the initial dataset. Doing so, one may drastically increase the number of images available in a database.

In Mahe et al. (2026), we applied this model to images extracted from the cosmological simulation Horizon-AGN (Dubois et al, 2014; Volonteri et al., 2020) to generate a large number of images presenting dual active galactic nuclei (AGN). The input images extracted from Horizon-AGN are not public, therefore we instead provide toy images to reproduce them, and we provide in /generated_images a sample of images generated from our method.

The Python environment necessary to run the model can be found in "env.yml".

## I-/ SLURM files :

Files used to command the training or test of the different parts of the model. Each file is associated to both a Python script and a configuration file including the corresponding hyperparameters. 
The whole Flow-VAE model can be trained and tested at once, using the "pipeline.sh" file, providing the right parameters. It is also possible to train, and test either the VAE or the Flow separately.

## II-/ Scripts :

The Python scripts associated with each configuration of the model, along with all individual functions in two distincts sub-folders : 'architecture' and 'utils'. These codes are used as the interface between the user and the VAE or the Flow.

- train_vae.py : Train the VAE and apply tests if asked (parameter "test_only" in configuration file train_vae.ynl in /config). We provide toy images in /data.

- test_vae.py : Test an already trained run of the VAE. An example of trained VAE weights is provided in /checkpoints.

- train_flow.py : Train the normalizing flow and run tests if "test_only" is true. Need trained VAE weights (example provided in /checkpoints) or direct latent space vectors (encodings, examples provided in /data).

- test_flow.py : Test an already trained run of the Flow, given both accurate VAE and Flow weights. 

- flow_vae.py : Train the whole Flow-VAE pipeline. Tests are also applied if "test_only" is true.

- test_gen.py : Generates directly new dual AGN images from a trained Flow run.

- test_gen.ipynb : Notebook used to demonstrate the generation of new dual AGN images from a trained Flow run.

# II-1/ Architecture :

Contains the main architectures and functions necessary for the construction of the models.

- flow_architecture.py : architecture of the Flow : a chain of Bijectors (MAF function).

- flow_training_loop.py : training loop of the Flow.
	
- generate_im.py : generation images from samples of the latent space from a VAE run.
	
- vae_architecture.py : architecture of the VAE : Encoder and Decoder (neural networks).

- vae_model.py : link between the Encoder and the Decoder, forming the VAE.

- vae_training_loop.py : training loop of the VAE.

# II-2/ Utils folder :

- data_generator_utils.py : Functions loading the data (images or vectors) from the pickle files stored in /data.

- data_utils.py : Contains the functions used to print the memory load (print_memory_usage), compute the range of data to store in the different datasets (comp_range), and define the type of preprocessing that needs to be applied to the data (preprocessing_type).

- diagnostics_flow_utils.py : Tests (diagnostics and plots) applied on a given Flow run, simplifying the construction of the model. [Only function used in test_flow.py]

- diagnostics_vae_utils.py : Tests (diagnostics and plots) applied on a given VAE run, simplifying the construction of the model. [Only function used in test_vae.py]

- fourier_utils.py : Contains the functions used to construct the Fourier mode of the VAE loss : create_asymmetrical_ring_mask, fast_fourier_shift, and masked_fft.

- gini_utils.py : Computes the Gini coefficient (Abraham et al. 2003) of each image inside a batch.

- loss_utils.py : VAE loss functions : Classical, Masked, Composite, Fourier.

- plot_utils.py : Metadata plots for the VAE and the Flow (images, histograms, corner plots, gini, profiles).

- preprocessing_utils.py : Preprocessing functions.
 
- profiles_utils.py : Contains the functions used to construct the dual and majax profiles on the images.
 
- save_encod.py : Contains the functions used to save vectors obtained from a given VAE run.

- save_flow_class.py : Class function to save flow model.

- score_gen_utils.py : Computes the generation score of the images generated from the Flow.

- train_utils.py : Functions used to train the models. For the VAE : selection of the wanted loss function (see loss_utils.py, train_vae.py). 

- umap_utils.py : Computes the UMAP of the latent space.
	
## III-/ Config :

Configurations (YML) files used to define the hyperparameters associated with the different parts of the model. Input data is provided in /data to test the model to understand its different operations, and trained weights are provided in /checkpoints. 

## IV-/ Checkpoints :

Trained weights of the VAE and the normalising flow.

For the VAE, these weights corresponds to the configuration (necessary in test_vae.py, train_flow.py, test_flow.py, test_gen.py and test_gen.ipynb):

- type_loss = 'Fourier'

- num_run = 220 

For the Flow, the weigths corresponds to the configuration (necessary in test_flow.py, test_gen.py, test_gen.ipynb):

- n_run = 55

## V-/ Data :

Input data for the VAE and Flow, stored as pickle files. The input images are labelled as "objet_n.pkl", whereas the VAE latent space vectors are labelled "encoding_n.pkl".
We also provides the latent space vectors as a list under the "encod_list.pkl" file, as well as the reconstruction scores of the Horizon-AGN images (see Mahe et al, 2026), to compute the generation scores in test_gen.py and test_gen.ipynb.

-------------------------------------------------------------

## Environment :
See also the env.yml file for a direct creation witch conda.

astropy                   6.1.3           py310h5eee18b_0  
corner                    2.2.2                    pypi_0    pypi
keras                     2.15.0                   pypi_0    pypi
matplotlib                3.9.2           py310h06a4308_0  
neptune                   1.12.0                   pypi_0    pypi
numpy                     1.26.4          py310h5f9d8c6_0  
pandas                    2.2.3                    pypi_0    pypi
pip                       24.2            py310h06a4308_0  
python                    3.10.15              he870216_1  
scikit-image              0.25.0          py310h6a678d5_0  
scikit-learn              1.6.1           py310h6a678d5_0  
scipy                     1.12.0          py310h5f9d8c6_0  
tensorboard               2.15.2                   pypi_0    pypi
tensorflow                2.15.0.post1             pypi_0    pypi
tensorflow-estimator      2.15.0                   pypi_0    pypi
tensorflow-probability    0.23.0                   pypi_0    pypi 
tqdm                      4.66.5                   pypi_0    pypi
umap-learn                0.5.7           py310hff52083_1    conda-forge

-------------------------------------------------------------

This is the README for the FVAE Project about Dual AGNs
