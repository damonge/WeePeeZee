# WeePeeZee

## What this contains so far
- The `data` folder contains subfolders that contain `.sacc` files with the power spectra and redshift distributions we used for the HSC paper. You can pretty much ignore all of them except for the `data/COADD` folder, which contains the data coadded over all fields.
- The `modules` folder contains a bunch of code to compute theory predictions. Most of this is stolen from `LSSLike`. In particular, if you want to understand what's currently in `LSSLike` (although in a much more lightweight form, look at `modules/theory_cls.py`. The script `theory_example.py` shows how to compute the theoretical prediction for the angular power spectra given the inputs in a SACC file.
- The script `data_generator.py` can create a simulated SACC file that contains
  - a) N(z)s made up of a smooth mean + some Gaussian fluctuations.
  - b) Power spectra made up of the true power spectra (corresponding to the true underlying parameters and smooth N(z)) + noise coming from the covariance matrix (i.e. a Gaussian realization of the power spectra).
  
  The script will save these files into a folder within the `data` folder with whatever name you want. To run it, type
  ```
  python data_generator.py <sim_name> <save_mean> <n_svd>
  ```
  where `sim_name` is the name of the folder where you want to save the results, `save_mean` is either 0 or 1, and governs whether you want to save an additional SACC file with the true power spectra and N(z)s, and `n_svd` is the number of principal eigenvalues you want to use when generating the random N(z)s. For now **set that to 4**.
