# Combined Learning of Linear Parameter-Varying Models and Robust Control Invariant Sets
This repository contains the code for the paper "Combined Learning of Linear Parameter-Varying Models and Robust Control Invariant Sets" by S.K.Mulagaleti and A.Bemporad (2024). Preprint available at [arXiv:2411.18166](https://arxiv.org/pdf/2411.18166).

## Code structure
The main file to run is 'main.py'. This file consists of the following three main parts:
1. Generate/Load dataset
2. qLPV identification
3. Concurrent identification

All three parts save results a pickle file in the folder 'identification_results/$folder_name$.
If the file already exists, the results are loaded.
To recompute the results, set overwrite_data = True in the function call.

In Step 1: 
- Select plant model and number of samples.
- The base class for plant models is in plant_models/plant_model.py
- Dataset generation is in data_generation.py
- Options:
    - scale_data: Boolean to scale the data
    - N_train: Number of training samples
    - N_test: Number of testing samples
    - overwrite_data: Boolean to overwrite the existing dataset or not
- By default, inputs are randomly sampled inside the constraints.
- Uncomment the line 'U_tot = generate_multisine(nu, N_train, 0.1, 100, -system.u_lim, system.u_lim, 10)' to generate multisine inputs.
- The multisine function is present in the folder 'utils'.

In Step 2:
- Quasi - LPV models of the form

$$x^+ = A(p(x,u))x + B(p(x,u))u, \ \ \  y = Cx$$

are identified using the function 'qLPV_identification.py'.
- Options:
    - Only_SysID: Boolean - True : Only system identification; False : Also identify control invariant set.
    - use_init_LTI: Boolean to initialize qLPV identification with LTI model
    - sizes: $$(nx,nq,nth,nH)=$$ (number of states, number of parameters, number of neurons in each layer, number of activation layers)
    - only_px: Boolean - If True, the scheduling function is p(x) else p(x,u)
    - kappa: Disturbance set inflation factor
    - N_MPC: Horizon length affecting size metric of control invariant set
    - id_params: Dictionary containing the following parameters:
        - eta: Learning rate for Adam optimizer
        - rho_th: Regularization parameter for thresholding
        - adam_epochs: Number of epochs for Adam optimizer
        - lbfgs_epochs: Number of epochs for LBFGS optimizer
        - iprint: Print frequency for Adam optimizer
        - memory: Memory for LBFGS optimizer
        - train_x0: Boolean - If True, the initial state is an optimization variable during training
        - weight_RCI: Weight for the RCI set computation 
        - N_MPC: Horizon length for RCI set size factor
        - kappa_p: Regularization parameter for scheduling order reduction
        - kappa_x: Regularization parameter for state order reduction

In Step 3:
- The function 'concurrent_identification.py' computes the qLPV model, with control-invariant set regularization. The set template was constructed in Step 2.

- This function has the following options in addition to the ones in Step 2:
    - QP_form: 0 for primal QP, 1 for dual QP
    - regular: Regularization parameter for control invariant set computation. Need $$>0$$ if dual form is used.


## Additional Notes
- The template matrix $\tilde{F}$ is selected in the function 'LTI_identification.py'. By default, the set $X=\{x:\tilde{F}x \leq 1\}$ is the $$\infty$$-norm ball. This can be changed by modifying the function.

- The function utils/qLPV_BFR.py computes the BFRs for the models. It reads a dictionary of the model parameters, the inputs and outputs, and option to use observer gain.

- The script controller_synthesis/integral_tracking_filter.py synthesizes and tests the safe tracking controller. It uses the model class in controller_synthesis/qLPV_model.py which is initialized using the parameters from Step 3. 

## Acknowledgements

This research activity was supported by the European Research Council (ERC), Advanced Research Grant COMPACT (Grant Agreement No. 101141351)

## Installation
This project requires the following Python packages:

- [jax_sysid](https://github.com/bemporad/jax-sysid)
- [flax](https://github.com/google/flax)
- [qpax](https://github.com/kevin-tracy/qpax)
- [casadi](https://web.casadi.org/)
- [pycvxset](https://github.com/merlresearch/pycvxset) 
- [control](https://python-control.readthedocs.io/)


You can install a part of the required packages using `pip`:

```bash
pip install jax-sysid flax qpax casadi control
```

- Please refer to https://github.com/merlresearch/pycvxset for installation of pycvxset. Functions from this package are used for polytope operations.
