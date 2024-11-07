from imports import *
from data_generation import generate_dataset

#Generate/Load dataset
from plant_models.nonlinear_MSD import nonlinear_MSD
system = nonlinear_MSD(initial_state=np.zeros(10))
scale_data = True
N_train = 20000
N_test = 5000
np.random.seed(21)
folder_name = 'MassSpringDamper_chain'
file_name = 'dataset'
overwrite_data = False
dataset = generate_dataset(system, N_train, N_test, scale_data, folder_name, file_name, overwrite_data) 

#Initialization
from qLPV_identification import qLPV_identification
nx = 4
nq = 3 #Number of parameters
nth = 3 #Number of neurons in each layer
nH = 1 #Number of activation layers
sizes = (nx, nq, nth, nH)
only_px = 0 #0 if p(x,u), 1 if p(x) for scheduling function
kappa = 1.1 #Disturbance set inflation factor
N_MPC = 50 #Horizon length for RCI set size factor
id_params = {'eta': 0.001, 'rho_th': 0.000, 'adam_epochs': 1000, 'lbfgs_epochs': 5000, 'iprint': 100, 'memory': 10,
            'train_x0': False, 'weight_RCI':2., 'N_MPC': N_MPC, 'kappa_p': 0, 'kappa_x': 0.}
model_LPV, model_LTI, RCI_LPV, RCI_LTI = qLPV_identification(dataset, sizes, kappa, only_px, id_params, plot_results = False, use_init_LTI = True, only_SysID = False)






