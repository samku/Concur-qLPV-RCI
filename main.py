from imports import *
current_directory = Path(__file__).parent
from utils.generate_file_path import generate_file_path
from data_generation import generate_dataset
from qLPV_identification import qLPV_identification
from concurrent_identification import concurrent_identification


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
file_name = 'initial_models'
overwrite_data = False
file_path = generate_file_path(folder_name, file_name, current_directory)
if not file_path.exists() or overwrite_data:
    only_SysID = False
    use_init_LTI = True
    nx = 2 #State dimension
    nq = 3 #Number of parameters
    nth = 3 #Number of neurons in each layer
    nH = 1 #Number of activation layers
    sizes = (nx, nq, nth, nH)
    only_px = 0 #0 if p(x,u), 1 if p(x) for scheduling function
    kappa = 1.1 #Disturbance set inflation factor
    N_MPC = 50 #Horizon length for RCI set size factor
    id_params = {'eta': 0.001, 'rho_th': 0.000, 'adam_epochs': 2000, 'lbfgs_epochs': 5000, 'iprint': 100, 'memory': 10,
                'train_x0': True, 'weight_RCI':100, 'N_MPC': N_MPC, 'kappa_p': 0, 'kappa_x': 0.}
    model_LPV, model_LTI, RCI_LPV, RCI_LTI = qLPV_identification(dataset, sizes, kappa, only_px, id_params, 
                                                                use_init_LTI = use_init_LTI, only_SysID = only_SysID)
    models = {}
    models['model_LPV'] = model_LPV
    models['model_LTI'] = model_LTI
    models['RCI_LPV'] = RCI_LPV
    models['RCI_LTI'] = RCI_LTI
    models['sizes'] = sizes
    models['only_px'] = only_px
    models['kappa'] = kappa
    models['N_MPC'] = N_MPC

    #Save dataset
    with open(file_path, 'wb') as f:
        pickle.dump(models, f)
    print('Initial models saved to ', file_path)
else:
    with open(file_path, 'rb') as f:
        models = pickle.load(f)

#Concurrent identification
file_name = 'concurrent_SysID'
overwrite_data = True
file_path = generate_file_path(folder_name, file_name, current_directory)
if not file_path.exists() or overwrite_data:
    model_LPV = models['model_LPV']
    model_LTI = models['model_LTI']
    RCI_LPV = models['RCI_LPV']
    RCI_LTI = models['RCI_LTI']
    sizes = models['sizes']
    only_px = models['only_px']
    kappa = models['kappa']
    N_MPC = models['N_MPC']

    id_params = {'eta': 0.001, 'rho_th': 0.001, 'adam_epochs': 3000, 'lbfgs_epochs': 0, 
                'iprint': 100, 'memory': 100, 'train_x0': True, 'weight_RCI': 0.0001, 'N_MPC': N_MPC, 'QP_form': 0, 'regular': 0.0}
    model_LPV_concur, RCI_concur = concurrent_identification(dataset, model_LPV, RCI_LPV, sizes, only_px, kappa, id_params)

    models['model_LPV_concur'] = model_LPV_concur
    models['RCI_concur'] = RCI_concur

    #Save dataset
    with open(file_path, 'wb') as f:
        pickle.dump(models, f)
    print('Concurrent identification saved to ', file_path)
else:
    with open(file_path, 'rb') as f:
        models = pickle.load(f)

