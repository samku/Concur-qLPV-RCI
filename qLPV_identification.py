from imports import *

def qLPV_identification(dataset, sizes, kappa, only_px, id_params, plot_results = False, use_init_LTI = True, only_SysID = False):
    
    #Extract data
    Y_train = dataset['Y_train']
    Y_test = dataset['Y_test']
    Ys_train = dataset['Ys_train']
    Us_train = dataset['Us_train']
    Ys_test = dataset['Ys_test']
    Us_test = dataset['Us_test']
    N_train = Ys_train.shape[0] 
    ny = Ys_train.shape[1]
    nu = Us_train.shape[1]
    nx = sizes[0]
    nq = sizes[1]
    nth = sizes[2]
    nH = sizes[3]
    constraints = dataset['constraints']    

    # Optimization params
    iprint = id_params['iprint']
    memory = id_params['memory']
    eta = id_params['eta']
    rho_th = id_params['rho_th']
    adam_epochs = id_params['adam_epochs']
    lbfgs_epochs = id_params['lbfgs_epochs']
    train_x0 = id_params['train_x0']
    weight_RCI = id_params['weight_RCI']
    N_MPC = id_params['N_MPC']
    kappa_p = id_params['kappa_p']
    kappa_x = id_params['kappa_x']

    jax.config.update("jax_enable_x64", True)

    #Extract sizes
    print('Identifying model with nx:', nx, 'nq:', nq, 'nth:', nth, 'nH:', nH)

    if use_init_LTI:
        from LTI_identification import initial_LTI_identification  
        model_LTI, RCI_LTI = initial_LTI_identification(Ys_train, Us_train, nx, constraints, kappa, N_MPC, only_SysID = only_SysID)

        #Do system identification inside the RCI set
        #Extract LTI data
        A_LTI = model_LTI['A']
        B_LTI = model_LTI['B']
        C_LTI = model_LTI['C']
        L_LTI = model_LTI['L']

        if not only_SysID:
            W_LTI = RCI_LTI['W']
            w0 = W_LTI[0]
            epsw = W_LTI[1]

            #Constraints
            F = RCI_LTI['F']
            yRCI = RCI_LTI['yRCI']
            uRCI = RCI_LTI['uRCI']
            V = RCI_LTI['V']
            m = F.shape[0]
            m_bar = len(V)
            E = RCI_LTI['E']
            HY = constraints['HY']
            hY = constraints['hY']
            hY_tight = hY - (HY @ w0 + kappa*np.abs(HY) @ epsw)
            mY = HY.shape[0]
            Y_set = polytope.Polytope(HY, hY)
            Y_vert = polytope.extreme(Y_set)
            vY = len(Y_vert)
            HU = constraints['HU']
            hU = RCI_LTI['hU_modified']
            mU = HU.shape[0]

            #Build inequalities
            F_Vjy_uj = np.zeros((m*m_bar, (nx**2)+nx*nu))
            H_Vjy = np.zeros((mY*m_bar, ny*nx))
            rhs_1 = np.kron(np.ones(m_bar),yRCI)
            rhs_2 = np.kron(np.ones(m_bar),hY_tight)
            for j in range(m_bar):
                Fy_matrix = F@np.kron(np.eye(nx),V[j]@yRCI)
                Fu_matrix = F@np.kron(np.eye(nx),uRCI[:,j].T)
                F_Vjy_uj[j*m:(j+1)*m,:] = np.hstack((Fy_matrix,Fu_matrix))
                H_Vjy[j*mY:(j+1)*mY,:] = HY@np.kron(np.eye(ny),V[j]@yRCI)
            A_ineq = np.kron(np.eye(nq),F_Vjy_uj)
            b_ineq = np.kron(np.ones(nq),rhs_1)
            A_ineq = block_diag(A_ineq, H_Vjy)
            b_ineq = np.hstack((b_ineq, rhs_2))

            num_ineq = A_ineq.shape[0]
   
    # Define the optimization variables
    key = jax.random.PRNGKey(10)
    key1, key2, key3 = jax.random.split(key, num=3)
    A = 0.0001*jax.random.normal(key1, (nq,nx,nx))
    B = 0.0001*jax.random.normal(key2, (nq,nx,nu))
    C = 0.0001*jax.random.normal(key2, (ny,nx))
    if only_px == 1:
        Win = 0.01*jax.random.normal(key1, (nq-1, nth, nx))
    else:
        Win = 0.01*jax.random.normal(key1, (nq-1, nth, nx+nu))
    bin = 0.01*jax.random.normal(key2, (nq-1,nth))
    Whid = 0.01*jax.random.normal(key3, (nq-1, nH-1, nth, nth))
    bhid = 0.01*jax.random.normal(key1, (nq-1, nH-1, nth))
    Wout = 0.01*jax.random.normal(key2, (nq-1, nth))
    bout = 0.01*jax.random.normal(key3, (nq-1, 1))

    if use_init_LTI:
        #Initialize model with LTI
        for i in range(nq):
            A = A.at[i].set(jnp.array(A_LTI))
            B = B.at[i].set(jnp.array(B_LTI))
            C = jnp.array(C_LTI)
    
        if not only_SysID:
            #Build functions
            @jax.jit
            def RCI_constraints(A,B,C):
                AB_vectorized = jnp.zeros((nq*(nx**2+nx*nu),))
                for i in range(nq):
                    AB_vectorized = AB_vectorized.at[i*(nx**2+nx*nu):(i+1)*(nx**2+nx*nu)].set(jnp.hstack((A[i].flatten(),B[i].flatten())))
                C_vectorized = C.flatten()  
                ABC_vector = jnp.hstack((AB_vectorized,C_vectorized))
                return A_ineq @ ABC_vector - b_ineq
    
    @jax.jit
    def parameter_fcn(x,u,Win,bin,Whid,bhid,Wout,bout):
        if only_px==0:
            z = jnp.hstack((x,u))
        else:
            z = x
        p = jnp.array([[1.]])
        for i in range(nq-1):
            post_linear = Win[i]@z+bin[i]
            post_activation = nn.swish(post_linear)
            for j in range(nH-1): 
                post_linear = Whid[i][j]@post_activation+bhid[i][j]
                post_activation = nn.swish(post_linear)
            post_linear = Wout[i]@post_activation+bout[i]
            p = jnp.vstack((p,jnp.exp(post_linear)))
        p = p/jnp.sum(p)
        return p
    
    @jax.jit
    def state_fcn(x,u,params):
        A, B, C, Win, bin, Whid, bhid, Wout, bout = params
        p = parameter_fcn(x,u,Win,bin,Whid,bhid,Wout,bout)
        A_tot = A[0]*p[0]
        B_tot = B[0]*p[0]
        for i in range(1,nq):
            A_tot += A[i]*p[i]
            B_tot += B[i]*p[i]
        return A_tot@x+B_tot@u

    @jax.jit
    def output_fcn(x,u,params):
        A, B, C, Win, bin, Whid, bhid, Wout, bout = params
        return C@x

    @jax.jit
    def SS_forward_output(x, u, params):
        A, B, C, Win, bin, Whid, bhid, Wout, bout = params
        y_current = output_fcn(x, u, params)
        x_next = state_fcn(x, u, params).reshape(-1)
        return x_next, y_current
    
    @jax.jit
    def custom_regularization(params,x0): 
        custom_R = 0.
        A, B, C, Win, bin, Whid, bhid, Wout, bout = params

        if kappa_p>0.:
            gp_lasso = 0.
            for i in range(nq-1):
                gp_lasso += tau_p*jnp.linalg.norm(Wout[i], ord=2)
            custom_R = gp_lasso

        if kappa_x>0.:
            gx_lasso = 0.
            for i in range(nq):
                for j in range(nx):
                    A_comp_row = A[i][j,:]
                    gx_lasso += kappa_x*jnp.linalg.norm(A_comp_row, ord=2)
                    A_comp_col = A[i][:,j]
                    gx_lasso += kappa_x*jnp.linalg.norm(A_comp_col, ord=2)
                    B_comp_row = B[i][j,:]
                    gx_lasso += kappa_x*jnp.linalg.norm(B_comp_row, ord=2)
                    if i<nq-1:
                        Wout_comp_col = Win[i][:,j]
                        gx_lasso += kappa_x*jnp.linalg.norm(Wout_comp_col, ord=2) 
                C_comp_col = C[:,j]
                gx_lasso += kappa_x*jnp.linalg.norm(C_comp_col, ord=2)
            custom_R += gx_lasso

        if use_init_LTI and not only_SysID:
            #Ensure error bound are respected
            f_sim = partial(SS_forward_output, params=params)
            y_hat = jax.lax.scan(f_sim, jnp.zeros((nx,)), Us_train)[1]
            y_error = Ys_train-y_hat
            W_error = jnp.sum((jnp.maximum(0,jnp.abs(y_error-w0)-epsw))**2)
            #Keep RCI valid
            AB_vectorized = jnp.zeros((nq*(nx**2+nx*nu),))
            for i in range(nq):
                AB_vectorized = AB_vectorized.at[i*(nx**2+nx*nu):(i+1)*(nx**2+nx*nu)].set(jnp.hstack((A[i].flatten(),B[i].flatten())))
            C_vectorized = C.flatten()  
            ABC_vector = jnp.hstack((AB_vectorized,C_vectorized))
            RCI_violation = A_ineq @ ABC_vector - b_ineq
            custom_R = custom_R+1*W_error+weight_RCI*jnp.sum((jnp.maximum(0,RCI_violation))**2)

        return custom_R
    
    #Pass to optimizer
    model = Model(nx, ny, nu, state_fcn=state_fcn, output_fcn=output_fcn, Ts=1)
    model.init(params=[A, B, C, Win, bin, Whid, bhid, Wout, bout])
    model.loss(rho_th=rho_th, train_x0 = train_x0, custom_regularization=custom_regularization)  
    model.optimization(adam_eta=eta, adam_epochs=adam_epochs, lbfgs_epochs=lbfgs_epochs, memory=memory, iprint=iprint)
    model.fit(Ys_train, Us_train)
    identified_params = model.params
    A_new = np.array(model.params[0])
    B_new = np.array(model.params[1])
    C_new = np.array(model.params[2])
    Win_new = np.array(model.params[3])
    bin_new = np.array(model.params[4])
    Whid_new = np.array(model.params[5])
    bhid_new = np.array(model.params[6])
    Wout_new = np.array(model.params[7])
    bout_new = np.array(model.params[8])
    model_LPV = {'A': A_new, 'B': B_new, 'C': C_new, 'Win': Win_new, 'bin': bin_new, 'Whid': Whid_new, 'bhid': bhid_new, 'Wout': Wout_new, 'bout': bout_new}
    
    x0_train = model.learn_x0(Us_train, Ys_train, LBFGS_refinement=True, verbosity=0) 
    #Evaluate BFR - Train
    params_orig = [A, B, C, Win, bin, Whid, bhid, Wout, bout]
    f_sim = partial(SS_forward_output, params=params_orig)
    y_hat_orig = jax.lax.scan(f_sim, jnp.zeros((nx,)), Us_train)[1]
    f_sim = partial(SS_forward_output, params=identified_params)
    y_hat = jax.lax.scan(f_sim, x0_train, Us_train)[1]
    BFR_orig = np.maximum(0, 1 - np.linalg.norm(Ys_train - y_hat_orig, ord=2) / np.linalg.norm(Ys_train - np.mean(Ys_train), ord=2))
    BFR_new = np.maximum(0, 1 - np.linalg.norm(Ys_train - y_hat, ord=2) / np.linalg.norm(Ys_train - np.mean(Ys_train), ord=2))
    print('BFR: Orig (LTI)', BFR_orig, 'New (LPV init)', BFR_new)
    y_hat_orig_unscaled = dataset['y_scaler'][0]+y_hat_orig/dataset['y_scaler'][1]
    y_hat_unscaled = dataset['y_scaler'][0]+y_hat/dataset['y_scaler'][1]
    BFR_orig_unscaled = np.maximum(0, 1 - np.linalg.norm(Y_train - y_hat_orig_unscaled, ord=2) / np.linalg.norm(Y_train - np.mean(Y_train), ord=2))
    BFR_new_unscaled = np.maximum(0, 1 - np.linalg.norm(Y_train - y_hat_unscaled, ord=2) / np.linalg.norm(Y_train - np.mean(Y_train), ord=2))
    print('BFR unscaled: Orig (LTI)', BFR_orig_unscaled, 'New (LPV init)', BFR_new_unscaled)

    plt.plot(Y_train)
    plt.plot(y_hat_unscaled,'--')
    plt.show()

    print('Wout:', Wout_new)

    #x0_test = model.learn_x0(Us_test, Ys_test, LBFGS_refinement=True, verbosity=0) 
    x0_test = jnp.zeros((nx,))
    #Evaluate BFR - Test
    f_sim = partial(SS_forward_output, params=params_orig)
    yt_hat_orig = jax.lax.scan(f_sim, jnp.zeros((nx,)), Us_test)[1]
    f_sim = partial(SS_forward_output, params=identified_params)
    yt_hat = jax.lax.scan(f_sim, x0_test, Us_test)[1]
    BFR_orig = np.maximum(0, 1 - np.linalg.norm(Ys_test - yt_hat_orig, ord=2) / np.linalg.norm(Ys_test - np.mean(Ys_test), ord=2))
    BFR_new = np.maximum(0, 1 - np.linalg.norm(Ys_test - yt_hat, ord=2) / np.linalg.norm(Ys_test - np.mean(Ys_test), ord=2))
    print('BFR (test): Orig (LTI)', BFR_orig, 'New (LPV init)', BFR_new)
    yt_hat_orig_unscaled = dataset['y_scaler'][0]+yt_hat_orig/dataset['y_scaler'][1]
    yt_hat_unscaled = dataset['y_scaler'][0]+yt_hat/dataset['y_scaler'][1]
    BFR_orig_unscaled = np.maximum(0, 1 - np.linalg.norm(Y_test - yt_hat_orig_unscaled, ord=2) / np.linalg.norm(Y_test - np.mean(Y_test), ord=2))
    BFR_new_unscaled = np.maximum(0, 1 - np.linalg.norm(Y_test - yt_hat_unscaled, ord=2) / np.linalg.norm(Y_test - np.mean(Y_test), ord=2))
    print('BFR unscaled (test) : Orig (LTI)', BFR_orig_unscaled, 'New (LPV init)', BFR_new_unscaled)

    @jax.jit
    def SS_state_sim(x, u, params):
        A, B, C, Win, bin, Whid, bhid, Wout, bout = params
        x_next = state_fcn(x, u, params).reshape(-1)
        return x_next, x_next

    @jax.jit
    def SS_parameter_sim(carry, xu, params):
        A, B, C, Win, bin, Whid, bhid, Wout, bout = params
        scheduling_param = parameter_fcn(xu[:nx],xu[nx:],Win,bin,Whid,bhid,Wout,bout)
        return carry, scheduling_param.reshape(-1)  
    
    f_sim = partial(SS_state_sim, params=identified_params)
    X_train = jax.lax.scan(f_sim, jnp.zeros((nx,)), Us_train)[1]
    XU_train = jnp.hstack((X_train,Us_train))
    f_sim = partial(SS_parameter_sim, params=identified_params)
    p_traj = jax.lax.scan(f_sim, 0., XU_train)[1]
    
    print('Wout:', Wout_new)
    plt.plot(p_traj)
    plt.show()

    print('A_LTI:', A[0])
    print('A_LPV_init:', A_new)
    print('B_LTI:', B[0])
    print('B_LPV_init:', B_new)
    print('C_LTI:', C)
    print('C_LPV_init:', C_new)




    if not only_SysID:
        RCI_vector =RCI_constraints(A_new,B_new,C_new)
        plt.plot(RCI_vector)
        print(np.max(RCI_vector))
        plt.show()

        if plot_results:
            fig, (ax1,ax2) = plt.subplots(1,2)
            ax1.plot(Y_train)
            ax1.plot(y_hat_unscaled,'--')
            ax2.plot(Y_test)
            ax2.plot(yt_hat_unscaled,'--')
            plt.show()

        print('A_LTI:', A[0])
        print('A_LPV_init:', A_new)

        if use_init_LTI:
            #Verify invariance
            X_set = polytope.Polytope(F, yRCI)
            figure, (ax1, ax2) = plt.subplots(1, 2)
            X_set.project([1,2]).plot(ax=ax1, color='k', alpha=0.2)
            for i in range(nq):
                RCI_plus_vert = np.zeros((m_bar,nx))
                for j in range(m_bar):
                    RCI_plus_vert[j] = A_new[i]@V[j]@yRCI+B_new[i]@uRCI[:,j]
                ax1.scatter(RCI_plus_vert[:,0], RCI_plus_vert[:,1], c='r')
            
            y_error = Ys_train-y_hat
            y_error_orig = Ys_train-y_hat_orig
            for i in range(ny):
                color_rand = np.random.rand(3,)
                ax2.plot(y_error[:,i], color=color_rand)
                ax2.plot(y_error_orig[:,i], color=color_rand, linestyle='--')   
                ax2.plot(np.ones(N_train)*(w0[i]+epsw[i]), color=color_rand, linestyle='--')
                ax2.plot(np.ones(N_train)*(w0[i]-epsw[i]), color=color_rand, linestyle='--')
            
            plt.show()

        #Recompute RCI set
        RCI_LPV = RCI_LTI.copy()
        opti = ca.Opti()
        w_hat_LPV = opti.variable(ny)
        eps_w_LPV = opti.variable(ny)
        for i in range(N_train):
            opti.subject_to(Ys_train[i]<=y_hat[i]+w_hat_LPV+0.9*eps_w_LPV)
            opti.subject_to(Ys_train[i]>=y_hat[i]+w_hat_LPV-0.9*eps_w_LPV)
        opti.subject_to(eps_w_LPV>=0)
        opti.minimize(ca.dot(eps_w_LPV,eps_w_LPV))
        opti.solver('ipopt')
        sol = opti.solve()
        w_hat_LPV = np.array([sol.value(w_hat_LPV)])
        eps_w_LPV = np.array([sol.value(eps_w_LPV)])
        W_LPV = np.vstack((w_hat_LPV,eps_w_LPV))

        print('W_LPV:', W_LPV)  
        print('W_LTI:', RCI_LTI['W'])

        #Recompute RCI set
        opti = ca.Opti()
        yLPV = opti.variable(m)
        uLPV = opti.variable(nu,m_bar)
        bLPV = opti.variable(nx,vY)
        opti.subject_to(E@yLPV<=0)
        for i in range(m_bar):
            for j in range(nq):
                vector = F@(A_new[j]@V[i]@yLPV+B_new[j]@uLPV[:,i])-yLPV
                opti.subject_to(vector<=0)
            opti.subject_to(HU@uLPV[:,i]<=hU)
            opti.subject_to(HY@C_new@V[i]@yLPV+HY@w_hat_LPV+kappa*np.abs(HY)@eps_w_LPV<=hY)
        A_mean = np.mean(A_new, axis = 0)
        B_mean = np.mean(B_new, axis = 0)
        cost = 0.
        x_traj = opti.variable(nx*vY, N_MPC+1)
        u_traj = opti.variable(nu*vY, N_MPC)
        for i in range(vY):
            x_traj_loc = x_traj[nx*i:nx*(i+1),:]
            u_traj_loc = u_traj[nu*i:nu*(i+1),:]
            opti.subject_to(x_traj_loc[:,0]==np.zeros((nx)))
            for t in range(N_MPC):
                opti.subject_to(x_traj_loc[:,t+1]==A_mean@x_traj_loc[:,t]+B_mean@u_traj_loc[:,t])
                opti.subject_to(F@x_traj_loc[:,t]<=yLPV)
                opti.subject_to(HU@u_traj_loc[:,t]<=hU)
                vector = C_new@x_traj_loc[:,t]-Y_vert[i]
                cost = cost+ca.dot(vector,vector)
            opti.subject_to(F@x_traj_loc[:,N_MPC]<=yLPV)
            vector = C_new@x_traj_loc[:,N_MPC]-Y_vert[i]
            cost = cost+ca.dot(vector,vector)
        
        opti.minimize(cost)
        opti.solver('ipopt')
        sol = opti.solve()
        yLPV = sol.value(yLPV)
        uLPV = sol.value(uLPV)
        costLPV = sol.value(cost)
        x_traj = sol.value(x_traj)
        u_traj = sol.value(u_traj)

        fig, (ax1, ax2) = plt.subplots(2,1)
        polytope.Polytope(F,yLPV).project([1,2]).plot(ax=ax1)
        for i in range(vY):
            x_traj_loc = x_traj[nx*i:nx*(i+1),:]
            ax1.plot(x_traj_loc[0,:],x_traj_loc[1,:],'r')
            y_loc = C_new@x_traj_loc
            ax2.plot(y_loc[0])
            ax2.plot(np.ones(N_MPC)*Y_vert[i],'k--')
        ax1.autoscale()
        plt.show()

        RCI_LPV['yRCI'] = yLPV
        RCI_LPV['uRCI'] = uLPV
        RCI_LPV['x_traj'] = x_traj
        RCI_LPV['u_traj'] = u_traj
        RCI_LPV['W'] = W_LPV
        RCI_LPV['cost'] = costLPV
    else:
        RCI_LPV = {}

    return model_LPV, model_LTI, RCI_LPV, RCI_LTI



        



