from imports import *

def initial_LTI_identification(Ys_data, Us_data, nx, constraints, kappa, N_MPC, only_SysID = False):
    N = len(Ys_data)
    nu = Us_data.shape[1]
    ny = Ys_data.shape[1]
    Ys_data = Ys_data
    Us_data = Us_data
    model = LinearModel(nx, ny, nu, feedthrough=False)
    model.loss(rho_x0=0.0001, rho_th=0.001)
    model.optimization(adam_eta=0.001, adam_epochs=1000, lbfgs_epochs=1000)
    model.fit(Ys_data, Us_data)
    A, B, C, D = model.ssdata()

    #Simulate from origin
    X_sim = np.zeros((N,nx))
    Y_sim = np.zeros((N,ny))
    for i in range(N-1):
        X_sim[i+1] = A@X_sim[i]+B@Us_data[i]
        Y_sim[i] = C@X_sim[i]

    BFR_SS = np.maximum(0, 1-np.linalg.norm(Ys_data-Y_sim,2)/np.linalg.norm(Ys_data-np.mean(Y_sim),2))
    print('BFR_SS: ', BFR_SS)

    model_dict = {} 
    model_dict['A'] = A 
    model_dict['B'] = B
    model_dict['C'] = C
    model_dict['L'] = np.zeros((nx,ny))
    model_dict['BFR_SS'] = BFR_SS

    if not only_SysID:
        #Extract disturbances
        prediction_error = Ys_data-Y_sim
        max_prediction_error = np.max(prediction_error)
        min_prediction_error = np.min(prediction_error)
        w0 = 0.5*(max_prediction_error+min_prediction_error)
        epsw = 0.5*(max_prediction_error-min_prediction_error)
        if ny==1:
            w0 = np.array([w0])
            epsw = np.array([epsw])
        
        #Compute RCI set for LTI system
        HY = constraints['HY']
        hY = constraints['hY']
        HU = constraints['HU']
        hU = constraints['hU']
        hY_tight = hY - (HY @ w0 + kappa*np.abs(HY) @ epsw)
        Y_set = Polytope(A = HY, b = hY)
        Y_vert = Y_set.V
        mY = len(Y_vert)

        #For 2D, select number of facets of template
        #Similar to A One-step Approach to Computing a Polytopic Robust Positively Invariant Set, P.Trodden
        m = 4
        if nx == 2:
            angle = 2*np.pi/m
            F = np.array([]).reshape(0, nx)
            for i in range(m):
                row_add = [[np.cos(i*angle), np.sin((i)*angle)]]
                F = np.concatenate((F, row_add), axis=0)
        else:
            F = np.vstack((np.eye(nx), -np.eye(nx)))
        
        m = F.shape[0]
        y0 = np.ones(m)
        X_template = Polytope(A = F, b = y0)
        Xt_vert = X_template.V
        m_bar = len(Xt_vert)
        print('Identifying RCI set with ', m, ' faces and ', m_bar, ' vertices')
        
        opti = ca.Opti()
        WM_init = opti.variable(nx,nx)
        WMinv_init = ca.inv(WM_init)
        opti.set_initial(WM_init, np.eye(nx))
        uRCI_init = opti.variable(nu, m_bar)
        u_bound_mod = opti.variable(nu)
        du_bound = hU[:nu] - u_bound_mod
        cost_bound = ca.dot(du_bound, du_bound)
        hU_modified = ca.vertcat(u_bound_mod, u_bound_mod)

        for k in range(m_bar):
            vector = F @ (WMinv_init @ (A @ WM_init @ Xt_vert[k] + B @ uRCI_init[:,k])) - y0
            opti.subject_to(vector <= 0)
            vector = HY @ C @ WM_init @ Xt_vert[k] - hY_tight
            opti.subject_to(vector <= 0)
            vector = HU @ uRCI_init[:,k] - hU_modified
            opti.subject_to(vector <= 0)

        #Size
        cost_traj = 0.
        x_traj = opti.variable(nx*mY, N_MPC+1)
        u_traj = opti.variable(nu*mY, N_MPC)
        for i in range(mY):
            x_traj_loc = x_traj[nx*i:nx*(i+1),:]
            u_traj_loc = u_traj[nu*i:nu*(i+1),:]
            opti.subject_to(x_traj_loc[:,0]==np.zeros((nx)))
            for t in range(N_MPC):
                opti.subject_to(x_traj_loc[:,t+1]==A@x_traj_loc[:,t]+B@u_traj_loc[:,t])
                opti.subject_to(F@WMinv_init@x_traj_loc[:,t]<=y0)
                opti.subject_to(HU@u_traj_loc[:,t]<=hU_modified)
                vector = C@x_traj_loc[:,t]-Y_vert[i]
                cost_traj = cost_traj+ca.dot(vector,vector)
            opti.subject_to(F@WMinv_init@x_traj_loc[:,N_MPC]<=y0)
            vector = C@x_traj_loc[:,N_MPC]-Y_vert[i]
            cost_traj = cost_traj+ca.dot(vector,vector)

        opti.minimize(cost_traj+1000*cost_bound)
        opti.solver('ipopt')
        sol = opti.solve()
        WM = sol.value(WM_init)
        WMinv = sol.value(WMinv_init)
        uRCI = sol.value(uRCI_init)
        x_traj = sol.value(x_traj)
        if nu==1:
            uRCI = np.reshape(uRCI, (1, m_bar))
        hU_modified = sol.value(hU_modified)
        cost = sol.value(cost_traj)

        print('hU_modified: ', hU_modified)
        print('hU: ', hU)

        fig, ax2 = plt.subplots()
        for i in range(mY):
            x_traj_loc = x_traj[nx*i:nx*(i+1),:]
            y_loc = C@x_traj_loc
            ax2.plot(y_loc[0])
            ax2.plot(np.ones(N_MPC)*Y_vert[i],'k--')
        ax2.autoscale()
        plt.show()

        X_RCI_vertices = np.zeros((m_bar,nx))
        Y_RCI_vertices = np.zeros((m_bar,ny))
        for i in range(m_bar):
            X_RCI_vertices[i] = WM @ Xt_vert[i]
            Y_RCI_vertices[i] = C @ WM @ Xt_vert[i]  
        
        
        fig, (ax1, ax2) = plt.subplots(2,1)
        RCI_set = Polytope(A = F @ WMinv, b = y0)
        RCI_set_proj = RCI_set.projection(project_away_dim = [2,3])
        RCI_set_proj.plot(ax = ax1, patch_args={"facecolor": "m"})
        ax1.scatter(X_RCI_vertices[:,0], X_RCI_vertices[:,1], color='r')
        ax1.autoscale()
        if ny>=2:
            Y_set.project([1,2]).plot(ax2, color='b', alpha=0.5)
            ax2.scatter(Y_RCI_vertices[:,0], Y_RCI_vertices[:,1], color='r')
            ax2.autoscale()
            plt.show()
        else:
            print('Vertices: ', np.array([np.max(Y_RCI_vertices), np.min(Y_RCI_vertices)]))
            print('Y constraints: ', Y_vert)
        plt.show()

        #Construct CC set
        F = F @ WMinv
        
        V = [] #Vertex maps
        all_combinations = list(combinations(range(m), nx))
        for i in range(m_bar): #Keep index of input and state the same
            for j in range(len(all_combinations)):
                id_loc = np.array(all_combinations[j])
                V_test = F[id_loc]
                h_test = y0[id_loc]
                if np.linalg.norm(V_test @ WM @ Xt_vert[i] - h_test , np.inf)<=1e-5:
                    ones_mat = np.zeros((nx,m))
                    for k in range(nx):
                        ones_mat[k, id_loc[k]] = 1
                    V.append(np.linalg.inv(V_test) @ ones_mat)

        E = np.array([]).reshape(0, m) #Configuration constraints
        for k in range(m_bar):
            local_mat = F @ V[k] - np.eye(m)
            for j in range(m):
                E = np.concatenate((E, [local_mat[j]]), axis = 0)
        E = E[~np.all(E == 0, axis=1)]

        RCI_dict = {'F': F, 'V': V, 'E': E, 'yRCI': y0, 'uRCI': uRCI, 'x_traj': x_traj, 'hU_modified':hU_modified,
                    'cost': sol.value(cost), 'Y_RCI_vertices':Y_RCI_vertices, 'W': np.vstack((w0, epsw))}
        
    else:
        RCI_dict = {}

    return model_dict, RCI_dict