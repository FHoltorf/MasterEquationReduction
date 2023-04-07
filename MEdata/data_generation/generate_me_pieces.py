import numpy as np 

def generate_full_master_equation_LTI(network, symmetrized = False):
    temperature = network.T
    pressure = network.P  # not used in this module, pressure dependence implicitly considered in equilibrium densities.
    e_list = network.e_list
    j_list = network.j_list
    dens_states = network.dens_states
    m_coll = network.Mcoll
    k_ij = network.Kij
    f_im = network.Fim # reactant to isomer
    g_nj = network.Gnj # isomer to product/reactant
    n_isom = network.n_isom
    n_reac = network.n_reac
    n_prod = network.n_prod
    n_grains = network.n_grains
    n_j = network.n_j
    eq_ratios = network.eq_ratios 
    
    beta = 1. / (8.314 * temperature)

    indices = -np.ones((n_isom, n_grains, n_j), np.int)
    n_rows = 0
    for r in range(n_grains):
        for s in range(n_j):
            for i in range(n_isom):
                if dens_states[i, r, s] > 0:
                    indices[i, r, s] = n_rows
                    n_rows += 1

    # Construct full ME matrix
    me_mat = np.zeros([n_rows, n_rows], np.float64)
    b_mat = np.zeros([n_rows, n_reac], np.float64) # input matrix
    k_mat = np.zeros([n_prod, n_rows], np.float64) # product formation

    # Collision terms
    for i in range(n_isom):
        for r in range(n_grains):
            for s in range(n_j):
                if indices[i, r, s] > -1:
                    for u in range(r, n_grains):
                        for v in range(s, n_j):
                            me_mat[indices[i, r, s], indices[i, u, v]] = m_coll[i, r, s, u, v]
                            me_mat[indices[i, u, v], indices[i, r, s]] = m_coll[i, u, v, r, s]

    # Isomerization terms
    for i in range(n_isom):
        for j in range(i):
            if k_ij[i, j, n_grains - 1, 0] > 0 or k_ij[j, i, n_grains - 1, 0] > 0:
                for r in range(n_grains):
                    for s in range(n_j):
                        u, v = indices[i, r, s], indices[j, r, s]
                        if u > -1 and v > -1:
                            me_mat[v, u] = k_ij[j, i, r, s]
                            me_mat[u, u] -= k_ij[j, i, r, s]
                            me_mat[u, v] = k_ij[i, j, r, s]
                            me_mat[v, v] -= k_ij[i, j, r, s]

    # association/dissociation
    for i in range(n_isom):
        for n in range(n_prod):
            if g_nj[n, i, n_grains - 1, 0] > 0: # wth is this?
                for r in range(n_grains):
                    for s in range(n_j):
                        u = indices[i, r, s] # column index -> isomer
                        if u > -1: # when is this -1? when not?
                            k_mat[n,u] = g_nj[n,i,r,s] # formation of product n from isomer i,r,s
                            if n < n_reac:
                                b_mat[u,n] = f_im[i, n, r, s] * dens_states[n + n_isom, r, s] * (2 * j_list[s] + 1) * np.exp(-e_list[r] * beta)
                    
    
    # symmetrization
    if symmetrized:
        s_mat = np.zeros(n_rows, np.float64)
        for i in range(n_isom):
            for r in range(n_grains):
                for s in range(n_j):
                    index = indices[i, r, s]
                    if index > -1:
                        s_mat[index] = np.sqrt(dens_states[i, r, s] * (2 * j_list[s] + 1) \
                                            * np.exp(-e_list[r] / (8.314 * temperature)) * eq_ratios[i])

    else:
        s_mat = np.ones(n_rows, np.float64)

    return me_mat, k_mat, b_mat, indices, s_mat
                            
