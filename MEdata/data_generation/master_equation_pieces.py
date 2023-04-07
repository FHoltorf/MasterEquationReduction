from arkane.input import load_input_file
from rmgpy.pdep.me import generate_full_me_matrix
from generate_me_pieces import *  

job_list, reaction_dict, species_dict,transition_state_dict, network_dict, level_of_theory = load_input_file("input_no_reverse.py");
pdep_job = job_list[0]

network = pdep_job.network
for reaction in pdep_job.network.path_reactions:
    transition_state = reaction.transition_state
    if transition_state.conformer and transition_state.conformer.E0 is None:
        transition_state.conformer.E0 = (sum([spec.conformer.E0.value_si for spec in reaction.reactants])
                                                 + reaction.kinetics.Ea.value_si, 'J/mol')
        logging.info('Approximated transitions state E0 for reaction {3} from kinetics '
                             'A={0}, n={1}, Ea={2} J/mol'.format(reaction.kinetics.A.value_si,
                                                                 reaction.kinetics.n.value_si,
                                                                 reaction.kinetics.Ea.value_si, reaction.label))
pdep_job.initialize()

import numpy as np
T_range = np.linspace(1000,1500,int((1500-1000)/50) + 1)
P_range = [0.01, 0.1, 1.0]
for T in T_range:
    for P in P_range:
        name_P = str(round(P, 4))
        name_T = str(round(T, 4))
        print("Temperature "+name_T+" and pressure "+name_P)
        name_M = "../"+"M_"+name_P+"_"+name_T
        name_K = "../"+"K_"+name_P+"_"+name_T
        name_B = "../"+"B_"+name_P+"_"+name_T 
        name_T = "../"+"T_"+name_P+"_"+name_T
        network.set_conditions(T,P*1e5)
        network.calculate_equilibrium_ratios()
        M, K, B, indices, S = generate_full_master_equation_LTI(network, symmetrized=True)
        np.savetxt(name_M+".csv", M, delimiter = ",") 
        np.savetxt(name_M+"idx.csv", indices.reshape(indices.shape[0], indices.shape[1]*indices.shape[2]), delimiter = ",")
        np.savetxt(name_K+".csv", K, delimiter = ",")
        np.savetxt(name_B+".csv", B, delimiter = ",")
        np.savetxt(name_T+".csv", S, delimiter = ",")
