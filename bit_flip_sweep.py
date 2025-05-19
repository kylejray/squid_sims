import sys, os, builtins
import numpy as np
import datetime
import json
import copy
import gc
import matplotlib.pyplot as plt

sys.path.append(os.path.expanduser('~/source'))
from quick_sim import setup_sim
from kyle_tools import separate_by_state
from kyle_tools.info_space import is_bundle
from sus.protocol_designer import System
from sus.library.fq_systems import fq_pot

sys.path.append(os.path.expanduser('~/source/simtools/'))
# from infoenginessims.api import *
from infoenginessims.simprocedures import basic_simprocedures as sp
from infoenginessims.simprocedures import running_measurements as rp
from infoenginessims.simprocedures import trajectory_measurements as tp
from infoenginessims.simprocedures.basic_simprocedures import ReturnFinalState
from FQ_sympy_functions import find_px, DeviceParams, fidelity, fidelity_array


# params 1:
Dev = DeviceParams()


# these are some relevant dimensionful scales: Dev.alpha is the natural units for the JJ fluxes and Dev.U_0 is the natural scale for the potential
# IMPORTANT: all energies are measured in units of Dev.U_0
# default_real = (.084, -2.5, 12, 6.2, .2)


# these are important dimensionless simulation quantities, accounting for 
# m being measured in units of Dev.C, lambda in units of 1/Dev.R, energy in units of Dev.U_0

m_prime = np.array((1, 1/4))
lambda_prime = np.array((2, 1/2))
fidelity_thresh = .99
ds_res=.02
prelim_bundle_size=4

L_sweep = Dev.L*(np.linspace(.1,1,7))

L_dict={'param':'L', 'sweep':L_sweep}

def sweep_param(Dev=Dev, sweep_dict=L_dict, N=10_000, N_test=50_000, delta_t=1/200, d_s_c_init=[.2, .2], d_s_max=.44, minimize_ell=False):

    param_vals = sweep_dict['sweep']
    param = sweep_dict['param']
    cnt=0
    date = get_date()

    output_dict={'kT_prime':Dev.kT_prime, 'start_time':date}
    output_dict['param_sweep'] = param_vals
    output_dict['sim_results'] = []

    d_s_c = [item for item in d_s_c_init]
    for param_val in param_vals:
        d_s_c[0] -= ds_res
        if d_s_c[0] < d_s_c_init[0]: 
            d_s_c = [item for item in d_s_c_init]

        cnt += 1
        print("\n {}={};  {} of {}".format(param, param_val, cnt, len(param_vals)))
        temp_dict = {}
        temp_dict['notes'] = []

        Dev.change_vals({param: param_val})

        try: check_device(Dev)
        except AssertionError:
            print('failed at Device check') 
            continue

        
        date2 = get_date()
        print('\n generating prelim valid eq state')

        invalid_initial_state = True
        tries=0

        while invalid_initial_state:
            print(d_s_c)
            tries += 1
        
            try: store_sys, comp_sys = set_systems(Dev, comp_tau=10, d_store_comp=d_s_c)
            except:
                print('unexpectedly failed at system creation')
                break
            
            
            init_state = generate_eq_state(store_sys, N_test, Dev.kT_prime, eq_verbose=False)


            try:
                verify_eq_state(init_state, verbose=False)
                invalid_initial_state = False
            except AssertionError:
                if d_s_c[0] < d_s_max:
                    d_s_c[0] += ds_res
                else:
                    break
        
        temp_dict['store_params'] = store_sys.protocol.params
        temp_dict['comp_params'] = comp_sys.protocol.params
        temp_dict['device'] = Dev.to_dict()


        if invalid_initial_state:
            print('\n bad initial_state at param change')
            temp_dict['initial_state'] = init_state
            temp_dict['terminated'] = True
            output_dict['sim_results'].append(temp_dict)
            continue
                    
            
        is_bools = separate_by_state(init_state[...,0,0])
        prelim_init, prelim_weights = is_bundle(init_state, is_bools, prelim_bundle_size)
        
        date3 = get_date()
        print('took {} tries, time elapsed: {} \n starting prelim sim'.format(tries, duration_minutes(date2,date3)))
        no_candidates = True
        while no_candidates:

            prelim_sim = generate_sim(comp_sys, prelim_init, Dev, delta_t, set_bundle_evo_procs, weights=prelim_weights)
            #prelim_sim = generate_sim(comp_sys, init_state, Dev, 2*delta_t, set_mean_procs)


            prelim_sim.output = prelim_sim.run()

            try:
                tau_candidates, t_crit = get_tau_candidate(prelim_sim)
                no_candidates = False

            except :
                if comp_sys.protocol.t_f > 25:
                    break
                else:
                    comp_sys.protocol.t_f += 10
        
        prelim_d={}
        prelim_d = prelim_sim.output.__dict__
        prelim_d['dt'] = prelim_sim.dt
        prelim_d['nsteps'] = prelim_sim.nsteps
        temp_dict['prelim_sim'] = prelim_d
        

        date4 = get_date()
        print('time elapsed: {} \n'.format(duration_minutes(date3, date4)))

        if no_candidates:
            temp_dict['terminated'] = True
            output_dict['sim_results'].append(temp_dict)
            print('failed at getting time candidates from prelim sim')
            continue

        prelim_d['tau_list'] = tau_candidates
        prelim_d['t_crit'] = t_crit

    

        if minimize_ell:
            sys_candidates = []
            test_times=[]
            t_crit=[]
            Devs=[]
            ell_sims = []
            for idx, item in enumerate(tau_candidates):
                temp_Dev = copy.deepcopy(Dev)
                temp_store, temp_comp = store_sys, comp_sys
                t_p, t_pdc = t_crit, item
                print('\n', 'one side initial_vals', t_p, t_pdc, t_p/t_pdc, temp_Dev.gamma)
                old_ratio = 3
                i=0
                while abs(t_p-t_pdc-1) > .025 and i<10:
                    ell_d = {}
                
                    old_ratio, i_plus = change_ell(temp_Dev, t_p, t_pdc, old_ratio)
                    i += i_plus

                    try: temp_store, temp_comp = set_systems(temp_Dev, comp_tau=10)
                    except AssertionError: continue

                    '''
                    for item in [temp_store, temp_comp]:
                        item.protocol.params[2,:] = temp_Dev.gamma
                    '''
                    for item in [temp_store,temp_comp]:
                        print(item.protocol.params[:,0])

                    init_state = generate_eq_state(store_sys, N_test, Dev.kT_prime)

                    try: verify_eq_state(init_state, verbose=True)
                    except AssertionError:
                        print('\n ran into bad initial state in ell sweep and cancelled it')
                        i=10
                        continue

                    ell_sim = generate_sim(comp_sys, init_state, temp_Dev, 2*delta_t, set_mean_procs)
                    ell_sim.output = prelim_sim.run()

                    z_states = ell_sim.output.zero_means['values']
                    o_states = ell_sim.output.one_means['values']

                    t_pdc, t_p = get_tau_candidate(z_states, o_states, times)
                    t_pdc=t_pdc[idx]

                    print('\n', t_p, t_pdc, t_p/t_pdc, temp_Dev.gamma)

                    ell_d = ell_sim.output.__dict__
                    ell_d['gamma'] = copy.deepcopy(temp_Dev).gamma
                    ell_d['t_p'] = t_p
                    ell_d['t_pdc'] = t_pdc

                    ell_sims.append(ell_d)
                
                test_times[idx] = t_pdc
                t_crit[idx] = t_p
                sys_candidates.append([temp_store, temp_comp])
                Devs.append(temp_Dev)


        else:
            Devs = [Dev]
            t_crit = [t_crit]
            test_times= [tau_candidates]
            sys_candidates= [[store_sys, comp_sys]]

        '''
        if minimize_ell:
            temp_dict['ell_sims'] = ell_sims
        '''
        temp_dict['tau_list']=[]
        temp_dict['sims'] = []

        

        for curr_sys, tau_list, t_crit, Dev in list(zip(sys_candidates, test_times, t_crit, Devs)):

            store_sys, comp_sys = curr_sys
            

            tries=0
            invalid_initial_state=True
            while invalid_initial_state:
                init_state = generate_eq_state(store_sys, N, Dev.kT_prime)
                try:
                    verify_eq_state(init_state)
                    if tries>0:
                        print(' \n took {} tries'.format(tries))
                    invalid_initial_state = False
                except AssertionError:
                    tries+=1
                    if tries < 10:
                        print(' \n unexpected bad initial_state, retrying'.format(tries))
                    else:
                        print('\n no valid initial state found in {} tries'.format(tries))
                        temp_dict['notes'].append('unexpected varification failure in sim initial state')
                        break
        
            min_work, w_idx = sweep_tau_2(Dev, t_crit, tau_list, init_state, comp_sys, store_sys, delta_t, temp_dict)

            temp_dict['min_work'] = min_work
            temp_dict['min_work_index'] = w_idx

        output_dict['sim_results'].append(temp_dict)


    end_date = get_date()
    output_dict['duration'] = duration_minutes(date, end_date)
    
    return output_dict

def get_date():
    return datetime.datetime.now()

def duration_minutes(start_time, end_time):
    delta_t = end_time-start_time
    
    return round(delta_t.seconds/60,2)

def check_device(Device):
    assert 4*Device.gamma > Device.beta, '\n gamma must be >beta/4, it is set too small'
    assert Device.beta > 1, '\n beta cant be < 1, it is set too small'
    assert Device.quantum_ratio() < .1, 'approaching quantum regime'

def set_systems(Device, eq_tau=1, comp_tau=1, d_store_comp=[.2,.2]):

    g, beta, dbeta = Device.gamma, Device.beta, Device.dbeta
    pxdc_crit = -2*np.arccos(1/beta)+ (beta/(2*g))*np.sqrt(1-1/beta**2)
    pxdc_store, pxdc_comp = pxdc_crit+d_store_comp[0], pxdc_crit-d_store_comp[1]

    px_store = find_px(dbeta, pxdc_store, mode='min_of_max')
    px_comp = find_px(dbeta, pxdc_comp, mode='min_of_mid')

    s_params = [px_store, pxdc_store, g, beta, dbeta]
    c_params = [px_comp, pxdc_comp, g, beta, dbeta]

    fq_pot.default_params = s_params
    store_sys = System(fq_pot.trivial_protocol(), fq_pot)
    store_sys.protocol.t_f = eq_tau
    store_sys.mass= m_prime

    fq_pot.default_params = c_params
    comp_sys = System(fq_pot.trivial_protocol(), fq_pot)
    comp_sys.protocol.t_f = comp_tau
    comp_sys.mass = m_prime

    return(store_sys, comp_sys)

def generate_eq_state(eq_sys, N, kT_prime, domain=None, eq_verbose=False):
    pxdc_store = eq_sys.protocol.params[1,0]
    gamma_store = eq_sys.protocol.params[2,0]
    if domain is None:
        range = np.e/np.sqrt(gamma_store/2)
        domain = [[-4, pxdc_store-2*range], [4, pxdc_store+2*range]]
    
    init_state = eq_sys.eq_state(N, beta=1/(kT_prime), manual_domain=domain, axes=[1,2], verbose=eq_verbose)
    return init_state

def info_state_means(initial_state, info_subspace=np.s_[...,0,0]):
    info_state_means=[]
    is_bools = separate_by_state(initial_state[info_subspace])
    for key in is_bools.keys():
        info_state_means.append(initial_state[is_bools[key]].mean(axis=0))
    return np.array(info_state_means)

def generate_sim(comp_sys, initial_state, Device, delta_t, proc_function, **proc_kwargs):
    gamma = (lambda_prime/m_prime) * np.sqrt(Dev.L*Device.C) / (Device.R*Device.C) 
    theta = 1/m_prime
    eta = (Device.L/(Device.R**2 * Device.C))**(1/4) * np.sqrt(Device.kT_prime*lambda_prime) / m_prime
    
    procs = proc_function(initial_state, **proc_kwargs)

    return setup_sim(comp_sys, initial_state, procedures=procs, 
                sim_params=[gamma, theta, eta], dt=delta_t)

def set_mean_procs(initial_state):
    is_bools = separate_by_state(initial_state[...,0,0])
    mean_procs = [
            sp.ReturnInitialState(),
            sp.ReturnFinalState(),
            sp.MeasureMeanValue(rp.get_current_state, output_name = 'zero_means', trial_request=is_bools['0']),
            sp.MeasureMeanValue(rp.get_current_state, output_name = 'one_means', trial_request=is_bools['1'])
            ]
    return mean_procs

def set_mean_evolution_procs(info_state_means):
    mean_evo_procs = [
        sp.ReturnFinalState(),
        sp.ReturnInitialState(),
        sp.MeasureStepValue(rp.get_current_state, trial_request=np.s_[0], output_name='zero_means'),
        sp.MeasureStepValue(rp.get_current_state, trial_request=np.s_[1], output_name='one_means')
    ]
    return mean_evo_procs

def set_bundle_evo_procs(state_bundle, weights=None):
    is_bools = separate_by_state(state_bundle[...,0,0])
    w_z = weights[is_bools['0']]
    w_o = weights[is_bools['1']]

    bundle_evo_procs = [
        sp.ReturnFinalState(),
        sp.ReturnInitialState(),
        sp.MeasureMeanValue(rp.get_current_state, trial_request=is_bools['0'], output_name='zero_means', weights=w_z),
        sp.MeasureMeanValue(rp.get_current_state, trial_request=is_bools['1'], output_name='one_means', weights=w_o)
    ]
    return bundle_evo_procs

def set_tau_procs(initial_state, start=500, stop=600, skip=1, all_state_skip=5):
    is_bools = separate_by_state(initial_state[...,0,0])

    tau_procedures = [
              sp.ReturnFinalState(),
              sp.ReturnInitialState(),
              sp.MeasureAllState(trial_request=slice(0, 50), step_request=np.s_[::all_state_skip]),
              sp.MeasureAllState(trial_request=np.s_[:], step_request=np.s_[start::skip], output_name='final_state_array'),  
              sp.MeasureMeanValue(rp.get_kinetic, output_name='kinetic' ),
              sp.MeasureMeanValue(rp.get_potential, output_name='potential'),
              sp.MeasureMeanValue(rp.get_current_state, output_name = 'zero_means', trial_request=is_bools['0']),
              sp.MeasureMeanValue(rp.get_current_state, output_name = 'one_means',  trial_request=is_bools['1']),
              tp.CountJumps(state_slice=np.s_[...,0,0], step_request=np.s_[start::skip])
             ]
    return tau_procedures

def set_general_procs(initial_state, all_state_skip=5):
    is_bools = separate_by_state(initial_state[...,0,0])
    real_procedures = [
              sp.ReturnFinalState(),
              sp.ReturnInitialState(),
              sp.MeasureAllState(trial_request=slice(0, 50), step_request=np.s_[::all_state_skip]),  
              sp.MeasureMeanValue(rp.get_kinetic, output_name='kinetic' ),
              sp.MeasureMeanValue(rp.get_potential, output_name='potential'),
              sp.MeasureMeanValue(rp.get_current_state, output_name = 'zero_means', trial_request=is_bools['0']),
              sp.MeasureMeanValue(rp.get_current_state, output_name = 'one_means',  trial_request=is_bools['1']),
              tp.CountJumps(state_slice=np.s_[...,0,0])
             ]
    return real_procedures

def find_zero(a, t, burn_in=0, mode='decreasing'):
    D = -2
    if mode == 'increasing':
        D = 2
    assert len(a)==len(t), '\n a and t need to be same length'
    for i,item in enumerate(a):
        if i > burn_in:
            if np.sign(item) - np.sign(last_item) == D:
                t_i = t[i-1] + (t[i]-t[i-1])* abs(last_item)/(abs(last_item)+abs(item))
                return i, t_i
        
        last_item = item

def get_tau_candidate(sim, burn=None):
    z_means = sim.output.zero_means['values']
    o_means = sim.output.one_means['values']
    t = np.linspace(0, sim.dt*sim.nsteps, sim.nsteps+1)

    t_list =[[],[]]
    if burn is None:
        burn = int(1/(t[1]-t[0]))
    i_z, t_z = find_zero(z_means[...,0,1], t, burn_in=burn)
    i_o, t_o = find_zero(o_means[...,0,1], t, burn_in=burn, mode='increasing')
    i_crit = int((i_z+i_o)/2)
    t_crit = (t_o + t_z)/2

    assert np.sign(z_means[i_crit,0,0]) == 1 and np.sign(o_means[i_crit,0,0]) == -1, 'not a good swap'

    for item in [z_means, o_means]:
        t_list[1].append(find_zero(item[i_crit:,1,1], t[i_crit:])[1])

        dt_left = find_zero(item[i_crit::-1,1,1],-t[i_crit::-1]+t[i_crit], mode='increasing')[1]

        t_list[0].append(t[i_crit]-dt_left)

    for i,item in enumerate(t_list):
        t_list[i] = np.mean(item)

    #delta = min([abs(item-t_crit) for item in t_list])
    #t_list = list(filter(lambda x: abs(x-t_crit) <= 2*delta, t_list ))
    
    return t_list, t_crit

def verify_eq_state(init_state, symmetry_tol=.9, separation_tol=3, verbose=False, return_bool=False):
    phi = init_state[...,0,0]

    multistate =  len(np.shape(phi)) > 1
    if verbose:
        print('multistate:',multistate)

    inf_states={}
    inf_states['0'], inf_states['1'] = phi<0, phi>0

    n_z, n_o = np.sum(inf_states['0'], axis=0), np.sum(inf_states['1'], axis=0)

    symmetry = np.isclose(np.minimum(n_z,n_o), np.maximum(n_z, n_o), rtol=symmetry_tol)


    #phi_z, phi_o = phi[inf_states['0']], phi[inf_states['1']]
    phi_z, phi_o = np.ma.masked_where(~inf_states['0'], phi), np.ma.masked_where(~inf_states['1'], phi)
    lim_z = np.mean(phi_z, axis=0)+ separation_tol*np.std(phi_z, axis=0)
    lim_o = np.mean(phi_o, axis=0)- separation_tol*np.std(phi_o, axis=0)

    separation = np.less(lim_z, lim_o)

    if hasattr(separation, 'mask'):
        separation = separation.data


    if multistate:
        if any(separation == False):
            narrow_separation = np.less(np.max(phi_z, axis=0)+.25, np.min(phi_o, axis=0))
            separation = separation | narrow_separation
    else:
        if separation is False:
            if max(phi_z)+.25 < min(phi_o):
                separation=True


    if verbose and not multistate:
        print('symmetry:',symmetry)
        if not symmetry: 
            print('n_z:{},n_o:{}'.format(n_z,n_o))
        print('separation:',separation)
        if not separation:
            print('<zz>:{}, <zo>:{}'.format(np.mean(phi_z, axis=0),np.mean(phi_o, axis=0)))
            print('<sz>:{}, <so>:{}'.format(np.std(phi_z, axis=0),np.std(phi_o, axis=0)))
            print('lim_z: {}, lim_o: {}'.format(lim_z, lim_o))

    if return_bool:
        if multistate:
            return symmetry * separation
        else:
            return bool(symmetry * separation)

    assert separation, 'not separated'
    assert symmetry, 'not symmetric'


def change_ell(Device, t_p, t_pdc, previous_ratio, test_mode=False):
    current_ell = Device.ell
        
    if abs(t_p/t_pdc - 1) <= .95*abs(previous_ratio-1):

        #new_ell = current_ell * ((t_p/t_pdc)**2 +1)/2
        new_ell = current_ell * (t_p/t_pdc)**2
        new_ratio = t_p/t_pdc
        i_plus = 1
    else:
        #new_ell = 2*current_ell /(previous_ratio**2+1)
        new_ell = current_ell * (1/previous_ratio**2)
        new_ratio = previous_ratio
        i_plus = 10
    
    if test_mode:
        new_ell = current_ell + np.sign(t_p/t_pdc-1)*current_ell * .2
        new_ratio = t_p/t_pdc
        i_plus=1

    Device.change_vals({'ell':new_ell})
    return new_ratio, i_plus

def newtime(t1, w1, times, works, iter, delta_t):
    if iter >= 15:
        return t1, [t1, w1] , iter+1
    t2 = times[iter]
    w2 = works[iter]

    if w2 < w1:
        return t2+delta_t*np.sign(t2-t1), [t2, w2], iter+1
    if w2 > w1:
        return t1-delta_t*np.sign(t2-t1), [t1, w1], iter+1

def sweep_tau_2(Dev, t_crit, tau_list, init_state, comp_sys, store_sys, delta_t, write_dict):

    temp_dict={}

    max_time = np.max(tau_list)+.25
    min_time = np.min(tau_list)-.25
    start_idx = np.int(min_time/delta_t)

    comp_sys.protocol.t_f = max_time
    print('\n starting_sim')
    date0 = get_date()

    sim = generate_sim(comp_sys, init_state, Dev, delta_t, set_tau_procs, all_state_skip=int(1/(20*delta_t)), start=start_idx, )
    sim.output=sim.run()

    ###
    date1 = get_date()
    print('time elapsed: {} \n processing sim'.format(duration_minutes(date0,date1)))
    ###

    tau_steps = len(range(sim.nsteps)[start_idx:])
    times = min_time + np.linspace(0, tau_steps*delta_t, tau_steps+1)

    final_states = sim.output.final_state_array['states']

    final_W = store_sys.get_potential(final_states, 0) - comp_sys.get_potential(final_states, 0)
    init_W = comp_sys.get_potential(init_state, 0).transpose() - store_sys.get_potential(init_state, 0).transpose()
    net_W = final_W + init_W
    net_W = net_W.transpose() / Dev.kT_prime


    mean_W = np.mean(net_W, axis=0)
    std_W = np.std(net_W, axis=0)

    min_mean = np.min(mean_W)
    min_idx = int(np.argwhere(mean_W==min_mean))
    print('min_work:{:.2f} at t={:.3f}'.format(min_mean, times[min_idx]))

    jump_array = np.array([ [item['0'],item['1']] for item in sim.output.trajectories['step_list'] ])
    fid = fidelity_array(jump_array)
    valid_fid = fid['overall'] > fidelity_thresh

    fin_s = [final_states[:,i,...] for i in range(len(final_states[0,:,0,0]))]

    valid_fs = [verify_eq_state(state, return_bool=True, separation_tol=2.8) for state in fin_s]
   
 

    lengths = [len(item) for item in [mean_W, fin_s, times, valid_fid]]

    assert max(lengths)==min(lengths), 'works:{}, final_s:{}, times:{}, fid:{} need same dimensions'.format(*lengths)



    if not (valid_fid[min_idx] & valid_fs[min_idx]):
        print('original min_work fid:{}, valid final state:{}'.format(fid['overall'][min_idx], valid_fs[min_idx]))
        try : 
            min_mean = np.min( [ item[0] for item in  zip(mean_W, valid_fid, valid_fs) if bool(item[1]*item[2]) ] )
            min_idx = int(np.argwhere(mean_W==min_mean))
        except:
            print('found no success tau value')
            write_dict['no good swap']=True
            pass
        
        print('min valid changed to:{:.2f} at t={:.3f}'.format(min_mean, times[min_idx]))

    ###
    date2 = get_date()
    print('time elapsed: {}'.format(duration_minutes(date1,date2)))
    ###

    sim.output.final_state = fin_s[min_idx]
    sim.output.tau = times[min_idx]
    sim.output.final_W = net_W[:,min_idx]


    '''
    fig, ax = plt.subplots(2,2)
    ax[0,0].errorbar(times, mean_W, yerr=3*std_W/100)
    ax[1,0].scatter(times, [item['overall'] for item in fid])
    
    for item in [sim.output.zero_means, sim.output.one_means]:
        for slc in [np.s_[start_idx:,0,0], np.s_[start_idx:,0,1], np.s_[start_idx:,1,1]]:
            ax[0,1].plot(times, item['values'][slc])
    
    ax[1,1].plot(times, sim.output.kinetic['values'][start_idx:])
    plt.show()
    '''
    
    del(sim.output.trajectories)
    del(sim.output.final_state_array)
    gc.collect()
    

    sim.output.device=Dev.to_dict()

    sim.output.dt = sim.dt
    sim.output.tau = times[min_idx]
    sim.output.store_params = store_sys.protocol.params[:,0]
    sim.output.comp_params = comp_sys.protocol.params[:,0]
    

    temp_dict['mean_W'] = mean_W
    temp_dict['std_W'] = std_W
    temp_dict['fidelity']=fid
    temp_dict['valid_final_state']=valid_fs

    temp_dict['sim'] = sim.output.__dict__


    write_dict['tau_sweep'] = temp_dict
    write_dict['tau_list'] = times


    return min_mean, min_idx


def sweep_tau(Dev, t_crit, tau_list, init_state, comp_sys, store_sys, delta_t, write_dict, tau_resolution=.025, verbose=True):
    tau_cnt = 0

    for tau in tau_list:
        tau_cnt+=1
        times=[]
        t_new = tau
        w_list = [] 
        iter=0

        while iter <= 15:
        
            if t_new in times:
                t_new = parabola_minimize(times, w_list)
                iter=15

            print("\r tau {} iteration {} ".format(tau_cnt, iter), end="")
            comp_sys.protocol.t_f = t_new
    
            sim = generate_sim(comp_sys, init_state, Dev, delta_t, set_general_procs, all_state_skip=int(1/(20*delta_t)))
            sim.output = sim.run()
    
            final_state=sim.output.final_state
            sim.output.fidelity = fidelity(sim.output.trajectories)
    
            final_W = store_sys.get_potential(final_state, 0) - comp_sys.get_potential(final_state, 0)
            init_W = comp_sys.get_potential(init_state, 0) - store_sys.get_potential(init_state, 0)
            net_W = (final_W + init_W)/Dev.kT_prime
            
            sim.output.final_W = net_W
            sim.output.dt = sim.dt
            sim.output.nsteps = sim.nsteps
            sim.output.init_state = init_state
            sim.output.device = Dev.to_dict()
            sim.output.store_params = store_sys.protocol.params[:,0]
            sim.output.comp_params = comp_sys.protocol.params[:,0]
            write_dict['sims'].append(sim.output.__dict__)
    
            w_list.append(np.mean(net_W))
            times.append(t_new)

            if iter==0:
                curr_w = w_list[0]
                curr_t = times[0]
                t_new += tau_resolution*np.sign(t_crit-t_new)
                iter += 1
            else:
                if verbose:
                    print('t_old, t_new, w_old, w_new',curr_t, times[-1], curr_w, w_list[-1])
                t_new, [curr_t, curr_w], iter = newtime(curr_t, curr_w, times, w_list, iter, tau_resolution)
        
        write_dict['tau_list'].extend(times)
    
    
    return get_best_work(write_dict['sims'])

def get_best_work(sim_list, fidelity_threshold=fidelity_thresh, return_valid_fs=False):
    valid_fs = [verify_eq_state(sim['final_state'], return_bool=True) for sim in sim_list]

    valid_fids = [ sim['fidelity']['overall']>=fidelity_threshold for sim in sim_list ]

    valid_sims = [ item[0] and item[1] for item in zip(valid_fids, valid_fs)]

    mean_works = [ np.mean(sim['final_W']) for sim in sim_list ]

    valid_works = np.array(mean_works)[np.array(valid_sims)]

    min_work, index = None, None
    if len(valid_works) > 0 :
        min_work = np.min(valid_works)
        index = np.squeeze(np.where(mean_works==min_work))

    if return_valid_fs:
        return min_work, index, valid_fs
    else:
        return min_work, index

from scipy.optimize import curve_fit

def parabola_minimize(x,y):
    assert len(x)==len(y), 'x and y need same dimensions'

    def parabola(x, a, b, c):
        return a*x**2 + b*x + c
    
    params, _ = curve_fit(parabola, x, y)

    p_fit = lambda x: parabola(x, *params)
    
    return -params[1]/(2*params[0])
    
