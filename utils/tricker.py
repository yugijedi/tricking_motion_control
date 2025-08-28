import mvnx
import networkx as nx
import numpy as np
import os
import sys
import body
import model
import calcs
import calcs_threshold
import utils
from collections import Counter
import RB_functions
import pickle
import shelve
import xml.etree.ElementTree as ET
import combo_dict

sys.path.insert(0, 'C:/Users/ukasb/Documents/TMA/Scripts')
mvnxdir = 'C:/Users/ukasb/Documents/TMA/mvnx_data'
evaldir = 'C:/Users/ukasb/Documents/TMA/eval_data'

class Tricker(object):
    def __init__(self,id):
        self.id = id
        self.height = body.heights[id]
        self.mass = body.masses[id]
        self.gender = body.genders[id]

    def print_stats(self):
        print('----------')
        print('ID: '+str(self.id))
        print('Height: '+str(self.height)+' cm')
        print('Mass: '+str(self.mass)+' kg')
        print('----------')

    def get_dates(self):
        dates = []
        for date in os.listdir(mvnxdir +'/'+ str(self.id)):
            if date == 't-pose':
                continue
            elif date.isnumeric():
                dates.append(date)
        return dates

    def list_combos(self,source='mvn',date=None,combo_name=None,load_folder='eval_data'):
        if load_folder == 'eval_data':
            evaldir_ = evaldir
        else:
            evaldir_ = 'C:/Users/ukasb/Documents/TMA/' + load_folder
        if source=='mvn':
            file_list = []
            all_file_names = os.listdir(mvnxdir + '/' + self.id + '/' + date)
            for file_name in all_file_names:
                if '.mvnx' in file_name:
                    file_list.append(file_name)
            return file_list
        elif source=='dict' and combo_name!=None:
            return os.listdir(evaldir_ + '/' + self.id + '/' + combo_name)
        elif source=='dict' and combo_name==None:
            D = {}
            for combo_name in os.listdir(evaldir_ + '/' + self.id):
                D.update({combo_name:[]})
                for combo_filename in os.listdir(evaldir_ + '/' + self.id + '/' + combo_name):
                    D[combo_name].append(combo_filename)
            return D

    def print_combos(self):
        dates = self.get_dates()
        print('----------')
        for date in dates:
            print(self.id+' performed on '+date[:2]+'.'+date[2:4]+'.'+date[4:6]+':')
            for combo_file in os.listdir(mvnxdir+'/'+str(self.id)+'/'+date):
                print('- '+combo_file.split('.')[0])
        print('----------')

    def write_tree_structure(self):
        DG = nx.DiGraph()
        if self.gender == 'male':
            DG.add_nodes_from(
                [(i, {'name': model.segm_dict_m[i]['name'], 'own_mass': self.mass * model.segm_dict_m[i]['rel_mass'],
                      'dist_mass': [], 'prox_mass': []}) for i in range(1, 24)])
        elif self.gender == 'female':
            DG.add_nodes_from(
                [(i, {'name': model.segm_dict_f[i]['name'], 'own_mass': self.mass * model.segm_dict_f[i]['rel_mass'],
                      'dist_mass': [], 'prox_mass': []}) for i in range(1, 24)])
        DG.add_edges_from(model.joint_list)
        return DG

    def load_combo(self,combo_name,combo_nr=None,date=None,time_window='full',source='mvn',load_folder='eval_data'):
        return Tricker.Combo(combo_name,self.id,time_window,source,combo_nr=combo_nr,date=date,load_folder=load_folder)

    def segm_mass_props(self, mode='by_number', segm_size_source='deLeva'):
        mass_dict = {}
        names = model.segm_list
        masses = model.segm_masses(body.masses[self.id], body.genders[self.id])
        com_poses = model.segm_COM_pos(body.heights[self.id], body.genders[self.id])
        inertias = model.segm_inertias(body.masses[self.id], body.heights[self.id], body.genders[self.id])
        for i in range(23):
            if mode == 'by_number':
                mass_dict.update(
                    {i + 1: {'name': names[i], 'mass': masses[i], 'com_pos': com_poses[i], 'inertia': inertias[i]}})
            elif mode == 'by_name':
                mass_dict.update(
                    {names[i]: {'number': i + 1, 'mass': masses[i], 'com_pos': com_poses[i], 'inertia': inertias[i]}})
        return mass_dict

    '''
    def extract_from_combos(self,quantity,combos,iteration_mode='list'):
        if iteration_mode=='dict':
            combo_list = list(combos[id].keys())
            trial_list = [combos[id][combo]['trials'] for combo in combo_list]
            flat_trial_list = [item for sublist in trial_list for item in sublist]
            for i in range(len(combo_list)):
                for j in range(len(trial_list[i])):
                    combo_name = combo_list[i] + str(trial_list[i][j])
                    print(combo_name)
                    combo = Tricker(id).load_combo(combo_name)

                    c = combo.in_contact()  # 1 if any extremity in contact 0 if not
                    t_switch = np.where(c[1:] - c[:-1])[0]
                    #L_body_sph, L_body_sph_f = combo.L_com(['Body'], sph=True, add_filt=True)
                    #t_o5 = np.where(L_body_sph_f[:, 0] > 1 / 5 * np.max(L_body_sph_f[:, 0]))[0][0]

                    D['alpha'] += list(L_body_sph_f[t_o5:t_switch[-1], 2] * 180 / np.pi)
                    D['beta'] += list(L_body_sph_f[t_o5:t_switch[-1], 1] * 180 / np.pi)
                    D['contact_rel_times'] += list(contacts()[1])
                    for ex in ['RightFoot', 'LeftFoot']:
                        c = combo.contact_state(ex_dict[ex], format='bool')[t_o5:t_switch[-1]]
                        phi, theta, psi = ex_angles(ex, combo, rmv_jumps=False)
        elif mode=='list':
            combo_list = combos
        '''

    def get_contacts_from_combos(self,combo_name_list,orthodox=True):
        c_ex_nr_list = []
        for combo_name in combo_name_list:
            combo = Tricker.load_combo(self, combo_name)
            c, rel_times, abs_times, c_ex_nr = combo.contacts()
            c_ex_nr_list += c_ex_nr
        trans_dict = {i:c_ex_nr_list.count(i) for i in c_ex_nr_list}

    def combo_extraction(self, fullcombo_name, date, mask, cutoff_value=model.omega_BC_norm_threshold, avg_win=60, win_ext=(0, 0),
                         save_combo=True, save_names=None,save_folder='eval_data'):
        fullcombo = self.load_combo(fullcombo_name, date=date)
        if mask=='omega_BC_norm':
            L_bC_abs = np.linalg.norm(fullcombo.dict['Body']['L_bC'],axis=1)[1:]
            I_bC_norm = self.mass*self.height**2/12
            omega_bC_norm = L_bC_abs/I_bC_norm
            quant_filt = utils.window_filter(omega_bC_norm,avg_win)
        if mask=='L_bC':
            quant_filt = utils.window_filter(np.linalg.norm(fullcombo.dict['Body'][mask], axis=1)[1:], window_length=avg_win)
        elif mask=='E_kin_spez':
            quant_filt = utils.window_filter(fullcombo.dict['Body']['E_kin'][1:]/self.mass, window_length=avg_win)
        within_combo = False
        subcombos = []
        if save_names is not None:
            exp_subcombo_count = combo_dict.subcombo_count(date)[fullcombo_name]
        j = 0
        for ti in range(fullcombo.tsteps-1):
            if quant_filt[ti] > cutoff_value and (ti==0 or within_combo == False): 
                start = ti
                print('subcombo',j,'starts at',start)
                within_combo = True
            elif quant_filt[ti] < cutoff_value and within_combo == True:
                end = ti
                print('subcombo',j,'ends at',end)
                within_combo = False
                win = (start - win_ext[0], end + win_ext[1])
                subcombo = self.load_combo(fullcombo_name, date=date, time_window=win)
                subcombos.append(subcombo)
                if save_combo == True:
                    if save_names is None:
                        subcombo.save_combo_dict(fullcombo_name + '_' + str(j),orig_file_name=fullcombo_name,save_folder=save_folder)
                        print(str(win) + ' saved as ' + subcombo.combo_name + '_' + str(j))
                    elif save_names is not None and (j+1) <= exp_subcombo_count:
                        subcombo.save_combo_dict(save_names[j],orig_file_name=fullcombo_name,save_folder=save_folder)
                        print(str(win) + ' saved as ' + save_names[j])
                    else:
                        print('j=',j,'and exp. subcombo count=',exp_subcombo_count)
                        break
                j += 1
        #return fullcombo,subcombos

    class Combo(object):
        def __init__(self,combo_name,id,time_window,source,combo_nr=None,date=None,load_folder='eval_data'):
            self.id = id
            self.combo_name = combo_name
            if source=='mvn':
                if date==None:
                    print('Please specify date when loading from mvn.')
                else:
                    self.date = date
                    self.file = mvnxdir + '/' + self.id + '/' + self.date + '/' + self.combo_name + '.mvnx'
                    self.mvn_data = mvnx.MVNX(self.file)
                    if time_window=='full':
                        self.time_window = (0,len(self.mvn_data.time))
                    elif isinstance(time_window,tuple):
                        if time_window[0] < 0 or time_window[1] > len(self.mvn_data.time):
                            print('Specified time window larger than total recording time. Time window "full" chosen')
                            self.time_window = (0,len(self.mvn_data.time))
                        else:
                            self.time_window = (time_window[0], time_window[1])
                    else:
                        print('Specify the time-window as \'full\' or as a tuple')
                    print('Chosen time window:', time_window)
                    self.tsteps = self.time_window[1] - self.time_window[0]
                    self.dict = calcs.readout(self.file, self.id, time_window=self.time_window)
            #elif source=='fbx':
            #    self.file = 'C:/Users/ukasb/Documents/MasterThesis/DataEval/'\
            #                +self.id+'/'+self.date+'/'+self.combo_name+'/worldtrafos.txt'
            #    self.tsteps = self.time_window[1]-self.time_window[0]
            #    self.dict = calcs.readout(self.file, self.id, time_window=self.time_window,source=source)
            elif source=='dict':
                if combo_nr==None:
                    print('Please specify combo number when loading from dict.')
                else:
                    if load_folder=='eval_data':
                        evaldir_ = evaldir
                    else:
                        evaldir_ = 'C:/Users/ukasb/Documents/TMA/' + load_folder
                    for combo_file_name in os.listdir(evaldir_ + '/' + self.id + '/' + self.combo_name):
                        if str(combo_nr)==combo_file_name[7:-4]:
                            self.combo_nr = combo_nr
                            self.date = combo_file_name[:6]
                            self.file = evaldir_ + '/' + self.id + '/' + self.combo_name + '/' \
                                        + self.date + '_' + str(self.combo_nr) + '.pkl'
                            with open(self.file, 'rb') as f:
                                self.dict = pickle.load(f)
                            self.time_window = self.dict['meta']['time_window']
                            self.tsteps = self.time_window[1] - self.time_window[0]
            elif source=='mvn_thr':
                self.date = date
                self.file = mvnxdir + '/' + self.id + '/' + self.date + '/' + self.combo_name + '.mvnx'
                print('Loading',self.combo_name,'done by',self.id,'on',self.date,'...')
                self.mvn_data = mvnx.MVNX(self.file)
                self.time_window = (0,len(self.mvn_data.time))
                self.tsteps = len(self.mvn_data.time)
                self.dict = calcs_threshold.readout(self.file, self.id, time_window=self.time_window)
                print('done')

        def save_combo_dict(self,save_name,combo_nr=None,orig_file_name=None,save_folder='eval_data'):
            self.dict.update({'meta': {'name': save_name, 'tricker_id': self.id, 'date': self.date,
                                       'time_window': self.time_window}})
            if orig_file_name!=None:
                self.dict.update({'orig_file':orig_file_name})
            if save_folder == 'eval_data':
                file_path = evaldir + '/' + self.id + '/' + save_name
            else:
                file_path = 'C:/Users/ukasb/Documents/TMA/' + save_folder + '/' + self.id + '/' + save_name
            counter = 0
            file_name = file_path + '/' + self.date + '_{}.pkl'
            while os.path.isfile(file_name.format(counter)):
                counter += 1
            file_name = file_name.format(counter)
            if combo_nr==None:
                self.combo_nr = counter
            else:
                self.combo_nr = combo_nr
            os.makedirs(os.path.dirname(file_name), exist_ok=True)
            with open(file_name, 'wb') as f:
                pickle.dump(self.dict, f)

        def get_id(self):
            return self.id

        def get_twist_direction(self):
            return #sign of L_body dot z0: positive is left-twisting and negative right-twisting.
                   # should also return if it changes during the period, if yes should it still be
                   # regarded as a "single" combo?

        #JOINTS

        def write_tree(self, source='mvn'):  # populates the generic human tree structure with joint pos attributes
            self.tree = model.tree_structure()
            for segm in ET.parse(self.file).getroot()[2][1]:
                if segm.attrib['label'] == 'Pelvis':  # pelvis has three childs: jL5S1, jRightHip, jLeftHip
                    nx.set_edge_attributes(self.tree,
                                           {(1, 2): {'pos': [float(el) for el in segm[0][1][0].text.split(' ')]},
                                            (1, 16): {'pos': [float(el) for el in segm[0][2][0].text.split(' ')]},
                                            (1, 20): {'pos': [float(el) for el in segm[0][3][0].text.split(' ')]}})
                elif segm.attrib['label'] == 'T8':  # T8 has three childs: jT1C7, jRightT4Shoulder, jLeftT4Shoulder
                    nx.set_edge_attributes(self.tree,
                                           {(5, 6): {'pos': [float(el) for el in segm[0][1][0].text.split(' ')]},
                                            (5, 8): {'pos': [float(el) for el in segm[0][2][0].text.split(' ')]},
                                            (5, 12): {'pos': [float(el) for el in segm[0][3][0].text.split(' ')]}})
                else:  # all other segments have only one child (excl. fingers)
                    nx.set_edge_attributes(self.tree,
                                           {(5, 6): {'pos': [float(el) for el in segm[0][1][0].text.split(' ')]}})

        def q_for_biorbd(self,return_dq=False,return_ddq=False,incl_segm_trans=False):
            segm_list = model.segm_list
            # joint angles:
            rot = np.loadtxt(self.joint_angle_file).reshape((len(segm_list) + 1, self.tsteps, 3))
            # reordering for conversion between extrinsic and intrinsic rotations: x,y,z -> z,y,x
            order = [2, 1, 0]
            rot = rot[:, :, order]
            # translation of floating base
            trans_hips = np.loadtxt(self.base_transl_file).reshape((1, self.tsteps, 3))
            trans_hips /= 100
            trans_hips = trans_hips[:, :, order]
            if incl_segm_trans==True:
                # all other transl DOFs are set to zero (rigid segm lengths)
                trans = np.concatenate((trans_hips, np.zeros((len(segm_list), self.tsteps, 3))), axis=0)
                # interleaving rot and trans for each DOF
                q = np.zeros(((len(segm_list) + 1) * 2, self.tsteps, 3))
                q[::2] = rot / 360 * 2 * np.pi  # from degrees to radians
                q[1::2] = trans
                # cutting off first and last element and reshaping:
                q = np.swapaxes(q[1:-1], axis1=2, axis2=1)
                q = q.reshape((len(segm_list) * 6, self.tsteps))
            elif incl_segm_trans==False:
                q = np.zeros((len(segm_list) + 1, self.tsteps, 3))
                q[0] = trans_hips
                q[1:] = rot[1:] / 360 * 2 * np.pi  # from degrees to radians
                #reshaping:
                q = np.swapaxes(q, axis1=2, axis2=1)
                q = q.reshape((len(segm_list) * 3 + 3, self.tsteps))
            print('Loaded movement comprises ' + str(q.shape[0]) + ' DOF at ' + str(q.shape[1]) + ' timesteps.')
            dq = (q[:, 1:] - q[:, :-1]) * 60  # calculate dq from q
            ddq = (dq[:, 1:] - dq[:, :-1]) * 60
            if return_dq == True:
                return q, dq
            if return_ddq == True:
                return q, dq, ddq
            else:
                return q

        def q_for_plot(self,order=[2,1,0],return_dq=False):
            rot = np.loadtxt(self.joint_angle_file).reshape((len(Model.segm_list)+1, self.tsteps, 3))
            rot = rot[:, :, order]
            trans = np.loadtxt(self.base_transl_file).reshape((1, self.tsteps, 3))
            q = np.concatenate((trans, rot))
            q = np.delete(q, 1, 0)
            if return_dq == True:
                dq = (q[:, 1:] - q[:, :-1]) * 60  # calculate dq from q
                return dq
            else:
                return q

        #ANGULAR MOMENTUM

        def ext_leg_rot_angles(self,side,system='yeadon',rmv_jumps=False):
            L_bC_sph = utils.cart2sph(self.dict['Body']['L_bC'],self.tsteps)

            r_F = self.dict[side+'Foot']['r_sC_bC']
            r_T = self.dict[side+'Toe']['r_sC_bC']
            angles = utils.angles_from_rotmat(utils.R_10(L_bC_sph[:, 1], L_bC_sph[:, 2],self.tsteps)
                                                 @ utils.R_0ex(r_F, r_T,self.tsteps),self.tsteps
                                              ,rmv_jumps=rmv_jumps)
            return angles


        def I_bC(self,segm_list):
            return np.sum(np.array([self.dict[segm]['I_bC'] for segm in segm_list]),axis=0)

        def L_bC(self,segm_list):
            return np.sum(np.array([self.dict[segm]['L_bC'] for segm in segm_list]),axis=0)

        def limb_contrib(self,segm_list,type='dot_product'):
            L_bC = self.dict['Body']['L_bC']
            L_i = self.L_bC(segm_list)
            return np.einsum('ik,ik->i',L_i,L_bC) / np.linalg.norm(L_bC,axis=1)**2

        def r_lC_bC(self,limb,body_mass):
            listy = []
            limb_mass = 0
            for segm_nr in model.limb_dict_nr[limb]:
                segm_name = model.segm_dict_m[segm_nr]['name']
                segm_mass = model.segm_dict_m[segm_nr]['rel_mass'] * body_mass
                limb_mass += segm_mass
                segm_pos = self.dict[segm_name]['r_sC_bC']
                listy.append(segm_mass*segm_pos)
            return np.sum(np.array(listy), axis=0) / limb_mass

        def ext_leg_contrib(self,side):
            L_bC = self.dict['Body']['L_bC']
            #L_bC_f = utils.window_filter(self.dict['Body']['L_bC'],10)
            #these constants should be computed based on the body parameters
            length = 1.0 #1.08
            mass = 16.7
            prefactor = 1/3*mass*length**2 / np.linalg.norm(L_bC,axis=1)[1:]
            #the angle part
            phi = np.unwrap(self.ext_leg_rot_angles(side)[1:,0])
            phi_dot = np.append(np.diff(phi),0)*60
            theta = self.ext_leg_rot_angles(side)[1:,1]
            angle_part = phi_dot * np.cos(theta)**2
            #return prefactor * angle_part
            return utils.window_filter(prefactor * angle_part,5)


        #CONTACTS

        def contacts(self, z_thr=0.05, fl_thr=17, separation_mode='time',c_source='z_pos',c_return='binary'):
            c = np.zeros((self.tsteps-1, 4))
            if c_source=='mvn':
                c_mvn = self.dict['mvn_contacts'] #read out toe contacts
            else:
                c_mvn = np.zeros((self.tsteps-1, 4))
            bin_conv = {0:np.nan,1:0}
            ex_on_list = []
            ex_off_list = []
            t_on_list = []
            t_off_list = []
            t_trans_list = []
            ex_on_sublist = []
            ex_off_sublist = []
            t_on_sublist = []
            t_off_sublist = []
            fl_count = 0
            for ti in range(1,self.tsteps - 1):
                if c_source=='z_pos': #determine all contacts from z_pos
                    ex_nr_list = range(4)
                elif c_source=='mvn': #determine hand contacts from z_pos and toe contacts from mvn data
                    ex_nr_list = [0,1]
                    if c_return=='binary':
                        c[ti, 2] = c_mvn[ti, 1]  # right toe
                        c[ti, 3] = c_mvn[ti, 0]  # left toe
                    elif c_return=='nan':
                        c[ti, 2] = bin_conv[c_mvn[ti, 1]]  # right toe
                        c[ti, 3] = bin_conv[c_mvn[ti, 0]]  # left toe
                for ex_nr in ex_nr_list:
                    rz = self.dict[model.ex_list[ex_nr]]['r_sC'][ti, 2]
                    if c_return=='binary':
                        if rz < z_thr: #contact
                            c[ti, ex_nr] = 1
                        else:
                            c[ti, ex_nr] = 0
                    elif c_return=='nan':
                        if rz < z_thr: #contact
                            c[ti, ex_nr] = bin_conv[1]
                        else:
                            c[ti, ex_nr] = bin_conv[0]
                # consider contact state changes
                c_diff = c[ti] - c[ti - 1]
                ex_on = np.where(c_diff == 1)[0]
                ex_off = np.where(c_diff == -1)[0]
                # count flight times
                if np.sum(c[ti]) == 0:
                    fl_count += 1
                if len(ex_on) > 0 or ti == self.tsteps - 2:
                    if fl_count > fl_thr or ti == self.tsteps - 2:
                        ex_on_list.append(ex_on_sublist)
                        t_on_list.append(t_on_sublist)
                        ex_off_list.append(ex_off_sublist)
                        t_off_list.append(t_off_sublist)
                        ex_on_sublist = []
                        t_on_sublist = []
                        ex_off_sublist = []
                        t_off_sublist = []
                    fl_count = 0
                if len(ex_on) == 1 and ti > 0:  # one extremity comes in contact
                    ex_on_sublist.append(ex_on[0])
                    t_on_sublist.append(ti)
                if len(ex_off) == 1 and ti != self.tsteps - 2:  # one extremity takes off
                    ex_off_sublist.append(ex_off[0])
                    t_off_sublist.append(ti)
                    # test if order system is not fulfilled: one ex switches one and off during another stays in contact
                    #   in other words: if at least one ex is in contact after the switch (ti) and the ex that went off is also the last that went on, the order of contacts rule is not respected
                    #if np.sum(c[ti]) >= 1 and len(ex_on_sublist) > 1 and ex_off == ex_on_sublist[-1]:
                        #print('Order not respected at', ti)
                #if len(ex_on) > 1:  # two extremities went on at same time
                    #print('Double On at', ti)
                #if len(ex_off) > 1:
                    #print('Double Off at', ti)
            t_fl = np.where(np.sum(c, axis=1) == 0)[0]
            t_ss = np.where(np.sum(c, axis=1) == 1)[0]
            t_ds = np.where(np.sum(c, axis=1) == 2)[0]
            c_dict = {'contacts': c, 'ex_on': ex_on_list, 'ex_off': ex_off_list, 't_on': t_on_list, 't_off': t_off_list,
                      't_fl': t_fl, 't_ss': t_ss, 't_ds': t_ds}  # ,'t_transition':t_trans_list,}
            return c_dict

        def transitions(self,c_dict, filter='inbetween'):
            trans_dict = {'order': [], 'duration': [], 'timing': []}
            if filter == 'full':
                offset = 0
            else:
                offset = 1
            for i in range(offset, len(c_dict['ex_on']) - offset):
                # iterate over all transitions (if mode != full leave out first and last contact)
                ex_on_sub = c_dict['ex_on'][i]
                ex_off_sub = c_dict['ex_off'][i]
                t_on_sub = c_dict['t_on'][i]
                t_off_sub = c_dict['t_off'][i]
                if (len(t_on_sub)==0) and (len(t_off_sub)==0): #no contact switches in combo/data segment
                    continue
                if len(ex_on_sub) != len(ex_off_sub):  # this should never happen for contacts in between flight phases
                    print('Problem!')
                if ex_on_sub == ex_off_sub:  # if lists are equal the contact order rule is respected
                    timing = np.array(t_on_sub[1:]) - np.array(t_off_sub[:-1])
                    trans_dict['timing'].append(list(timing))
                    trans_dict['duration'].append([t_on_sub[0], t_off_sub[-1]])
                    if len(ex_on_sub) == 1:  # single-leg transitions
                        trans_dict['order'].append([ex_on_sub[0]])
                    else:
                        trans_dict['order'].append(ex_on_sub)
                else:
                    print('Order not respected!')
                    trans_dict['timing'].append([])
            return trans_dict

        def plot_contacts(self,t,plot_ax,c_source='z_pos',z_thr=0.05):
            c = self.contacts(c_source=c_source,c_return='nan',z_thr=z_thr)['contacts']
            for ex_nr in range(0,4):
                plot_ax.plot(t, c[:,ex_nr] + ex_nr, 's', color=model.segm_color_dict[model.ex_list[ex_nr]],markersize=5)
            plot_ax.set_yticks([0, 1, 2, 3], model.ex_list_abbrev)
            plot_ax.set_xlim(t[0], t[-1])

        #momentum order

        def mom_contrib_sorting(self,limb_list,mode='after_limb'):
            if mode == 'after_limb': #mostly for plotting
                D = dict([(limb,[]) for limb in limb_list])
                val_list = np.array([self.limb_contrib(model.limb_dict[limb]) for limb in limb_list])
                for ti in range(val_list.shape[1]):
                    lo_name, mid_name, hi_name = limb_list[np.argsort(val_list[:, ti])]
                    D[lo_name].append(0)
                    D[mid_name].append(1)
                    D[hi_name].append(2)
                    #sorted_val_list = np.sort(val_list[:, ti])
            elif mode == 'after_contrib': #for calculating phase switches
                D = {'lo':[],'mid':[],'hi':[],
                     'lo_switch':{'time':[],'limb':[]},
                     'hi_switch':{'time':[],'limb':[]},
                     'switch':{'time':[],'type':[]}}
                val_list = np.array([self.limb_contrib(model.limb_dict[limb]) for limb in limb_list])
                for ti in range(val_list.shape[1]):
                    lo_name, mid_name, hi_name = limb_list[np.argsort(val_list[:, ti])]
                    if ti>1:
                        if hi_name!=D['hi'][-1]: #hi and mid contrib switch
                            last_hi = self.limb_contrib(model.limb_dict[D['hi'][-1]])[ti]
                            last_mid = self.limb_contrib(model.limb_dict[D['mid'][-1]])[ti]
                            curr_hi = self.limb_contrib(model.limb_dict[hi_name])[ti]
                            curr_mid = self.limb_contrib(model.limb_dict[hi_name])[ti]
                            switch_strength = (curr_hi - last_mid) - (curr_mid - last_hi)
                            D['hi_switch']['time'].append(ti)
                            D['hi_switch']['limb'].append(hi_name)
                            D['switch']['time'].append(ti)
                            D['switch']['type'].append('hi')
                            #D['hi_switch']['strength'].append(switch_strength*60)
                        elif lo_name!=D['lo'][-1]: #lo and mid contrib switch
                            last_lo = self.limb_contrib(model.limb_dict[D['lo'][-1]])[ti]
                            last_mid = self.limb_contrib(model.limb_dict[D['mid'][-1]])[ti]
                            curr_lo = self.limb_contrib(model.limb_dict[lo_name])[ti]
                            curr_mid = self.limb_contrib(model.limb_dict[lo_name])[ti]
                            switch_strength = (curr_mid - last_lo) - (curr_lo - last_mid)
                            D['lo_switch']['time'].append(ti)
                            D['lo_switch']['limb'].append(lo_name)
                            D['switch']['time'].append(ti)
                            D['switch']['type'].append('lo')
                            #D['lo_switch']['strength'].append(switch_strength*60)
                    D['lo'].append(lo_name)
                    D['mid'].append(mid_name)
                    D['hi'].append(hi_name)
            else:
                D = {}
            return D