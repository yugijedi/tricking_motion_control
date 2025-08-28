import mvnx
import numpy as np
import scipy.spatial
import sys
sys.path.insert(0, 'C:/Users/ukasb/Documents/TMA/Scripts')
import model
import body
import RB_functions
import mvnx
import xml.etree.ElementTree as ET
g = 9.81

def readout(file,id,time_window):
    t0,t1 = time_window
    tsteps = t1-t0
    nsegm = 23
    fps = float(ET.parse(file).getroot()[2].attrib['frameRate'])

    #load mvnx data
    mvn = mvnx.MVNX(file)
    mvn_contacts = mvn.footContacts[t0:t1, [1, 3]]
    r_jC = np.swapaxes(np.reshape(mvn.position[t0:t1], [tsteps, nsegm, 3]), axis1=1, axis2=0)
    quat = np.swapaxes(np.reshape(mvn.orientation[t0:t1], [tsteps, nsegm, 4]), axis1=1, axis2=0)
    q = np.swapaxes(np.reshape(mvn.jointAngle[t0:t1], [tsteps, nsegm - 1, 3]), axis1=1, axis2=0)

    # dict that contains all data
    D = {}
    # scalar containers
    E_trans,E_rot,E_pot = np.array([np.zeros([nsegm, tsteps])] * 3)
    # vector containers
    r_sC,v_sC,r_sC_bC,v_sC_bC,L_sC,L_bC,L_oC,w = np.array([np.zeros([nsegm, tsteps, 3])] * 8)
    # matrix containers
    I_bC,R = np.array([np.zeros([nsegm, tsteps, 3, 3])] * 2)
    T = np.zeros([nsegm, tsteps, 4, 4])

    #load body model
    DG = model.model(body.genders[id],body.masses[id],mvn_file=file)

    #write dict
    for i in range(nsegm):
        for j in range(1,tsteps):
            #construct trafo matrix from local to global frame from segm pos and orientations
            R = RB_functions.special_trafo(scipy.spatial.transform.Rotation.from_quat(quat[i, j]).as_matrix())
            T[i,j,:-1,:-1] = R
            T[i,j,:-1,3] = r_jC[i,j]
            T[i,j,3,3] = 1
            dT = (T[i, j] - T[i, j - 1]) * fps
            v_jC = (r_jC[i, j] - r_jC[i, j - 1]) * fps

            #position, velocity and acceleration of segm CoMs in global frame
            r_sC[i, j] = r_jC[i,j] + R @ DG.nodes[i+1]['COM_pos']
            #angular velocity from trafo matrix
            w = RB_functions.V_from_dT(T[i, j], dT)[:-3]  #in world frame
            #v = dr/dt = dr0/dt + omega x (r-r0):
            v_sC[i, j] = v_jC + np.cross(w, r_sC[i,j]-r_jC[i,j])
            # inertia and angular momentum w.r.t. sC
            I_sC = R.T @ DG.nodes[i+1]['inertia'] @ R
            L_sC[i,j] = I_sC @ w
            # angular momentum w.r.t. global frame origin (oC)
            L_oC[i,j] = DG.nodes[i+1]['mass'] * np.cross(r_sC[i, j], v_sC[i, j]) + L_sC[i,j]
            # mechanical energy
            E_trans[i,j] = 1/2 * DG.nodes[i+1]['mass'] * np.dot(v_sC[i,j],v_sC[i,j])
            E_rot[i,j] = np.dot(w, 1/2 * I_sC @ w)
            E_pot[i,j] = 1/2 * DG.nodes[i+1]['mass'] * g * r_sC[i,j,2]

        # write segment quantities into dictionary, accessible as D[segment][quantity][timestep]
        D_segm = {'r_sC': r_sC[i], 'v_sC': v_sC[i],'T':T[i],
                  'E_trans':E_trans,'E_rot':E_rot,'E_pot':E_pot
                  }
        if i==0:
            D_segm.update({'q': np.zeros(tsteps)})
        else:
            D_segm.update({'q': q[i-1]})
        D.update({DG.nodes[i+1]['name']: D_segm})

    # Calculate total body quantities
    r_bC = np.average(r_sC, axis=0, weights=[DG.nodes[i+1]['mass']/body.masses[id] for i in range(23)])
    v_bC = np.average(v_sC, axis=0, weights=[DG.nodes[i+1]['mass']/body.masses[id] for i in range(23)])
    L_bC_oC = body.masses[id] * np.cross(r_bC, v_bC, axis=1)
    L_bC = np.sum(L_oC,axis=0) - L_bC_oC
    E_kin = np.sum(E_rot + E_trans, axis=0)
    D_body = {'r': r_bC, 'v': v_bC, 'L_bC': L_bC, 'E_kin': E_kin, 'E_pot':np.sum(E_pot,axis=0)}
    D.update({'Body': D_body,'mvn_contacts':mvn_contacts})

    for i in range(nsegm):
        r_sC_bC[i] = r_sC[i] - D['Body']['r']
        v_sC_bC[i] = v_sC[i] - D['Body']['v']
        #r_skew = RB_functions.skew(r_sC_bC[i,j])
        #I_bC[i,j] = -DG.nodes[i+1]['mass']*r_skew@r_skew + I_sC[i,j]
        L_bC_i = DG.nodes[i+1]['mass'] * np.cross(r_sC_bC[i], v_sC_bC[i],axis=1) + L_sC[i]
        D[DG.nodes[i+1]['name']].update(
            {'r_sC_bC': r_sC_bC[i], 'v_sC_bC': v_sC_bC[i],'L_bC': L_bC_i})
    

    return D