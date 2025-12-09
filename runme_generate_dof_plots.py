import matplotlib.pyplot as plt
import numpy as np
import pdb
from tqdm import tqdm

colors = ['r', 'g', 'b', 'orange', 'gray']
lambda_ = 0.85e-3 # mm
#D = 60 # mm. Diameter of the aperture

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()  # Create a second y-axis

dof_focal_length = []
for fi, f in enumerate(np.array([25, 50])):
    #c = 5e-3  # mm. No use having pixel size better than this
    res_array = np.arange(0.01, 20, 0.001)

    u = np.zeros((len(res_array),))
    v = np.zeros_like(u)
    dof = np.zeros((len(res_array),))
    D = f
    for i, res in tqdm(enumerate(res_array)):
        hyp_foc_flag = 0
        res_limit_hit = 0
        theta_max = np.arctan(D/2/f)
        theta_min = np.arcsin(0.61 * lambda_ / res)
        u_min = f + 1e-10
        u_max = D/2/np.tan(theta_min)
        c = 5e-10
        minimum_m_possible = -2 * c / res 
        maximum_u_possible = f * (1 - 1/minimum_m_possible)
        if maximum_u_possible < u_max:
            u_max = maximum_u_possible
            print(f'u_min: {u_min} , u_max: {u_max} for focal length {f}')
        if u_min > u_max:
            res_limit_hit = 1
            print(f'Cannot resolve to the required accuracy {res} with focal length {f}')
            
        #u_test_array = np.arange(u_min, u_max, 0.01)
        u_test_array = np.linspace(u_min, u_max, 10)
        dof_over_u = np.zeros_like(u_test_array)
        for ui, u_test in enumerate(u_test_array):
            if res_limit_hit:
                break
            D_opt = 2 * np.tan(theta_min) * u_test # To meet the resolution criteria, you have to make sure you satisfy theta_min at any u
            N = f / D_opt
            v_test = (u_test * f) / (u_test - f)
            m = v_test / u_test
            #c = 2 * res * m
            H = f**2 / (N*c) + f
            if u_test > H:
                print(f'hyperfocal distance found for resolution {res}')
                hyp_foc_flag = 1
            else:
                Dn = (u_test * f) / (f**2 + N * c * (u_test - f))
                Df = (u_test * f) / (f**2 - N * c * (u_test - f))
                dof_over_u[ui] = Df - Dn
        if res_limit_hit:
            dof[i] = -1.
        elif hyp_foc_flag:
            dof[i] = np.inf
            dof[i+1:] = -1
            break
        else:
            dof[i] = np.max(dof_over_u)
            u[i] = u_test_array[np.argmax(dof_over_u)]
            print(D_opt, D, m)
    idx = np.argwhere(dof > 0)
    try:
        print(f'N {N}')
    except NameError:
        print("N not defined")
    # Plot DOF on left axis (ax1)
    ax1.semilogy(res_array[idx], dof[idx], color=colors[fi], linestyle='-',
             label=f'Depth of field, f={f}mm')
    # Plot u on right axis (ax2)
    #ax2.plot(res_array[idx], u[idx] / 10, color=colors[fi], linestyle='--',
    #         label=f'u, f={f}cm')
    dof_focal_length.append(dof)
# Configure left y-axis (DOF)
tick_font_size = 18 
axis_label_font_size = 24 
ax1.set_xlim([0, 2.])
ax1.grid(True)
ax1.grid(True, which='major', linestyle='-', linewidth=0.6, alpha=0.3)
ax1.grid(True, which='minor', linestyle='-', linewidth=0.5, color='gray', alpha=0.3)
id_fly = np.argwhere(np.isclose(res_array, 0.05, atol=1e-3)).flatten()[0]
id_mouse = np.argwhere(np.isclose(res_array, 1.8, atol=1e-3)).flatten()[0]
ax1.scatter(res_array[id_fly], dof[id_fly], color='r')
ax1.scatter(res_array[id_mouse], dof[id_mouse], color='r')
ax1.set_xlabel('Spatial Resolution (mm)', fontsize=axis_label_font_size)
ax1.set_ylabel('Depth of Field (mm)', color='black', fontsize=axis_label_font_size)
ax1.tick_params(axis='both', labelcolor='black', which='major', labelsize=tick_font_size)
ax1.set_ylim([0, 5000])
dof_lim = dof.max()
#ax1.set_ylim([0, 600])

# Configure right y-axis (u)
#ax2.set_ylabel('Object Distance u (mm)', color='black')
#ax2.tick_params(axis='y', labelcolor='black')
# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
#lines2, labels2 = ax2.get_legend_handles_labels()
#ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
ax1.legend(lines1, labels1, loc='best', fontsize=tick_font_size)
ax2.set_yticks([])
plt.show()
