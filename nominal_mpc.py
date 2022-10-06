import casadi
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import time

class MountainCarMPCParam:
    def __init__(self) -> None:
        self.nx = 2
        self.nu = 1
        self.opt_delta_t = 0.2
        self.opt_n_hrzn = 100
        self.term_cost_factor = 10
        self.car_mass = 0.2
        self.gravity_acc = 9.8
        self.friction_coef = 0.3
        self.x_term = 0.5
        self.a_max = 0.2
        self.a_min = -0.2
        self.x_max = 0.6 # ATTENTION:
        self.x_min = -1.2
        self.v_max = 2.5
        self.v_min = -2.5
        # Simulation
        self.n_sim = 200
        self.s_init = np.array([-0.5, 0])
        # Grid
        self.x_n_grid = 50
        self.v_n_grid = 50


class MountainCarMPCParamSolver:
    def __init__(self, param:MountainCarMPCParam) -> None:
        self.param_ = param
        # System Dynamics
        s_dynamics = casadi.SX.sym('s', self.param_.nx)
        u_dynamics = casadi.SX.sym('a')
        s_next = s_dynamics + casadi.vcat((s_dynamics[1]*self.param_.opt_delta_t,\
            (-self.param_.gravity_acc*self.param_.car_mass*casadi.cos(3*s_dynamics[0]) \
                + u_dynamics - self.param_.friction_coef*s_dynamics[1])\
                    *self.param_.opt_delta_t))
        fun = casadi.Function('fun_next_s', [s_dynamics, u_dynamics], [s_next])
        # Optimization Variables
        nv = self.param_.nx+self.param_.nu
        self.opt_vars = casadi.MX.sym('opt_vars', self.param_.opt_n_hrzn*nv + self.param_.nx)
        self.opt_states = [self.opt_vars[nv*k: nv*k + self.param_.nx] for k in range(self.param_.opt_n_hrzn+1)]
        self.opt_controls = [self.opt_vars[nv*k + self.param_.nx: (k+1)*nv] for k in range(self.param_.opt_n_hrzn)]
        # Initial state
        self.param_x0 = casadi.MX.sym('param_x0', self.param_.nx)
        # Equality constraints: system dynamics, 0, ...n_hrzn
        self.g = [self.opt_states[0]- self.param_x0]
        lb_g = [np.zeros((self.param_.nx, ))]
        ub_g = [np.zeros((self.param_.nx, ))]
        for idx in range(self.param_.opt_n_hrzn):
            x_next = fun(self.opt_states[idx], self.opt_controls[idx])
            self.g.append(self.opt_states[idx+1] - x_next)
            lb_g.append(np.zeros((self.param_.nx, )))
            ub_g.append(np.zeros((self.param_.nx, )))
        # Inequality constraints: input constraints, 0, ...n_hrzn-1
        for idx in range(self.param_.opt_n_hrzn):
            # acceleration at stage idx
            self.g.append(self.opt_controls[idx])
            lb_g.append([self.param_.a_min])
            ub_g.append([self.param_.a_max])
        # Inequality constraints: state constraints, 0, ...n_hrzn
        for idx in range(self.param_.opt_n_hrzn+1):
            # position and velocity at stage idx
            self.g.append(self.opt_states[idx])
            lb_g.append([self.param_.x_min, self.param_.v_min])
            ub_g.append([self.param_.x_max, self.param_.v_max])
        # Sigmoid function
        x_sigmoid = casadi.SX.sym('x_sigmoid')
        fun_sigmoid = casadi.Function('fun_sigmoid', [x_sigmoid], [-1/(1+casadi.exp(-x_sigmoid))])
        # Cost function
        self.obj = 0
        for idx in range(self.param_.opt_n_hrzn):
            # self.obj = self.obj + casadi.fabs(self.opt_states[idx][0] - self.param_.x_term)
            self.obj = self.obj + fun_sigmoid(self.opt_states[idx][0] - self.param_.x_term)
        self.obj = self.obj + fun_sigmoid(self.opt_states[self.param_.opt_n_hrzn][0] - self.param_.x_term) \
            * self.param_.term_cost_factor
        
        self.g_vcat = casadi.vertcat(*self.g)
        self.lbg_cat = np.concatenate(lb_g)
        self.ubg_cat = np.concatenate(ub_g)
        
        self.nlp = {"x": self.opt_vars, "f":self.obj, "g":self.g_vcat, "p":self.param_x0}
        self.ipopt_opts = {"print_time": False, 
                    "ipopt.print_level": 0, 
                    "ipopt.max_iter": 10000}
        # Call the IPOPT solver with nlp object and options
        self.solver = casadi.nlpsol("solver", "ipopt", self.nlp, self.ipopt_opts)
        self.initialized_flag = False
        
        
    def solve(self, s_current):
        if self.initialized_flag == False:
            x_guess = np.tile(s_current, reps=(self.param_.opt_n_hrzn+1, 1))
            u_guess = np.zeros((self.param_.opt_n_hrzn+1, self.param_.nu))
            xu_guess_stack = (np.hstack((x_guess, u_guess))).flatten(order='C')
            self.opt_guess = xu_guess_stack[:-self.param_.nu]
            self.initialized_flag = True
        
        sol = self.solver(x0=self.opt_guess, lbg=self.lbg_cat, ubg=self.ubg_cat, 
                         p=s_current)
        if not self.solver.stats()['success']:
            raise Exception("fail")
        # for next timeslot
        sol_val = sol['x'].full().flatten()
        s_next = sol_val[self.param_.nx+self.param_.nu:self.param_.nx*2+self.param_.nu]
        u0 = sol_val[self.param_.nx:self.param_.nx+self.param_.nu]
        self.opt_guess = sol_val
        return s_next, u0
    
    
def plot_traj(x, ts):
    plt.figure(1)
    plt.plot(ts, x)
    plt.show()


def main():
    param = MountainCarMPCParam()
    mpc_solver = MountainCarMPCParamSolver(param)
    sim_traj = np.zeros((param.n_sim+1, param.nx))
    ts = np.linspace(start=0, stop=param.n_sim*param.opt_delta_t, num=param.n_sim+1, endpoint=True)
    sim_traj[0,:] = param.s_init
    t_array = np.zeros((param.n_sim,))
    u0_array = np.zeros((param.n_sim+1, param.nu))
    for idx in range(param.n_sim):
        # print("idx=",idx,"pos=",sim_traj[idx,0])
        t_start = time.process_time()
        sim_traj[idx+1,:], u0_array[idx] = mpc_solver.solve(sim_traj[idx,:])
        t_end = time.process_time()
        t_array[idx] = t_end - t_start
        print(idx, t_end - t_start)
        
    plot_traj(u0_array[:], ts)
    print(np.mean(t_array), np.sum(t_array))
    
    
def gird_dataset():
    param = MountainCarMPCParam()
    mpc_solver = MountainCarMPCParamSolver(param)
    input_s = np.zeros((param.x_n_grid * param.v_n_grid, param.nx))
    output_a = np.zeros((param.x_n_grid * param.v_n_grid, param.nu))
    counter = 0
    for x_i, x_val in enumerate(np.linspace(start=param.x_min, stop=param.x_max, num=param.x_n_grid)):
        mpc_solver.initialized_flag = False
        for v_i, v_val in enumerate(np.linspace(start=param.v_min, stop=param.v_max, num=param.v_n_grid)):
            _, u0 = mpc_solver.solve(np.array([x_val, v_val]))
            input_s[counter, 0] = x_val
            input_s[counter, 1] = v_val
            output_a[counter] = u0
            print(counter, input_s[counter,:], output_a[counter])
            counter += 1
            
    scipy.io.savemat("deltaT02M100.mat",{'input_s':input_s, 'output_a':output_a})
    print(output_a)
    
if __name__ == "__main__":
    main()