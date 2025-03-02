import numpy as np
from src.BuckleyLeverett import *


class ExplicitAcidSolver:
    def __init__(self,
                 num_of_grid_points: int, grid_step: float,
                 initial_porosity_field: np.ndarray[float],
                 initial_saturation_field: np.ndarray[float],
                 k_a, m, S_s, c_eq,
                 water_viscosity, oil_viscosity,
                 molar_weight_of_carbonate, molar_weight_of_acid, 
                 carbonate_density, acid_density, salt_density, water_density, co2_density):
        
        # хар-ки
        self.num_of_grid_points = num_of_grid_points
        self.grid_step = grid_step
        self.k_a = k_a
        self.m = m
        self.S_s = S_s
        self.c_eq = c_eq
        self.water_viscosity = water_viscosity
        self.oil_viscosity = oil_viscosity
        self.molar_weight_of_carbonate = molar_weight_of_carbonate
        self.molar_weight_of_acid = molar_weight_of_acid
        self.carbonate_density = carbonate_density
        self.acid_density = acid_density
        self.salt_density = salt_density
        self.water_density = water_density
        self.co2_density = co2_density

        # время
        self.time = 0

        # массив пористости
        self.porosity_field = initial_porosity_field

        # массив водонасыщенности
        self.water_saturation_field = initial_saturation_field

        # массивы концентраций
        self.acid_concentration_field = np.zeros(self.num_of_grid_points)
        self.salt_concentration_field = np.zeros(self.num_of_grid_points)
        self.co2_concentration_field = np.zeros(self.num_of_grid_points)


    def time_step(self):
        # tau_W = initial_porosity * h / (2 * velocity) * (1 - EPS)
        # tau_J = 2 * rho1 * M2**m / (k_a * M1 * S_s * rho2**m * (conc_left - c_eq)**m) / 50
        # tau = min(tau_W, tau_J)

        # if tau_J < tau_W:
        # 	print("Ограничитель по скорости реакции")
        # else:
        # 	print("Ограничитель по скорости переноса")
        # t = np.arange(0, total_time + tau, tau)
        # num_of_time_points = t.shape[0]
        # print(f"{num_of_time_points} временных итераций")
        pass    

    def passer(self, time: float,
                water_saturation_left: np.ndarray,
                acid_conc_left: np.ndarray,
                fluid_velocity_left: np.ndarray):
        pass

    def predict(self,
                num_of_time_steps: int, time_step: float,
                water_saturation_left: np.ndarray,
                acid_conc_left: np.ndarray,
                fluid_velocity_left: np.ndarray):

        # массив пористости
        phi = self.porosity_field
        phi_ = np.zeros(self.num_of_grid_points)

        # массив водонасыщенности
        S = self.water_saturation_field
        S_ = np.zeros(self.num_of_grid_points)

        # массивы концентраций
        conc2 = self.acid_concentration_field
        conc2_ = np.zeros(self.num_of_grid_points)

        conc3 = self.salt_concentration_field
        conc3_ = np.zeros(self.num_of_grid_points)

        conc5 = self.co2_concentration_field
        conc5_ = np.zeros(self.num_of_grid_points)

        # массив скорости потока жидкости и скорости реакции
        J = np.zeros(self.num_of_grid_points)
        W = np.zeros(self.num_of_grid_points)

        for n in np.arange(0, num_of_time_steps-1):
            S[0] = water_saturation_left[n]
            S_[0] = water_saturation_left[n]
            
            conc2[0] = acid_conc_left[n]
            conc2_[0] = acid_conc_left[n]

            W[0] = fluid_velocity_left[n]

            if conc2[0] != 0:
                J[0] = self.k_a * S[0] * (1 - phi[0]) * self.molar_weight_of_carbonate / 2 * \
                self.S_s * np.power(self.acid_density / self.molar_weight_of_acid * (conc2[0] - self.c_eq), self.m)
            
            phi_[0] = phi[0] + time_step / self.carbonate_density * J[0]

            for i in np.arange(1, self.num_of_grid_points):
                if conc2[i] != 0:
                    J[i] = self.k_a * S[i] * (1 - phi[i]) * self.molar_weight_of_carbonate / 2 * \
                        self.S_s * np.power(self.acid_density / self.molar_weight_of_acid * (conc2[i] - self.c_eq), self.m)
                
                phi_[i] = phi[i] + time_step / self.carbonate_density * J[i]

                W[i] = W[i-1] + self.grid_step * J[i] * (-1/self.carbonate_density - 2/self.acid_density + \
                                                    1/self.salt_density + 1/self.water_density + 1/self.co2_density)

                b0 = b(S[i], self.water_viscosity, self.oil_viscosity)
                b_ = b(S[i-1], self.water_viscosity, self.oil_viscosity)

                S_[i] = (phi[i] * S[i] + time_step * (J[i] * (-2/self.acid_density + 1/self.salt_density \
                                                              + 1/self.water_density + 1/self.co2_density) - \
                        (W[i] * b0 - W[i-1] * b_) / self.grid_step)) / phi_[i]
                
                if S_[i] >= 1e-9:
                    conc2_[i] = (phi[i] * S[i] * conc2[i] - time_step * (2/self.acid_density * J[i] + \
                                (conc2[i] * W[i] * b0 - conc2[i-1] * W[i-1] * b_) / self.grid_step)) / (phi_[i] * S_[i])
                    conc3_[i] = (phi[i] * S[i] * conc3[i] + time_step * (1/self.salt_density * J[i] - \
                                (conc3[i] * W[i] * b0 - conc3[i-1] * W[i-1] * b_) / self.grid_step)) / (phi_[i] * S_[i])
                    conc5_[i] = (phi[i] * S[i] * conc5[i] + time_step * (1/self.co2_density * J[i] - \
                                (conc5[i] * W[i] * b0 - conc5[i-1] * W[i-1] * b_) / self.grid_step)) / (phi_[i] * S_[i])
                    
            phi = np.copy(phi_)
            S = np.copy(S_)
            conc2 = np.copy(conc2_)
            conc3 = np.copy(conc3_)
            conc5 = np.copy(conc5_)
            J = np.zeros(self.num_of_grid_points)
            W = np.zeros(self.num_of_grid_points)

        # время
        self.time += time_step * num_of_time_steps

        # массив пористости
        self.porosity_field = phi

        # массив водонасыщенности
        self.water_saturation_field = S

        # массивы концентраций
        self.acid_concentration_field = conc2
        self.salt_concentration_field = conc3
        self.co2_concentration_field = conc5

        # return {
        #     "time": time_step * num_of_time_steps,
        #     "porosity": phi, 
        #     "water_saturation": S, 
        #     "acid_concentration": conc2,
        #     "salt_concentration": conc3,
        #     "co2_concentration": conc5
        #     }