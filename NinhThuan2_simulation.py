import numpy as np
import pandas as pd

class NinhThuanSimulation:
    def __init__(self, num_samples=1000, distances=[10, 30, 60, 100], time_steps=7):
        self.num_samples = num_samples
        self.distances = distances  # km
        self.time_steps = time_steps  # số ngày mô phỏng
        self.Q_base = 1.5e10  # Bq/s, dựa trên JAEA
        self.H = 100  # m, chiều cao nguồn
        self.Gamma = 3.26e-5  # mSv/ngày per Bq/m³, từ FGR 12
        self.stability_class = 'D'  # điều kiện trung bình
        self.decay_constant = np.log(2) / (30.17 * 365)  # hằng số phân rã Cs-137 (1/ngày)

    def run_simulation(self):
        # Dữ liệu thực tế từ Ninh Thuận
        base_radiation = np.random.normal(0.06, 0.01, self.num_samples)  # mSv/ngày tại 60 km
        wind_speed = np.random.normal(9.3, 1.5, self.num_samples)  # m/s
        wind_speed = np.clip(wind_speed, 3, 15)
        wind_direction = np.concatenate([
            np.random.vonmises(np.radians(45), 2, self.num_samples//2) * 180 / np.pi,
            np.random.vonmises(np.radians(225), 2, self.num_samples//2) * 180 / np.pi
        ])
        wind_direction = (wind_direction + 360) % 360
        rainfall = np.random.normal(3.1, 1.0, self.num_samples)  # mm/ngày
        rainfall = np.clip(rainfall, 0, 10)
        core_temperature = np.random.normal(2800, 200, self.num_samples)  # °C
        pressure = np.random.normal(0.84, 0.1, self.num_samples)  # MPa
        core_damage = np.random.uniform(60, 90, self.num_samples)  # %

        results = []
        for i in range(self.num_samples):
            temp_factor = 1 + (core_temperature[i] - 2800) / 2800
            pressure_factor = 1 + (pressure[i] - 0.84) / 0.84
            damage_factor = core_damage[i] / 70
            Q_i_base = self.Q_base * temp_factor * pressure_factor * damage_factor
            u = wind_speed[i]
            if u == 0:
                u = 1e-6

            for t in range(self.time_steps):
                # Giảm Q_i tuyến tính sau ngày thứ 3
                Q_i = Q_i_base * max(1 - 0.2 * max(0, t - 3), 0.2)  # Giảm 20%/ngày sau ngày 3, tối thiểu 20%
                for distance in self.distances:
                    x = distance * 1000  # m
                    sigma_y = 0.08 * x * (1 + 0.0001 * x)**-0.5
                    sigma_z = 0.06 * x * (1 + 0.0015 * x)**-0.5
                    C = (Q_i / (np.pi * u * sigma_y * sigma_z)) * np.exp(-0.5 * (self.H / sigma_z)**2)
                    C = C * np.exp(-self.decay_constant * t)  # Phân rã theo thời gian
                    rain_factor = 1 - 0.05 * rainfall[i] / 4.45  # Mưa rửa trôi
                    C = C * np.clip(rain_factor, 0.5, 1.0)
                    radiation_level = C * self.Gamma
                    results.append({
                        'base_radiation': base_radiation[i],
                        'wind_speed': wind_speed[i],
                        'wind_direction': wind_direction[i],
                        'rainfall': rainfall[i],
                        'core_temperature': core_temperature[i],
                        'pressure': pressure[i],
                        'core_damage': core_damage[i],
                        'radiation_level': radiation_level,
                        'distance': distance,
                        'time_day': t
                    })

        results_df = pd.DataFrame(results)
        results_df.to_csv("NinhThuan_results_time.csv", index=False)
        print(results_df.groupby(['time_day', 'distance'])['radiation_level'].mean().unstack())
        return results_df

if __name__ == "__main__":
    sim = NinhThuanSimulation(num_samples=1000, time_steps=7)
    results = sim.run_simulation()
    print(results.head(10))