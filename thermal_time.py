import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import os

class APSIMWheatPhenology:
    def __init__(self, R_p=1.5, R_v=1.5, sowing_date=None, H_snow=0, D_seed=40, T_lag=40, r_e=1.5):
        self.R_p = R_p
        self.R_v = R_v
        self.V = 0
        self.cumulative_TT = 0
        self.TT_prime = 0
        self.TT_post = 0
        self.sowing_date = sowing_date
        self.H_snow = H_snow
        self.D_seed = D_seed
        self.T_lag = T_lag
        self.r_e = r_e
        self.emergence_date = None
        self.emergence_TT = 0

    def photoperiod_factor(self, L_p):
        return round(1 - 0.002 * self.R_p * (20 - L_p) ** 2, 3)

    def crown_temperature_max(self, T_max):
        if T_max >= 0:
            return round(T_max, 3)
        else:
            return round((2 + T_max * (0.4 + 0.0018 * (self.H_snow - 15) ** 2)), 3)

    def crown_temperature_min(self, T_min):
        if T_min >= 0:
            return round(T_min, 3)
        else:
            return round((2 + T_min * (0.4 + 0.0018 * (self.H_snow - 15) ** 2)), 3)

    def crown_temperature(self, T_max, T_min):
        T_cmax = self.crown_temperature_max(T_max)
        T_cmin = self.crown_temperature_min(T_min)
        return round((T_cmax + T_cmin) / 2, 3)

    def daily_thermal_time(self, T_c):
        if 0 < T_c <= 26:
            return round(T_c, 3)
        elif 26 < T_c <= 34:
            return round((34 - T_c) * 26 / 8, 3)
        else:
            return 0

    def vernalisation_increment(self, T_c, T_max, T_min):
        if T_max < 30 and T_min < 15:
            return round(min(1.4 - 0.0778 * T_c, 0.5 + 13.44 * (T_c / (T_max - T_min + 3) ** 2)), 3)
        return 0

    def devernalisation_increment(self, T_max):
        if T_max > 30 and self.V < 10:
            return round(min(0.5 * (T_max - 30), self.V), 3)
        return 0

    def update_vernalisation(self, T_c, T_max, T_min):
        delta_v = self.vernalisation_increment(T_c, T_max, T_min)
        delta_vd = self.devernalisation_increment(T_max)
        self.V += delta_v - delta_vd
        return round(self.V, 3)

    def vernalisation_factor(self):
        return round(1 - (0.0054545 * self.R_v + 0.0003) * (50 - self.V), 3)

    def estimate_day_length(self, date: datetime, latitude: float) -> float:
        day_of_year = date.timetuple().tm_yday
        declination = 23.44 * np.sin(np.deg2rad(360 / 365 * (day_of_year - 81)))

        rad_latitude = np.deg2rad(latitude)
        rad_declination = np.deg2rad(declination)
        twilight_angle = np.deg2rad(6)

        cos_hour_angle = (np.sin(-twilight_angle) - np.sin(rad_latitude) * np.sin(rad_declination)) / \
                         (np.cos(rad_latitude) * np.cos(rad_declination))
        cos_hour_angle = np.clip(cos_hour_angle, -1, 1)
        hour_angle = np.arccos(cos_hour_angle)

        day_length = 2 * np.rad2deg(hour_angle) / 15
        return round(day_length, 3)

    def germination_to_emergence(self):
        return round(self.T_lag + self.r_e * self.D_seed, 3)

    def accumulate_daily_values(self, daily_data, latitude):
        results = []
        for index, row in daily_data.iterrows():
            date = datetime(row['year'], 1, 1) + timedelta(days=row['day'] - 1)

            if self.sowing_date and date < self.sowing_date:
                continue

            L_p = self.estimate_day_length(date, latitude)  # day length in hours
            f_D = self.photoperiod_factor(L_p)
            T_c = self.crown_temperature(row['maxt'], row['mint'])
            delta_tt = self.daily_thermal_time(T_c)
            self.cumulative_TT += delta_tt

            if date > self.sowing_date:
                self.emergence_TT += delta_tt
                if not self.emergence_date and self.emergence_TT >= self.germination_to_emergence():
                    self.emergence_date = date

            if date > self.sowing_date + timedelta(days=1):
                V = self.update_vernalisation(T_c, row['maxt'], row['mint'])
                f_V = self.vernalisation_factor()
            else:
                V = self.V
                f_V = 1

            min_f_D_f_V = min(f_D, f_V)
            TT_prime_increment = delta_tt * min_f_D_f_V

            if not self.emergence_date or (self.emergence_date and date <= self.emergence_date):
                TT_post = delta_tt
            else:
                TT_post = TT_prime_increment

            self.TT_post += TT_post

            results.append({
                "Date": date,
                "Year": row['year'],
                "Month": date.month,
                "Day": date.day,
                "T_max": round(row['maxt'], 3),
                "T_min": round(row['mint'], 3),
                "L_p": L_p,
                "Photoperiod factor (f_D)": f_D,
                "Crown temperature (T_c)": T_c,
                "Total vernalisation (V)": V,
                "Vernalisation factor (f_V)": f_V,
                "delta_TT": round(TT_post, 3),
                "Cumulative_TT": round(self.TT_post, 3),
                "Emergence_threshold": self.germination_to_emergence(),
                "Emergence_date": self.emergence_date.timetuple().tm_yday if self.emergence_date else None
            })

        return pd.DataFrame(results)


def main():
    latitude = 38.14787
    file_path = './input/input_weather.csv'
    daily_data = pd.read_csv(file_path)

    sowing_date = datetime(1975, 11, 5)
    apsim_wheat = APSIMWheatPhenology(R_p=1.5, R_v=1.5, sowing_date=sowing_date)
    results_df = apsim_wheat.accumulate_daily_values(daily_data, latitude)

    output_path = './output'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_filename = 'tt_output.csv'
    results_df.to_csv(os.path.join(output_path, output_filename), index=False)

    print(results_df.head())


if __name__ == "__main__":
    main()
