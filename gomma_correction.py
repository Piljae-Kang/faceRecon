import numpy as np
import matplotlib.pyplot as plt

# 주어진 데이터
real_color = np.array([52, 85, 122, 160, 200, 243])
irradiance = np.array([3.1, 8.6, 18.4, 36.1, 59.7, 90.0])

# 감마 값 설정
gamma = 2.4

# 감마 보정 함수
def gamma_correction(input_values, gamma):
    return np.power(input_values, gamma)

# 실제 색상에 감마 보정 적용
corrected_color = gamma_correction(real_color, gamma)

correlation_coefficient = np.corrcoef(corrected_color, irradiance)[0, 1]

print(f"상관 계수: {correlation_coefficient}")

# 결과 출력
plt.plot(irradiance,corrected_color, label='Gamma Corrected Real Color', marker='o')
plt.xlabel('irradiance')
plt.ylabel('corrected_color')
plt.show()