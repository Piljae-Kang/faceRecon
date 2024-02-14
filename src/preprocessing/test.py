import numpy as np

# 예시로 가정한 데이터
raw_R = np.random.randint(0, 256, size=(3, 4), dtype=np.uint64)
raw_G = np.random.randint(0, 256, size=(3, 4), dtype=np.uint64)
raw_B = np.random.randint(0, 256, size=(3, 4), dtype=np.uint64)

# R, G, B를 3차원으로 합치기
raw_RGB = np.concatenate([raw_R, raw_G, raw_B], axis=-1)

# 결과 출력
print("Raw R:")
print(raw_R)
print("\nRaw G:")
print(raw_G)
print("\nRaw B:")
print(raw_B)
print("\nConcatenated raw_RGB:")
print(raw_RGB)

print(raw_RGB.shape)