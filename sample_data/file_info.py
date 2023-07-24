import wave

import torch

# wav 파일 열기sample
file = wave.open("sample.wav", "r")

# 파일 정보 출력
print("Channels: ", file.getnchannels())  # 채널 수
print("Sample width: ", file.getsampwidth())  # 샘플의 너비
print("Frame rate (Sample rate): ", file.getframerate())  # 프레임 레이트
print("Frame count (Sample length): ", file.getnframes())  # 프레임 수
print("Duration: ", file.getnframes() / file.getframerate())  # 재생 시간

time = file.getnframes() / file.getframerate()

# 파일 닫기
file.close()

# .pth 파일 로드
tensor = torch.load("sample.pth")

# tensor의 shape 출력
tensor_time_seq = tensor.shape[2]
print(tensor_time_seq, "은", time, "x 75 = ", time * 75, "와 같습니다.")

import os
import torch
import matplotlib.pyplot as plt


def find_shape2_in_pth_files(directory):
    shape2_list = []

    # 디렉토리 내의 모든 파일과 하위 디렉토리를 순회
    for root, dirs, files in os.walk(directory):
        for file in files:
            # .pth 파일인 경우
            if file.endswith(".pth"):
                # 파일 로드
                tensor = torch.load(os.path.join(root, file))
                # shape[2]가 존재하는 경우
                if len(tensor.shape) > 2:
                    # shape[2]를 리스트에 추가
                    shape2_list.append(tensor.shape[2])

    return shape2_list


# 사용 예
directory = "data"  # 검색할 디렉토리
shape2_list = find_shape2_in_pth_files(directory)

# 히스토그램 출력
plt.hist(shape2_list, bins="auto")
plt.title("Distribution of shape[2] in .pth files")
plt.xlabel("shape[2]")
plt.ylabel("Frequency")
plt.show()
