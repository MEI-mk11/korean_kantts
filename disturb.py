import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr

def remove_emotion(input_audio_path, output_audio_path):
    # 读取音频文件
    audio, sr = librosa.load(input_audio_path, sr=None)

    # 添加噪音
    noise = np.random.normal(0, 0.005, audio.shape)
    noisy_audio = audio + noise

    # 使用noisereduce库减少噪音
    # reduced_audio = nr.reduce_noise(audio_part=noisy_audio, noise_clip=noise)

    # 保存结果
    sf.write(output_audio_path, noisy_audio, sr)

if __name__ == "__main__":
    input_audio_file = "output.wav"
    output_audio_file = "output_audio_no_emotion.wav"

    remove_emotion(input_audio_file, output_audio_file)