import os
import numpy as np
import pyaudio
import wave
import librosa
import soundfile as sf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.utils import to_categorical

# Функция для записи голоса
def record_voice(filename, duration=5):
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = duration
    WAVE_OUTPUT_FILENAME = filename

    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    print("Recording...")
    frames = []
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("Finished recording.")
    stream.stop_stream()
    stream.close()
    audio.terminate()
    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()

# Функция для извлечения характеристик голоса
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma.T, axis=0)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_mean = np.mean(mel.T, axis=0)
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr.T, axis=0)
    features = np.hstack([mfccs_mean, chroma_mean, mel_mean, zcr_mean])
    return features

# Функция для аугментации данных (изменение скорости и высоты тона)
def augment_data(file_path):
    y, sr = librosa.load(file_path, sr=None)
    # Изменение скорости
    y_speed = librosa.effects.time_stretch(y, rate=1.1)
    # Изменение высоты тона
    y_pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
    return [(y_speed, sr), (y_pitch, sr)]

# Подготовка данных для обучения
def prepare_training_data(directory):
    X = []
    y = []
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            label = filename.split("_")[0]
            file_path = os.path.join(directory, filename)
            features = extract_features(file_path)
            X.append(features)
            y.append(label)
            # Аугментация данных
            augmented_data = augment_data(file_path)
            for y_aug, sr_aug in augmented_data:
                sf.write('aug.wav', y_aug, sr_aug)
                features_aug = extract_features('aug.wav')
                X.append(features_aug)
                y.append(label)
            os.remove('aug.wav')
    return np.array(X), np.array(y)

# Создание модели нейронной сети
def create_model(input_shape, num_classes):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Улучшенная модель с Conv1D слоями
def create_cnn_model(input_shape, num_classes):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv1D(64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Главная функция
def main():
    record_voice("test.wav")
    X, y = prepare_training_data("training_data")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_categorical = to_categorical(y_encoded)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_categorical, test_size=0.2, random_state=42)

    input_shape = (X_train.shape[1],)
    num_classes = y_categorical.shape[1]

    # Использование улучшенной модели с Conv1D
    model = create_cnn_model((input_shape[0], 1), num_classes)

    # Подготовка данных для Conv1D
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

    test_features = extract_features("test.wav").reshape(1, -1)
    test_features_scaled = scaler.transform(test_features)
    test_features_scaled = test_features_scaled[..., np.newaxis]

    prediction_prob = model.predict(test_features_scaled)
    prediction = np.argmax(prediction_prob, axis=1)

    predicted_label = le.inverse_transform(prediction)
    print("Распознанный пользователь:", predicted_label[0])

    threshold = 0.6  # Порог уверенности
    if np.max(prediction_prob) < threshold:
        print("Пользователь не найден в базе данных")
    else:
        authorized_user = "Dinar"  # Замените на нужное имя пользователя
        if predicted_label[0] == authorized_user:
            print("Доступ разрешен")
        else:
            print("Доступ запрещен")

    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_test_labels = np.argmax(y_test, axis=1)

    # Ensure target names match actual classes
    target_names = le.classes_

    # Specify labels parameter
    labels = np.unique(y_test_labels)

    print(classification_report(y_test_labels, y_pred, target_names=target_names, labels=labels, zero_division=0))

if __name__ == "__main__":
    main()
