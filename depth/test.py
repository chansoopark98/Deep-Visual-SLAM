import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 가상 데이터 생성
np.random.seed(42)
X = np.random.rand(100).astype(np.float32)
y = 3 * X + 2 + np.random.randn(*X.shape) * 0.1

# 데이터 시각화
plt.scatter(X, y)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Generated Data')
plt.show()

# 텐서플로우 데이터셋으로 변환
dataset = tf.data.Dataset.from_tensor_slices((X, y))
dataset = dataset.shuffle(buffer_size=100).batch(10)

# 선형 회귀 모델 정의
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# 모델 컴파일
model.compile(optimizer='sgd', loss='mse')

# 모델 학습
model.fit(dataset, epochs=100)

# 학습된 모델로 예측
predictions = model.predict(X)

# 예측 결과 시각화
plt.scatter(X, y, label='Original Data')
plt.plot(X, predictions, color='red', label='Fitted Line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Fitted Line vs Original Data')
plt.legend()
plt.show()
