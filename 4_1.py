# 단층 신경망 구성
import tensorflow as tf
import numpy as np

# [털, 날개] 없으면 0, 있으면 1
x_data = np.array([
    [0, 0], [1, 0], [1, 1], [0, 0], [0, 0], [0, 1]])
# [털 없고 날개 없음], [털 있고 날개 없음], [털 날개 다 있음] , ., ., [털 없고 날개 있음]

# 레이블 데이터 구성(원-핫 인코딩)
# 기타 = [1, 0, 0]
# 포유류 = [0, 1, 0]
# 조류 = [0, 0, 1]
y_data = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
    [1, 0, 0],
    [0, 0, 1]
])

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# -1 부터 1까지 변하는 가중치를 2행 3열 배열로 정의. 그 이유는 입력값이 1행 2열 배열이고 출력 값이 1행 3열 배열이기 때문
W = tf.Variable(tf.random_uniform([2, 3], -1., 1.))
# 레이블 수인 3개의 요소를 가진 변수로 설정
b = tf.Variable(tf.zeros([3]))

# X 값에 가중치 W를 곱하고 편향 b를 더한 값을 L에 대입
L = tf.add(tf.matmul(X, W), b)
# L값을 활성화 함수인 ReLU에 적용하여 신경망 구성
L = tf.nn.relu(L)

# 신경망을 통해 나온 출력값을 Softmax 함수에 대입. Softmax 함수는 입력값을 정규화하여 출력
# Softmax 함수는 배열 내의 결괏값들의 전체 합이 1이 되도록 만들어준다. 전체가 1이니 각각은 해당 결과의 확률로 해석 가능
model = tf.nn.softmax(L)

# 손실 함수로서 교차 엔트로피 (오차) 함수를 이용(수식 직접 입력)
# Y는 실제 데이터 값, model 값은 신경망을 통해 나온 예측값.
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(model), axis=1))

# 경사하강법으로 최적화
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)

# 텐서플로의 세션 초기화
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 특징과 레이블 데이터를 이용해 100번의 학습 진행
for step in range(200):
    sess.run(train_op, feed_dict={X: x_data, Y: y_data})

    # 학습 도중 10번에 한 번씩 손실값을 출력
    if (step + 1) % 10 == 0:
        print(step + 1, sess.run(cost, feed_dict={X: x_data, Y: y_data}))


# 학습 결과 확인. argmax 함수(요소 중 가장 큰 값의 인덱스 찾아줌)를 사용 레이블 값을 출력
prediction = tf.argmax(model, axis=1)
target = tf.argmax(Y, axis=1)
print('예측값 : ', sess.run(prediction, feed_dict={X: x_data}))
print('실제값 : ', sess.run(target, feed_dict={Y: y_data}))

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도 : %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))