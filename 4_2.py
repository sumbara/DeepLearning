# 심층 신경망 구현 -> 단층 신경망 모델에 가중치와 편향을 추가하기만 하면 된다
import tensorflow as tf
import numpy as np

x_data = np.array([
    [0, 0], [1, 0], [1, 1], [0, 0], [0, 0], [0, 1]])

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

# 가중치 설정
W1 = tf.Variable(tf.random_uniform([2, 10], -1., 1.)) # [특징 2개, 은닉층의 뉴런 수 10개]
W2 = tf.Variable(tf.random_uniform([10, 3], -1., 1.)) # [은닉층의 뉴런 수 10개, 분류 수 3개]

b1 = tf.Variable(tf.zeros([10])) # 은닉층의 뉴런 수
b2 = tf.Variable(tf.zeros([3])) # 분류 수

# 입력값에 첫 번째 가중치와 편향을 이용해 활성화 함수를 적용
L1 = tf.add(tf.matmul(X, W1), b1)
L1 = tf.nn.relu(L1)

# 출력층을 만들기 위해 두 번째 가중치와 편향을 적용하여 최종 모델을 생성
model = tf.add(tf.matmul(L1, W2), b2)

# 손실 함수로 교차 엔트로피 함수를 이용
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=model))

# 최적화 함수로 Adam 이용
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(100):
    sess.run(train_op, feed_dict={X: x_data, Y: y_data})

    if (step + 1) % 10 == 0:
        print(step + 1, sess.run(cost, feed_dict={X: x_data, Y: y_data}))

prediction = tf.argmax(model, 1)
target = tf.argmax(Y, 1)
print('예측값 : ', sess.run(prediction, feed_dict={X: x_data}))
print('실제값 : ', sess.run(target, feed_dict={Y: y_data}))

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도 : %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))
