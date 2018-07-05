import tensorflow as tf

x_data = [1,2,3]
y_data = [1,2,3]

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

# 선형 회귀 모델에 대한 식 세우기(W는 가중치, b는 편향)
hypotesis = W * X + b

# 회귀 모델의 식 값과 실제 값의 차의 제곱에 대한 손실 값 구하기 > 비용은 모든 데이터에 대한 손실값의 평균을 내어 구함
cost = tf.reduce_mean(tf.square(hypotesis - Y))
# 경사하강법 최적화 함수를 이용(가중치와 편향 값을 변경해가면서 손실값을 최소화). 학습률로 0.1 사용
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
# 비용(손실을 전체 데이터에 대해 구한)을 대입해 최적화를 수행
train_op = optimizer.minimize(cost)

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())

    for step in range(100) :
        _, cost_val = sess.run([train_op, cost], feed_dict={X: x_data, Y: y_data})

        print(step, cost_val, sess.run(W), sess.run(b))

    print("\n=== Test ===")
    print("X: 5, Y: ", sess.run(hypotesis, feed_dict={X: 5}))
    print("X: 2.5, Y: ", sess.run(hypotesis, feed_dict={X: 2.5}))