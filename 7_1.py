import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./mnist/data", one_hot=True)

X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

# 3 X 3 크기의 커널을 가진 컨볼루션 계층 생성
# 입력층 X와 첫 번째 계층의 가중치 W1을 가지고, 오른쪽과 아래쪽으로 한 칸씩 움직이는 커널을 가진 컨볼루션 계층 생성
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
# padding=SAME 은 커널 슬라이딩 시 이미지의 가장 외곽에서 한 칸 밖으로 움직이는 옵션
L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
# 2 X 2 크기의 커널을 가진 풀링 계층 생성. 슬라이딩 시 두 칸씩 움직인다(strides 속성).
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                    padding='SAME')

# 앞서 구성한 첫 번째 컨볼루션 계층 32개(첫 번째 컨볼루션 계층이 찾아낸 이미지의 특징 개수)를 받아서 다시 특징 추출
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                    padding='SAME')

# 직전 풀링의 크기 7X7X64 이므로, reshape 함수를 이용해 7X7X64 크기의 1차원 계층으로 만든 후,
# 배열 전체를 최종 출력값의 중간 단계인 256개의 뉴런으로 연결하는 신경망 생성
W3 = tf.Variable(tf.random_normal([7 * 7 * 64, 256], stddev=0.01))
L3 = tf.reshape(L2, [-1, 7 * 7 * 64])
L3 = tf.matmul(L3, W3)
L3 = tf.nn.relu(L3)
L3 = tf.nn.dropout(L3, keep_prob)

# L3 은닉층의 출력값 256개를 받아 최종 출력값 0~9 레이블을 갖는 10개의 출력값 생성
W4 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
model = tf.matmul(L3, W4)

# 손실 함수 생성
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
# 최적화 함수 생성
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

# 세션 생성
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 50
total_batch = int(mnist.train.num_examples / batch_size)

for epoch in range(15):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # MNIST 데이터를 28X28 형태로 재구성
        batch_xs = batch_xs.reshape(-1, 28, 28, 1)

        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs,
                                                             Y: batch_ys,
                                                             keep_prob: 0.7})
        total_cost += cost_val
    print('Epoch : ', '%04d' % (epoch + 1),
          'Avg. cost = ', '{:3f}'.format(total_cost / total_batch))

print('최적화 완료!')

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

print('정확도 : ', sess.run(accuracy, feed_dict={X: mnist.test.images.reshape(-1, 28, 28, 1),
                                              Y: mnist.test.labels,
                                              keep_prob: 1}))
