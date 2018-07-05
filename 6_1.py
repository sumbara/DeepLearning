import tensorflow as tf

# MNIST를 사용하기 위해 import
from tensorflow.examples.tutorials.mnist import input_data
# MNIST 데이터를 내려받고 레이블을 원-핫 인코딩 방식으로 읽어들이기
mnist = input_data.read_data_sets("./mnist/data", one_hot=True)

# 손글씨 이미지는 28X28 픽셀이다. 784개의 특징이다.
# None 으로 설정 되어 있는 차원에는 배치 크기(한 번에 학습시킬 MNIST 이미지의 개수)가 지정된다.
# 원하는 배치 크기로 명시해줘도 되지만, None으로 넣어주면 텐서플로가 알아서 계산한다
X = tf.placeholder(tf.float32, [None, 784])
# 레이블은 0-9까지의 숫자니 10개의 분류로 나눈다.
Y = tf.placeholder(tf.float32, [None, 10])

# 입력 특징의 개수는 784개(28X28). 첫 번째 은닉층의 뉴런 개수는 256개이다.
# 표준편차가 0.01인 정규분포를 가지는 임의의 값으로 뉴런(변수)을 초기화한다.
W1 = tf.Variable(tf.random_normal([784, 256], stddev=0.01))
L1 = tf.nn.relu(tf.matmul(X, W1))

# 두 번째 은닉층의 뉴런 개수도 256개이다. 각 계층으로 들어오는 입력값에 각각의 가중치를 곱하고
# ReLU 함수를 사용하는 신경망 계층을 만든다
W2 = tf.Variable(tf.random_normal([256, 256], stddev=0.01))
L2 = tf.nn.relu(tf.matmul(L1, W2))

# 출력층에서는 보통 활성화 함수 사용X. model 텐서에 W3 변수를 곱함으로 요소 10개짜리 배열이 출력.
# 10개는 0~9의 숫자를 의미하며 가장 큰 값의 인덱스가 예측 결과에 가까운 숫자이다.
W3 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
model = tf.matmul(L2, W3)

# 이미지에 대한 손실값 구하기(교차 엔트로피 오차 함수를 이용)
# tf.reduce_mean 함수를 이용해 미니배치의 평균 손실값을 구한다.
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
# 손실값을 최소화하는 최적화를 수행(학습률 0.001)
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

# 세션 시작
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 미니배치의 크기를 100개로 설정, 학습 데이터의 총 개수를 배치 크기로 나눠 미니배치가 총 몇개인지 저장
batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

# 데이터 전체를 학습하는 일을 15번 반복
for epoch in range(15):
    total_cost = 0
    for i in range(total_batch):
        # 학습할 데이터를 배치 크기만큼 가져와, 이미지 데이터는 xs에 레이블 데이터는 ys에 저장
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # 최적화시키고 손실값을 가져와서 저장한다
        _, cost_val = sess.run([optimizer, cost],
                               feed_dict={X: batch_xs, Y: batch_ys})
        total_cost += cost_val

    # 한 세대의 학습이 끝나면 학습한 세대의 평균 손실값을 출력
    print('Epoch : ', '%04d' % (epoch + 1),
          'Avg. cost = ', '{:3f}'.format(total_cost / total_batch))

print('최적화 완료!')

# 예측 결과인 model의 값과 실제 레이블인 Y의 값을 비교
is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
# tf.cast를 이용 is_correct를 0과 1로 변환. 변환한 값들을 평균을 내면 정확도가 나옮
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# 테스트 데이터를 이용해 accuracy 계산
print('정확도 : ', sess.run(accuracy, feed_dict={X: mnist.test.images,
                                                   Y:mnist.test.labels}))