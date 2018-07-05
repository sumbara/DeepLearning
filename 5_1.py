# 체크포인트 사용
import tensorflow as tf
import numpy as np

# 데이터 파일(data.csv)로 부터 데이터 읽어 들이고 변환
data = np.loadtxt('./data.csv', delimiter=',',
                  unpack=True, dtype='float32')
x_data = np.transpose(data[0:2])
y_data = np.transpose(data[2:])

##################
# 신경망 모델 구성
##################

# 학습에는 직접 사용되지 않고, 학습 횟수를 카운트 하는 변수 생성
global_step = tf.Variable(0, trainable=False, name='global_step')

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# 편향 없이 가중치만 사용하여 모델 생성
W1 = tf.Variable(tf.random_uniform([2, 10], -1., 1.))
L1 = tf.nn.relu(tf.matmul(X, W1))

W2 = tf.Variable(tf.random_uniform([10, 20], -1., 1.))
L2 = tf.nn.relu(tf.matmul(L1, W2))

W3 = tf.Variable(tf.random_uniform([20, 3], -1., 1.))
model = tf.matmul(L2, W3)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=model))

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
# 최적화 함수가 학습용 변수들을 최적화 할때마다 global_step 변수의 값이 1씩 증가
train_op = optimizer.minimize(cost, global_step=global_step)

##################
# 신경망 모델 학습
##################

sess = tf.Session()
# tf.global_variables는 앞서 정의한 변수들을 가져오는 함수이다.
saver = tf.train.Saver(tf.global_variables())

# 학습된 모델이 저장된 파일인 체크포인트를 ./model 디렉터리에서 가져온다
ckpt = tf.train.get_checkpoint_state('./model')
# 체크포인트가 있다면 학습된 값들을 불러오고, 아니면 변수를 새로 초기화한다.
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    sess.run(tf.global_variables_initializer())

for step in range(2):
    sess.run(train_op, feed_dict={X: x_data, Y: y_data})

    print('Step : %d, ' % sess.run(global_step),
          'Cost : %.3f' % sess.run(cost, feed_dict={X: x_data, Y: y_data}))

# 최적화가 끝난 뒤 학습된 변수들을 지정한 체크포인트 파일에 저장
saver.save(sess, './model/dnn.ckpt', global_step=global_step)

###########
# 결과 확인
###########

prediction = tf.argmax(model, 1)
target = tf.argmax(Y, 1)
print('예측값 : ', sess.run(prediction, feed_dict={X: x_data}))
print('실제값 : ', sess.run(target, feed_dict={Y: y_data}))

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도 : %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))