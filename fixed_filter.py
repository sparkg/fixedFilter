from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
#读取数据
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess=tf.InteractiveSession()
#构建cnn网络结构
#自定义卷积函数（后面卷积时就不用写太多）
def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')
#自定义池化函数
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
#设置占位符，尺寸为样本输入和输出的尺寸
x=tf.placeholder(tf.float32,[None,784])
y_=tf.placeholder(tf.float32,[None,10])
x_img=tf.reshape(x,[-1,28,28,1])


#=====================================================
temp = [np.array([[1,1,1],[0,0,0],[0,0,0]]), np.array([[0,0,0],[1,1,1],[0,0,0]]), np.array([[0,0,0],[0,0,0],[1,1,1]]),
               np.array([[1,0,0],[1,0,0],[1,0,0]]), np.array([[0,1,0],[0,1,0],[0,1,0]]), np.array([[0,0,1],[0,0,1], [0,0,1]]),
               np.array([[1,0,0],[0,0,0],[0,0,0]]), np.array([[0,1,0],[1,0,0],[0,0,0]]), np.array([[0,0,1],[0,1,0],[1,0,0]]),
               np.array([[0,0,0],[0,0,1],[0,1,0]]), np.array([[0,0,0],[0,0,0],[0,0,1]]), np.array([[0,0,1],[0,0,0],[0,0,0]]),
               np.array([[0,1,0],[0,0,1],[0,0,0]]), np.array([[1,0,0],[0,1,0],[0,0,1]]), np.array([[0,0,0],[1,0,0],[0,1,0]]),
               np.array([[0,0,0],[0,0,0],[1,0,0]]), np.array([[1,1,0],[0,0,1],[0,0,0]]), np.array([[0,0,0],[1,1,0],[0,0,1]]),
               np.array([[0,0,1],[1,1,0],[0,0,0]]), np.array([[0,0,0],[0,0,1],[1,1,0]]), np.array([[1,0,0],[0,1,0],[1,0,0]]),
               np.array([[0,1,0],[0,0,1],[0,1,0]]), np.array([[0,0,1],[0,1,0],[0,0,1]]), np.array([[0,1,0],[1,0,0],[0,1,0]]),
               np.array([[1,1,0],[0,0,1],[1,1,0]]), np.array([[0,1,1],[1,0,0],[0,1,1]])]
temp = list(map(lambda x: list(map(lambda y:list(y),x)), temp))
temp = np.array(temp)
temp = np.reshape(temp, [3,3,1,26])
#=====================================================
#设置第一个卷积层和池化层
w_conv1=(1/255.0) * tf.constant(temp, dtype=tf.float32)
b_conv1=tf.constant(0.1,shape=[26])
h_conv1=tf.nn.relu(conv2d(x_img,w_conv1)+b_conv1)
h_pool1=max_pool_2x2(h_conv1)


#设置第一个全连接层
w_fc1=tf.Variable(tf.truncated_normal([14*14*26,1024],stddev=0.1))
b_fc1=tf.Variable(tf.constant(0.1,shape=[1024]))
h_pool2_flat=tf.reshape(h_pool1,[-1,14*14*26])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+b_fc1)

#dropout（随机权重失活）
keep_prob=tf.placeholder(tf.float32)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

#设置第二个全连接层
w_fc2=tf.Variable(tf.truncated_normal([1024,10],stddev=0.1))
b_fc2=tf.Variable(tf.constant(0.1,shape=[10]))
y_out=tf.nn.softmax(tf.matmul(h_fc1_drop,w_fc2)+b_fc2)

#建立loss function，为交叉熵
loss=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_out),reduction_indices=[1]))
#配置Adam优化器，学习速率为1e-4
train_step=tf.train.AdamOptimizer(1e-4).minimize(loss)

#建立正确率计算表达式
correct_prediction=tf.equal(tf.argmax(y_out,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#开始喂数据，训练
tf.global_variables_initializer().run()

#写入文件
f = open(r'D:/fixed_5.txt','w')
for i in range(100000):
    batch=mnist.train.next_batch(50)
    if i%100==0:
        train_accuracy=accuracy.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1})
        print("step %d,train_accuracy= %g"%(i,train_accuracy))
        f.writelines("step %d,train_accuracy= %g\n"%(i,train_accuracy))
    train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})

#训练之后，使用测试集进行测试，输出最终结果
print("test_accuracy= %g"%accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1}))
f.close()
