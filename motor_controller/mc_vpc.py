import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from backend_scripts.model import place_cells
from backend_scripts.utils import get_default_hp, saveload
import datetime
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

hp = get_default_hp(task='wkm',platform='gpu')
pc = place_cells(hp)

nhid = 1024
nact = 40
beta = 4
lr = 0.001  #0.001
res = 51
omitg = 0.025
modelname = 'motor_controller_h_{}omg_{}_{}'.format(omitg,nhid, str(datetime.date.today()))
print(modelname)

pos = np.linspace(-0.8,0.8,res)
xx, yy = np.meshgrid(pos, pos)
g = np.concatenate([xx.reshape([res**2,1]),yy.reshape([res**2,1])],axis=1)
xy = np.copy(g)
x = []
dircomp = []
nogidx = []
i=0
for goal in range(res**2):
    for curr in range(res**2):
        x.append(np.concatenate([g[goal], xy[curr]]))

        if np.linalg.norm(g[goal],ord=2) < omitg:
            dircomp.append(np.zeros(2))
            nogidx.append(i)
        else:
            dircomp.append(g[goal] - xy[curr])
        i+=1

x = tf.cast(np.array(x),dtype=tf.float32)
dircomp = tf.cast(np.array(dircomp),dtype=tf.float32)
print('Data created . . . ')

thetaj = (2 * np.pi * np.arange(1, nact + 1)) / nact
akeys = tf.cast(np.array([np.sin(thetaj), np.cos(thetaj)]), dtype=tf.float32)
qk = tf.matmul(dircomp,akeys)
# q = np.zeros_like(qk)
# q[np.arange(len(q)), np.argmax(qk,axis=1)] = 1
q = tf.nn.softmax(beta * qk)
# q = q.numpy()
#
# idx = np.max(q,axis=1)<omitg
# q[idx] = 0


''' model definition x --> q'''

class motor_controller(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.h1 = tf.keras.layers.Dense(units=nhid, activation='relu',trainable=True,
                                                use_bias=False, kernel_initializer='glorot_uniform', name='h1')

        #self.drp1 = tf.keras.layers.Dropout(0.5)
        self.h2 = tf.keras.layers.Dense(units=nhid, activation='relu',trainable=True,
                                                use_bias=False, kernel_initializer='glorot_uniform', name='h2')
        #
        # self.h3 = tf.keras.layers.Dense(units=nhid, activation='relu',trainable=True,
        #                                         use_bias=False, kernel_initializer='glorot_uniform', name='h3')
        # self.drp2 = tf.keras.layers.Dropout(0.5)
        self.action = tf.keras.layers.Dense(units=nact, activation='softmax',
                                                use_bias=False, kernel_initializer='zeros', name='action')
        #self.ns = tf.keras.layers.GaussianNoise(0.025)

    def call(self, x):
        # h1 = self.drp1(self.h1(self.ns(x))) # self.h1(self.ns(x)) #self.h3(self.h2(self.h1(self.ns(x))))
        # h2 = self.drp2(self.h2(h1))
        a = self.action(self.h2(self.h1(x)))
        return a


model = motor_controller()

#loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
loss = tf.keras.losses.mean_squared_error

model.compile(run_eagerly=True,
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
    loss=loss, metrics=['accuracy'])

print(x.shape)
history = model.fit(x, q, epochs=50, batch_size=32, validation_split=0.05, shuffle=True)
model.summary()

qpred = model.predict_on_batch(x[:32])

plt.figure()
plt.subplot(221)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model mse loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')

plt.subplot(222)
plt.plot(q[0])
plt.plot(qpred[0])
plt.show()

plt.subplot(223)
plt.plot(q[16])
plt.plot(qpred[16])
plt.show()

plt.subplot(224)
plt.plot(q[30])
plt.plot(qpred[30])
plt.show()

#hw, aw = model.trainable_weights
#model.save_weights('motor_controller_weights_{}_{}'.format(nhid, str(datetime.date.today())))
#saveload('save','mc_w_{}_{}'.format(nhid, str(datetime.date.today())),[hw, aw])
model.save(modelname)








