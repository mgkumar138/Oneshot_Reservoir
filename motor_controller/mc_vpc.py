import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from backend_scripts.model import place_cells
from backend_scripts.utils import get_default_hp, saveload
import datetime

hp = get_default_hp(task='6pa',platform='gpu')
pc = place_cells(hp)

nhid = 1024
nact = 40
beta = 4
lr = 0.001  #0.001
res = 31
omitg = 0.75
modelname = 'eps_motor_controller_2h_ns_{}omg_{}_{}'.format(omitg,nhid, str(datetime.date.today()))
print(modelname)

checkpoint_path = '{}/cp.ckpt'.format(modelname)
checkpoint_dir = os.path.dirname(checkpoint_path)

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
        gns = g[goal] + np.random.normal(0,0.025, size=2)
        xyns = xy[curr] + np.random.normal(0,0.025, size=2)

        x.append(np.concatenate([gns, gns]))

        # if np.linalg.norm(g[goal],ord=2) < omitg:
        #     dircomp.append(np.zeros(2))
        #     nogidx.append(i)
        # else:
        dircomp.append(gns - xyns)
        i+=1

x = tf.cast(np.array(x),dtype=tf.float32)
dircomp = tf.cast(np.array(dircomp),dtype=tf.float32)
print('Data created . . . ')

thetaj = (2 * np.pi * np.arange(1, nact + 1)) / nact
akeys = tf.cast(np.array([np.sin(thetaj), np.cos(thetaj)]), dtype=tf.float32)
qk = tf.matmul(dircomp,akeys)
q = tf.nn.softmax(beta * qk)

# activate motor controller
allinputs = []
alloutputs = []
btstp = 5
for b in range(btstp):
    supeps = np.random.uniform(0, omitg, len(q))
    acteps = np.random.uniform(omitg, 1, len(q))
    alleps = np.concatenate([supeps, acteps])
    np.random.shuffle(alleps)
    for i in range(len(q)):
        eps = alleps[i]
        u = np.insert(x[i],2,eps)
        if eps >= omitg:
            z = q[i]
        else:
            z = np.zeros_like(q[i])
        allinputs.append(u)
        alloutputs.append(z)
print('Randomised . . . ')


# alleps = np.linspace(0,1,11)
# for eps in alleps:
#     a = np.insert(x,2,eps,axis=1)
#     if eps > omitg:
#         b = q
#     else:
#         b = np.zeros_like(q)
#     allinputs.append(a)
#     alloutputs.append(b)

allinputs = np.vstack(allinputs)
alloutputs = np.vstack(alloutputs)

''' model definition x --> q'''

class motor_controller(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.h1 = tf.keras.layers.Dense(units=nhid, activation='relu',trainable=True,
                                                use_bias=False, kernel_initializer='glorot_uniform', name='h1')
        self.h2 = tf.keras.layers.Dense(units=nhid, activation='relu',trainable=True,
                                                use_bias=False, kernel_initializer='glorot_uniform', name='h2')
        self.action = tf.keras.layers.Dense(units=nact, activation='softmax',
                                                use_bias=False, kernel_initializer='zeros', name='action')


    def call(self, x):
        # h1 = self.drp1(self.h1(self.ns(x))) # self.h1(self.ns(x)) #self.h3(self.h2(self.h1(self.ns(x))))
        # h2 = self.drp2(self.h2(h1))
        a = self.action(self.h2(self.h1(x)))
        return a


model = motor_controller()

randidx = np.random.choice(np.arange(len(allinputs)), 16, replace=False)
randidx = np.concatenate([randidx, np.arange(len(allinputs)-16,len(allinputs))])
#model.load_weights(checkpoint_path)
#ls, acc = model.evaluate(allinputs[randidx], alloutputs[randidx], verbose=2)

loss = tf.keras.losses.mean_squared_error
model.compile(run_eagerly=True,
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
    loss=loss, metrics=['accuracy'])
batch_size = 32
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
# model.save_weights(checkpoint_path.format(epoch=0))

print(allinputs.shape)
history = model.fit(allinputs, alloutputs, epochs=15, batch_size=batch_size, validation_split=0.05, shuffle=True, callbacks=[cp_callback])
model.summary()

qpred = model.predict_on_batch(allinputs[randidx])

plt.figure()
plt.subplot(221)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model mse loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')

plt.subplot(222)
plt.plot(alloutputs[randidx[0]])
plt.plot(qpred[0])
plt.show()

plt.subplot(223)
plt.plot(alloutputs[randidx[16]])
plt.plot(qpred[16])
plt.show()

plt.subplot(224)
plt.plot(alloutputs[randidx[30]])
plt.plot(qpred[30])
plt.show()

model.save(modelname)








