import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from tqdm.autonotebook import trange
import math
from models import Discriminator, Generator, Regressor, Regressor_u
import matplotlib.pyplot as plt

class SORangeGAN(keras.Model):
    def __init__(self, Y, noise_dim = 64, data_dim = 256, lambda0=2.0, lambda1=0.5, lambda2 = 0.0, kappa=-1, phi=30, reg='default', training='paper', inf_lstd = -8.0):
        super(SORangeGAN, self).__init__()
        
        self.noise_dim = noise_dim
        self.data_dim = data_dim
        self.EPSILON = 1e-7
        self.lambda0 = lambda0
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.phi = phi
        self.reg = reg
        self.training = training
        self.inf_lstd = inf_lstd
        
        self.generator = Generator(data_dim)
        self.discriminator = Discriminator()
        
        if reg == 'default':
            self.regressor = Regressor(Y.shape[1])
            
        elif reg =='uncertain':
            self.regressor = Regressor_u(Y.shape[1])
            
        else:
            self.regressor = Regressor(Y.shape[1])

        if kappa<0.0: 
            y_sorted = np.sort(np.unique(Y))
            kappa_base = abs(kappa)*np.max(y_sorted[1:] - y_sorted[0:-1])
            self.kappa = kappa_base
            
            print('kappa: %f'%(self.kappa))
        
        else:
            self.kappa = kappa
            print('kappa: %f'%(self.kappa))

    def get_balanced_batch(self, X, Y, batch_size):
        
        kappa = self.kappa
        
        batch_target_labels = np.random.uniform(low=np.min(Y),high=np.max(Y),size=[batch_size])
        
        batch_real_indx = np.zeros(batch_size, dtype=int)
        
        for j in range(batch_size):
            indx_real_in_vicinity =  np.where(np.abs(Y-batch_target_labels[j])<= kappa)[0]

            while indx_real_in_vicinity.shape[0] == 0:
                batch_target_labels[j] += np.random.uniform(low=-self.sigma,high=self.sigma)
                indx_real_in_vicinity =  np.where(np.abs(Y-batch_target_labels[j])<= kappa)[0]

            batch_real_indx[j] = np.random.choice(indx_real_in_vicinity, size=1)[0]

        X_batch = X[batch_real_indx]
        Y_batch = Y[batch_real_indx]
        
        return X_batch, Y_batch
    
    def negative_loglikelihood(self, c_target, c_mean, c_logstd):
        # Negative log likelihood
        EPSILON = 1e-7
        epsilon = (c_target - c_mean) / (tf.math.exp(c_logstd) + EPSILON)
        nll = c_logstd + 0.5 * tf.square(epsilon)
        nll = tf.reduce_sum(nll, axis=1)
        
        return tf.reduce_mean(nll)
    
    def get_regressor_train_step(self):
        @tf.function
        @tf.autograph.experimental.do_not_convert
        def regressor_train_step(X_batch, Y_batch, optimizer,optimizer2=None):
            
            loss_fn = keras.losses.MeanAbsoluteError()
            loss_fn_mse = keras.losses.MeanSquaredError()

            with tf.GradientTape() as tape:
                
                if self.reg == 'uncertain':
                    y_pred, log_std = self.regressor(X_batch)
                    L2 = loss = loss_fn_mse(Y_batch,y_pred)
                else:
                    y_pred = self.regressor(X_batch)
                    L2 = loss = loss_fn_mse(Y_batch,y_pred)
                L1 = loss_fn(Y_batch,y_pred)
                
            variables = self.regressor.trainable_weights
            gradients = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(gradients, variables))
            
            if self.reg == 'uncertain':
                with tf.GradientTape() as tape:
                    y_pred, log_std = self.regressor(X_batch)
                    loss = self.negative_loglikelihood(Y_batch,y_pred,log_std)
                variables = self.regressor.Dense_log_std.trainable_weights
                gradients = tape.gradient(loss, variables)
                optimizer2.apply_gradients(zip(gradients, variables))
                
            return loss, L1, L2
        
        return regressor_train_step
    
    
    def train_regressor(self, X_train, Y_train, X_test, Y_test, batch_size=256, train_steps=10000, lr=1e-4, balanced_training=True, early_stop_save=None):
        
        lr = keras.optimizers.schedules.ExponentialDecay(lr,decay_steps = train_steps//4, decay_rate = 0.4642, staircase=True)
        optimizer = keras.optimizers.Adam(lr,beta_1 = 0.5)
        optimizer2 = keras.optimizers.Adam(lr,beta_1 = 0.5)
        steps = trange(train_steps, desc='Training regressor Model', leave=True, ascii ="         =")
        
        validation_metric1 = keras.losses.MeanAbsoluteError()
        validation_metric2 = keras.losses.MeanSquaredError()
        
        if self.reg == 'uncertain':
                    Y_pred, log_std = self.regressor(X_test,training=False)
        else:
            Y_pred = self.regressor(X_test,training=False)
            log_std = 0.0
        m1 = validation_metric1(Y_test,Y_pred)
        m2 = validation_metric2(Y_test,Y_pred)
        m3 = self.negative_loglikelihood(Y_test,Y_pred,log_std)
        
        best = m1
        best_train = -1.0
        
        regressor_train_step = self.get_regressor_train_step()

        for step in steps:
            if balanced_training:
                X_batch,Y_batch = self.get_balanced_batch(X_train,Y_train,batch_size)
            
            else:
                ind = np.random.choice(X_train.shape[0], size=batch_size, replace=False)
                X_batch = X_train[ind]
                Y_batch = Y_train[ind]
            
            loss, L1, L2 = regressor_train_step(X_batch,Y_batch,optimizer,optimizer2)
            
            if (step+1)%50 == 0:
                if self.reg == 'uncertain':
                    Y_pred, log_std = self.regressor(X_test,training=False)
                else:
                    Y_pred = self.regressor(X_test,training=False)
                    log_std = 0.0
                m1 = validation_metric1(Y_test,Y_pred)
                m2 = validation_metric2(Y_test,Y_pred)
                m3 = self.negative_loglikelihood(Y_test,Y_pred,log_std)
                if early_stop_save and m1<=best:
                    best = m1
                    best_train = L1
                    self.regressor.save_weights(early_stop_save)

            
            steps.set_postfix_str('Train loss: %f | L2: %f | L1: %f, Validation L1: %f | L2: %f | nll: %f, lr: %f' % (loss,L2,L1,m1,m2,m3,optimizer._decayed_lr('float32')))
        print('Best Regressor Saved With: Validation_L1 = %f, Train_L1 = %f' % (best, best_train))

    def get_batch(self, X, Y, batch_size, balanced = True):
        
        #GAN batch
        kappa = self.kappa
        
        #get valid samples for a random range
        target = np.random.uniform(low=np.min(Y),high=np.max(Y))
        r = np.random.uniform(low=0.025,high=0.25)
        if target - r > 0.0:
            lb = target - r
        else:
            lb = 0.0
        
        if target + r < 1.0:
            ub = target + r
        else:
            ub = 1.0
        
        if balanced:
            X_real, Y_real = self.get_balanced_batch(X,Y,batch_size)
        else:
            ind = np.random.choice(X.shape[0],size=batch_size,replace=False)
            X_real, Y_real = X[ind],Y[ind]
        
        constraint = np.array([[lb,ub]]*batch_size)

        return X_real, Y_real, constraint
    
    def CDF(self,m,lstd,x):
        return 0.5*(1 + tf.math.erf((x-m)/tf.exp(lstd)/tf.math.sqrt(2.0)))
    
    def get_train_step(self):
        @tf.function
        @tf.autograph.experimental.do_not_convert
        def train_step(X_real, Y_real, condition, d_optimizer, g_optimizer):
            
            binary_cross_entropy_loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
            batch_size = X_real.shape[0]
            
            z = tf.random.normal([batch_size, self.noise_dim])
            X_fake = self.generator(z, condition)
            
            #Discriminator Training
            with tf.GradientTape() as tape:
                d_real = self.discriminator(X_real,condition)
                d_loss_real = binary_cross_entropy_loss_fn(tf.ones_like(d_real),d_real)
                loss = d_loss_real
                
            variables = self.discriminator.trainable_weights
            gradients = tape.gradient(loss, variables)
            d_optimizer.apply_gradients(zip(gradients, variables))

            with tf.GradientTape() as tape:
                d_fake = self.discriminator(X_fake,condition)
                d_loss_fake = binary_cross_entropy_loss_fn(tf.zeros_like(d_fake),d_fake)
                loss = d_loss_fake
                
            variables = self.discriminator.trainable_weights
            gradients = tape.gradient(loss, variables)
            d_optimizer.apply_gradients(zip(gradients, variables))
            
            z = tf.random.normal([batch_size, self.noise_dim])
            #Generator Training
            with tf.GradientTape() as tape:
                x_fake_train = self.generator(z, condition)
                d_fake = self.discriminator(x_fake_train, condition)
                g_loss = binary_cross_entropy_loss_fn(tf.ones_like(d_fake),d_fake)
                
                # Range Loss
                if self.reg == 'uncertain' and self.training == 'default':
                    y,lstd = self.regressor(x_fake_train,training=False)
                    range_loss = tf.reduce_mean(-(self.CDF(y,lstd,condition[0,1]) - self.CDF(y,lstd,condition[0,0])))

                elif self.reg == 'uncertain':
                    y,_ = self.regressor(x_fake_train,training=False)
                    cond_probs = (tf.math.sigmoid(self.phi*(y-condition[0,0])) - tf.math.sigmoid(self.phi*(y-condition[0,1]))) * tf.where((y - condition[:,0:1])*(y-condition[:,1:2])>0,1.0,0.0) + tf.where((y - condition[:,0:1])*(y-condition[:,1:2])>0,0.0,1.0)
                    range_loss = -tf.reduce_mean(cond_probs)
                    
                elif self.training == 'default':
                    y = self.regressor(x_fake_train,training=False)
                    range_loss = tf.reduce_mean(-(self.CDF(y,self.inf_lstd,condition[0,1]) - self.CDF(y,self.inf_lstd,condition[0,0])))

                else:
                    y = self.regressor(x_fake_train,training=False)
                    cond_probs = (tf.math.sigmoid(self.phi*(y-condition[0,0])) - tf.math.sigmoid(self.phi*(y-condition[0,1]))) * tf.where((y - condition[:,0:1])*(y-condition[:,1:2])>0,1.0,0.0) + tf.where((y - condition[:,0:1])*(y-condition[:,1:2])>0,0.0,1.0)
                    range_loss = -tf.reduce_mean(cond_probs)
    
                # Uniformity Loss
                good_samples = tf.gather(y,tf.where((y - condition[:,0:1])*(y-condition[:,1:2])<=0)[:,0])
                test_points = tf.random.uniform([10],minval=condition[0,0],maxval=condition[0,1])
                offsets = (good_samples - test_points)

                uniformity_loss = 0.0
                uniformity_loss += tf.reduce_sum(tf.math.abs(tf.reduce_sum(tf.where(offsets>0,offsets,0.0),0)/tf.maximum(tf.reduce_sum(tf.where(offsets>0,1.0,0.0),0),1) + test_points - (test_points+condition[0,1])/2))
                uniformity_loss += tf.reduce_sum(tf.math.abs(tf.reduce_sum(tf.where(offsets<0,offsets,0.0),0)/tf.maximum(tf.reduce_sum(tf.where(offsets<0,1.0,0.0),0),1) + test_points - (test_points+condition[0,0])/2))
                
                # Normalize Loss Based On The Range Length
                uniformity_loss = uniformity_loss/(condition[0,1]-condition[0,0])
                

                loss_total = g_loss + self.lambda1 * range_loss + self.lambda2 * uniformity_loss
                
            variables = self.generator.trainable_weights
            gradients = tape.gradient(loss_total, variables)
            g_optimizer.apply_gradients(zip(gradients, variables))
            
            return d_loss_real, d_loss_fake, g_loss, range_loss, uniformity_loss
        return train_step
    
    def train(self, X, Y, train_steps=10000, batch_size=32, disc_lr=1e-4, gen_lr=1e-4):
        
        disc_lr = keras.optimizers.schedules.ExponentialDecay(disc_lr,decay_steps = 2*train_steps//10, decay_rate = 0.8, staircase=True)
        gen_lr = keras.optimizers.schedules.ExponentialDecay(gen_lr,decay_steps = train_steps//10, decay_rate = 0.8, staircase=True)
        
        g_optimizer = keras.optimizers.Adam(gen_lr,beta_1 = 0.5)

        d_optimizer = keras.optimizers.Adam(disc_lr,beta_1 = 0.5)

        conds = []
        steps = trange(train_steps, desc='Training', leave=True, ascii ="         =")

        train_step = self.get_train_step()
        regressor_train_step = self.get_regressor_train_step()

        for step in steps:
            X_real, Y_real, condition = self.get_batch(X, Y, batch_size)
            conds.append(np.abs(condition[0,0]-condition[0,1]))
            X_real, Y_real, condition = tf.cast(X_real, tf.float32), tf.cast(Y_real, tf.float32), tf.cast(condition, tf.float32)
            d_loss_real, d_loss_fake, g_loss, range_loss, uniformity_loss = train_step(X_real, Y_real, condition, d_optimizer,g_optimizer)

            log_mesg = "%d: [D] real %+.7f fake %+.7f lr %+.7f" % (step+1, d_loss_real, d_loss_fake, d_optimizer._decayed_lr('float32'))
            log_mesg = "%s  [G] fake %+.7f lr %+.7f [C] rl %+.7f [U] ul %+.7f" % (log_mesg, g_loss, g_optimizer._decayed_lr('float32'), range_loss,uniformity_loss)
            
            steps.set_postfix_str(log_mesg)
        conds = np.array(conds)
        
class MORangeGAN(keras.Model):
    def __init__(self, Y, noise_dim = 64, data_dim = 256, lambda1=0.5, lambda2 = 0.0, kappa=-1, phi=50):
        super(MORangeGAN, self).__init__()
        
        self.noise_dim = noise_dim
        self.data_dim = data_dim
        self.EPSILON = 1e-7
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.phi = phi

        self.generator = Generator(data_dim)
        self.discriminator = Discriminator()
        self.regressor = Regressor(Y.shape[1])


        if kappa<0.0: 
            y_sorted = np.sort(np.unique(Y))
            kappa_base = abs(kappa)*np.max(y_sorted[1:] - y_sorted[0:-1])
            self.kappa = kappa_base
            
            print('kappa: %f'%(self.kappa))
        
        else:
            self.kappa = kappa
            print('kappa: %f'%(self.kappa))

        # discretize the space and organize data for faster balanced batch selection    
        A = np.linspace(0.0,1.0,21)+0.025
        a,b = np.meshgrid(A[:-1],A[:-1])
        ys = np.concatenate([a.reshape([-1,1]),b.reshape([-1,1])],1)
        self.ys = []
        for i in range(400):
            if np.where(np.sum(np.abs(Y-ys[i]) <= self.kappa,-1) == 2)[0].shape[0] != 0:
                self.ys.append(ys[i])
        self.ys = np.array(self.ys)
        

    def get_balanced_batch(self, X, Y, batch_size):
        
        kappa = self.kappa
        idx = np.random.choice(self.ys.shape[0],size=batch_size,replace=False)
        batch_target_labels = self.ys[idx]
        
        batch_real_indx = np.zeros(batch_size, dtype=int)
        
        for j in range(batch_size):
            indx_real_in_vicinity = np.where(np.sum(np.abs(Y-batch_target_labels[j]) <= kappa,-1) == 2)[0]
            
            while indx_real_in_vicinity.shape[0] == 0:
                batch_target_labels[j] = self.ys[np.random.choice(self.ys.shape[0])]
                indx_real_in_vicinity =  np.where(np.sum(np.abs(Y-batch_target_labels[j]) <= kappa,-1) == 2)[0]

            batch_real_indx[j] = np.random.choice(indx_real_in_vicinity, size=1)

        X_batch = X[batch_real_indx]
        Y_batch = Y[batch_real_indx]
        
        return X_batch, Y_batch

    def get_regressor_train_step(self):
        @tf.function
        @tf.autograph.experimental.do_not_convert
        def regressor_train_step(X_batch, Y_batch, optimizer):
            
            loss_fn = keras.losses.MeanAbsoluteError()
            loss_fn_mse = keras.losses.MeanSquaredError()

            with tf.GradientTape() as tape:
                y_pred = self.regressor(X_batch)
                loss = loss_fn_mse(Y_batch,y_pred)
    
                L1 = loss_fn(Y_batch,y_pred)
                
            variables = self.regressor.trainable_weights
            gradients = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(gradients, variables))
            
            return loss, L1
        
        return regressor_train_step
    
    
    def train_regressor(self, X_train, Y_train, X_test, Y_test, batch_size=256, train_steps=10000, lr=1e-4, balanced_training=True, early_stop_save=None):
        
        lr = keras.optimizers.schedules.ExponentialDecay(lr,decay_steps = train_steps//4, decay_rate = 0.4642, staircase=True)
        optimizer = keras.optimizers.Adam(lr,beta_1 = 0.5)
        steps = trange(train_steps, desc='Training regressor Model', leave=True, ascii ="         =")
        
        validation_metric1 = keras.losses.MeanAbsoluteError()
        validation_metric2 = keras.losses.MeanSquaredError()
        
        Y_pred = self.regressor(X_test)
        m1 = validation_metric1(Y_test,Y_pred)
        m2 = validation_metric2(Y_test,Y_pred)
        
        best = m1
        best_train = -1.0
        
        regressor_train_step = self.get_regressor_train_step()

        for step in steps:
            if balanced_training:
                X_batch,Y_batch = self.get_balanced_batch(X_train,Y_train,batch_size)
            
            else:
                ind = np.random.choice(X_train.shape[0], size=batch_size, replace=False)
                X_batch = X_train[ind]
                Y_batch = Y_train[ind]
            
            loss, L1 = regressor_train_step(X_batch,Y_batch,optimizer)
            
            if (step+1)%50 == 0:
                Y_pred = self.regressor(X_test,training=False)
                m1 = validation_metric1(Y_test,Y_pred)
                m2 = validation_metric2(Y_test,Y_pred)
                if early_stop_save and m1<=best:
                    best = m1
                    best_train = L1
                    self.regressor.save_weights(early_stop_save)

            
            steps.set_postfix_str('Train L2: %f | L1: %f, Validation L1: %f | L2: %f, lr: %f' % (loss,L1,m1,m2,optimizer._decayed_lr('float32')))
        print('Best Regressor Saved With: Validation_L1 = %f, Train_L1 = %f' % (best, best_train))

    def get_batch(self, X, Y, batch_size,balanced = True):
        
        #GAN batch
        kappa = self.kappa
        
        #get a random range
        target = np.random.uniform(low=np.min(Y),high=np.max(Y),size=2)
        r = np.random.uniform(low=0.025,high=0.25,size=2)
        lb = np.maximum(target - r, 0.0)
        ub = np.minimum(target + r, 1.0)
        
        
        if balanced:        
            X_real, Y_real = self.get_balanced_batch(X,Y,batch_size)
        else:
            ind = np.random.choice(X.shape[0],size=batch_size,replace=False)
            X_real, Y_real = X[ind],Y[ind]
            
        constraint = np.array([[lb[0],ub[0],lb[1],ub[1]]]*batch_size)

        return X_real, Y_real, constraint

    def get_train_step(self):
        @tf.function
        @tf.autograph.experimental.do_not_convert
        def train_step(X_real, Y_real, condition, d_optimizer, g_optimizer):
            
            binary_cross_entropy_loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
            batch_size = X_real.shape[0]
            
            z = tf.random.normal([batch_size, self.noise_dim])
            X_fake = self.generator(z, condition)
            
            #Discriminator Training
            with tf.GradientTape() as tape:
                d_real = self.discriminator(X_real,condition)
                d_loss_real = binary_cross_entropy_loss_fn(tf.ones_like(d_real),d_real)
                loss = d_loss_real
                
            variables = self.discriminator.trainable_weights
            gradients = tape.gradient(loss, variables)
            d_optimizer.apply_gradients(zip(gradients, variables))

            with tf.GradientTape() as tape:
                d_fake = self.discriminator(X_fake,condition)
                d_loss_fake = binary_cross_entropy_loss_fn(tf.zeros_like(d_fake),d_fake)
                loss = d_loss_fake
                
            variables = self.discriminator.trainable_weights
            gradients = tape.gradient(loss, variables)
            d_optimizer.apply_gradients(zip(gradients, variables))
            
            z = tf.random.normal([batch_size, self.noise_dim])
            
            #Generator Training
            with tf.GradientTape() as tape:
                x_fake_train = self.generator(z, condition)
                d_fake = self.discriminator(x_fake_train, condition)
                g_loss = binary_cross_entropy_loss_fn(tf.ones_like(d_fake),d_fake)

                y = self.regressor(x_fake_train,training=False)
                c = condition
                cond_probs_1 = (tf.math.sigmoid(self.phi*(y[:,0:1]-c[0,0])) - tf.math.sigmoid(self.phi*(y[:,0:1]-c[0,1]))) * tf.where((y[:,0:1] - c[:,0:1])*(y[:,0:1]-c[:,1:2])>0,1.0,0.0) + tf.where((y[:,0:1] - c[:,0:1])*(y[:,0:1]-c[:,1:2])>0,0.0,1.0)
                cond_probs_2 = (tf.math.sigmoid(self.phi*(y[:,1:2]-c[0,2])) - tf.math.sigmoid(self.phi*(y[:,1:2]-c[0,3]))) * tf.where((y[:,1:2] - c[:,2:3])*(y[:,1:2]-c[:,3:4])>0,1.0,0.0) + tf.where((y[:,1:2] - c[:,2:3])*(y[:,1:2]-c[:,3:4])>0,0.0,1.0)

                range_loss1 = -tf.reduce_mean(cond_probs_1)
                range_loss2 = -tf.reduce_mean(cond_probs_2)
                range_loss = (range_loss1 + range_loss2)/2.0
                
                aa = tf.gather(y[:,0:1],tf.where((y[:,0:1] - condition[:,0:1])*(y[:,0:1]-condition[:,1:2])<=0)[:,0])
                test = tf.random.uniform([10],minval=c[0,0],maxval=c[0,1])
                bb = (aa - test)


                uniformity_loss_1 = 0.0
                uniformity_loss_1 += tf.reduce_sum(tf.math.abs(tf.reduce_sum(tf.where(bb>0,bb,0.0),0)/tf.maximum(tf.reduce_sum(tf.where(bb>0,1.0,0.0),0),1) + test - (test+c[0,1])/2))
                uniformity_loss_1 += tf.reduce_sum(tf.math.abs(tf.reduce_sum(tf.where(bb<0,bb,0.0),0)/tf.maximum(tf.reduce_sum(tf.where(bb<0,1.0,0.0),0),1) + test - (test+c[0,0])/2))
                uniformity_loss_1 = uniformity_loss_1/(c[0,1]-c[0,0])

                aa = tf.gather(y[:,1:2],tf.where((y[:,1:2] - condition[:,2:3])*(y[:,1:2]-condition[:,3:4])<=0)[:,0])
                test = tf.random.uniform([10],minval=c[0,2],maxval=c[0,3])
                bb = (aa - test)


                uniformity_loss_2 = 0.0
                uniformity_loss_2 += tf.reduce_sum(tf.math.abs(tf.reduce_sum(tf.where(bb>0,bb,0.0),0)/tf.maximum(tf.reduce_sum(tf.where(bb>0,1.0,0.0),0),1) + test - (test+c[0,3])/2))
                uniformity_loss_2 += tf.reduce_sum(tf.math.abs(tf.reduce_sum(tf.where(bb<0,bb,0.0),0)/tf.maximum(tf.reduce_sum(tf.where(bb<0,1.0,0.0),0),1) + test - (test+c[0,2])/2))
                uniformity_loss_2 = uniformity_loss_2/(c[0,1]-c[0,0])
                
                uniformity_loss = (uniformity_loss_1 + uniformity_loss_2)/2.0


                loss_total = g_loss + self.lambda1 * range_loss + self.lambda2 * uniformity_loss
            variables = self.generator.trainable_weights
            gradients = tape.gradient(loss_total, variables)
            g_optimizer.apply_gradients(zip(gradients, variables))
            
            return d_loss_real, d_loss_fake, g_loss, range_loss, uniformity_loss
        return train_step
        
    def train(self, X, Y, train_steps=10000, batch_size=32, disc_lr=1e-4, gen_lr=1e-4):
        
        disc_lr = keras.optimizers.schedules.ExponentialDecay(disc_lr,decay_steps = 2*train_steps//10, decay_rate = 0.8, staircase=True)
        gen_lr = keras.optimizers.schedules.ExponentialDecay(gen_lr,decay_steps = train_steps//10, decay_rate = 0.8, staircase=True)
        
        g_optimizer = keras.optimizers.Adam(gen_lr,beta_1 = 0.5)

        d_optimizer = keras.optimizers.Adam(disc_lr,beta_1 = 0.5)
        
        conds = []
        steps = trange(train_steps, desc='Training', leave=True, ascii ="         =")

        train_step = self.get_train_step()

        for step in steps:
            X_real, Y_real, condition = self.get_batch(X, Y, batch_size)
            conds.append(np.abs(condition[0,0]-condition[0,1]))
            X_real, Y_real, condition = tf.cast(X_real, tf.float32), tf.cast(Y_real, tf.float32), tf.cast(condition, tf.float32)
            d_loss_real, d_loss_fake, g_loss, range_loss, uniformity_loss = train_step(X_real, Y_real, condition, d_optimizer,g_optimizer)
            log_mesg = "%d: [D] real %+.7f fake %+.7f lr %+.7f" % (step+1, d_loss_real, d_loss_fake, d_optimizer._decayed_lr('float32'))
            log_mesg = "%s  [G] fake %+.7f lr %+.7f [C] rl %+.7f [U] ul %+.7f" % (log_mesg, g_loss, g_optimizer._decayed_lr('float32'), range_loss,uniformity_loss)
            
            steps.set_postfix_str(log_mesg)
        conds = np.array(conds)

        print('min_l: %f, mean_l: %f, max_l: %f' % (conds.min(),conds.mean(),conds.max()))
        plt.hist(conds)