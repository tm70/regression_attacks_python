import tensorflow.compat.v1 as tf
import numpy as np

# CW attack adapted to regression
class regCW:
    def __init__(self, sess, model, shape, constant=None,
                 batch_size=1, initial_const=1e-3,
                 learning_rate=1e-2, binary_search_steps=9, max_iterations=10000,
                 abort_early=True):
        self.sess = sess
        self.batch_size = batch_size
        self.initial_const = initial_const
        self.LEARNING_RATE = learning_rate
        self.BINARY_SEARCH_STEPS = binary_search_steps
        self.MAX_ITERATIONS = max_iterations
        self.ABORT_EARLY = abort_early
        
        self.repeat = binary_search_steps >= 10
        
        self.I_KNOW_WHAT_I_AM_DOING_AND_WANT_TO_OVERRIDE_THE_PRESOFTMAX_CHECK = False
        
        modifier = tf.Variable(np.zeros((1,shape), dtype=np.float32))
        
        self.timg = tf.Variable(np.zeros((1,shape)), dtype=tf.float32)
        self.tout = tf.Variable(np.zeros(batch_size), dtype=tf.float32)
        self.const = tf.Variable(np.zeros(batch_size), dtype=tf.float32)
        
        self.assign_timg = tf.placeholder(tf.float32, (1,shape))
        self.assign_tout = tf.placeholder(tf.float32, batch_size)
        self.assign_const = tf.placeholder(tf.float32, [batch_size])
        
        self.newimg = modifier + self.timg
        
        self.output = model(self.newimg)
        
        # minimise perturbation
        self.loss1 = tf.reduce_sum(tf.square(self.newimg - self.timg), 1)
        
        # minimise distance to target
        # would need to minimax for untargeted?
        self.loss2 = tf.reduce_sum(tf.square(self.newimg - self.tout))
        
        if constant == None:
            self.l2dist = self.loss1 + self.const * self.loss2
        else:
            self.l2dist = self.loss1 + constant * self.loss2
        
        self.loss = tf.reduce_sum(self.l2dist)
        
        # Setup the adam optimizer and keep track of variables we're creating
        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
        self.train = optimizer.minimize(self.loss, var_list=[modifier])
        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]

        # these are the variables to initialize when we run
        self.setup = []
        self.setup.append(self.timg.assign(self.assign_timg))
        self.setup.append(self.tout.assign(self.assign_tout))
        self.setup.append(self.const.assign(self.assign_const))
        
        self.init = tf.variables_initializer(var_list=[modifier]+new_vars)
    
    def attack(self, data, targets):
        r = []
        print('go up to',len(data))
        for i in range(0,len(data), self.batch_size):
            print('tick',i)
            r.extend(self.attack_batch(data[i:i+self.batch_size], targets[i:i+self.batch_size]))
        
        return np.array(r)
    
    def attack_batch(self, data, targets):
        batch_size = self.batch_size
        
        # set the lower and upper bounds accordingly
        lower_bound = np.zeros(batch_size)
        CONST = np.ones(batch_size)*self.initial_const
        upper_bound = np.ones(batch_size)*1e10

        # the best l2, score, and image attack
        o_bestl2 = [1e10]*batch_size
        o_bestscore = [-1]*batch_size
        o_bestattack = [np.zeros(data[0].shape)]*batch_size
        
        for outer_step in range(self.BINARY_SEARCH_STEPS):
            print("o_best", o_bestl2)
            # completely reset adam's internal state.
            self.sess.run(self.init)
            batch = data[:batch_size]
            batchlab = targets[:batch_size]
    
            bestl2 = [1e10]*batch_size
            bestscore = [-1]*batch_size

            # The last iteration (if we run many steps) repeat the search once.
            if self.repeat == True and outer_step == self.BINARY_SEARCH_STEPS-1:
                CONST = upper_bound

            # set the variables so that we don't have to send them over again
            self.sess.run(self.setup, {self.assign_timg: batch,
                                       self.assign_tout: batchlab,
                                       self.assign_const: CONST})
            
            prev = np.inf
            for iteration in range(self.MAX_ITERATIONS):
                # perform the attack 
                _, l, l2s, scores, nimg = self.sess.run([self.train, self.loss, 
                                                         self.l2dist, self.output, 
                                                         self.newimg])

                if np.all(scores>=-.0001) and np.all(scores <= 1.0001):
                    if np.allclose(np.sum(scores,axis=1), 1.0, atol=1e-3):
                        if not self.I_KNOW_WHAT_I_AM_DOING_AND_WANT_TO_OVERRIDE_THE_PRESOFTMAX_CHECK:
                            raise Exception("The output of model.predict should return the pre-softmax layer. It looks like you are returning the probability vector (post-softmax). If you are sure you want to do that, set attack.I_KNOW_WHAT_I_AM_DOING_AND_WANT_TO_OVERRIDE_THE_PRESOFTMAX_CHECK = True")
                
                # print out the losses every 10%
                if iteration%(self.MAX_ITERATIONS//10) == 0:
                    print(iteration,self.sess.run((self.loss,self.loss1,self.loss2)))

                # check if we should abort search if we're getting nowhere.
                if self.ABORT_EARLY and iteration%(self.MAX_ITERATIONS//10) == 0:
                    if l > prev*.9999:
                        break
                    prev = l

                # adjust the best result found so far
                for e,(l2,sc,ii) in enumerate(zip(l2s,scores,nimg)):
                    if l2 < bestl2[e]:# and compare(sc, np.argmax(batchlab[e])):
                        bestl2[e] = l2
                        bestscore[e] = np.argmax(sc)
                    if l2 < o_bestl2[e]:# and compare(sc, np.argmax(batchlab[e])):
                        o_bestl2[e] = l2
                        o_bestscore[e] = np.argmax(sc)
                        o_bestattack[e] = ii

            # adjust the constant as needed
            for e in range(batch_size):
                if  bestscore[e] != -1:
                    # success, divide const by two
                    upper_bound[e] = min(upper_bound[e],CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e])/2
                else:
                    # failure, either multiply by 10 if no solution found yet
                    #          or do binary search with the known upper bound
                    lower_bound[e] = max(lower_bound[e],CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e])/2
                    else:
                        CONST[e] *= 10

        # return the best solution found
        o_bestl2 = np.array(o_bestl2)
        return o_bestattack

# FGSM attack in tensorflow
def attackFGSM(model, loss_object, x, y, epsilon=1e-3):
    with tf.GradientTape() as tape:
        tape.watch(x)
        prediction = model(x)
        loss = loss_object(y, prediction)
    gradient = tape.gradient(loss, x)
    signed_grad = tf.sign(gradient)
    return x + epsilon * signed_grad
        