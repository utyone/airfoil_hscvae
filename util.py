import logging
import os
import numpy as np
import tensorflow as tf
import sys
from sklearn.model_selection import train_test_split


def create_log(name):
    """Logging."""
    if os.path.exists(name):
        os.remove(name)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # handler for logger file
    handler1 = logging.FileHandler(name)
    handler1.setFormatter(logging.Formatter("H1, %(asctime)s %(levelname)8s %(message)s"))
    # handler for standard output
    handler2 = logging.StreamHandler()
    handler2.setFormatter(logging.Formatter("H1, %(asctime)s %(levelname)8s %(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def mnist_loader():
    from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
    mnist = read_data_sets('MNIST_data', one_hot=True)
    n_sample = mnist.train.num_examples
    return mnist, n_sample


def shape_2d(_x, batch_size):
    _x = _x.reshape(batch_size, -1)
    return np.expand_dims(_x, 3)


def train(model, epoch, lat, save_path="./", mode="conditional", input_image=False):
    """ Train model based on mini-batch of input data.

    :param model:
    :param epoch:
    :param save_path:
    :param mode: conditional, supervised, unsupervised
    :param input_image: True if use CNN for top of the model
    :return:
    """
    # Data preparation
    import random
    
    array= np.load("./data/NACA_coords.npy")
    CLCD = np.load("./data/NACA_perfs.npy")
    
    CL = np.array([CLCD[:,1]]).T
    CD = np.array([CLCD[:,2]]).T
    array = array[np.tile((CL>0.5), 496)].reshape(-1,496)
    CD = CD[CL>0.5].reshape(-1,1)
    CL = CL[CL>0.5].reshape(-1,1)
    
    array = array[np.tile((CL<1.2), 496)].reshape(-1,496)
    CD = CD[CL<1.2].reshape(-1,1)
    CL = CL[CL<1.2].reshape(-1,1)
    print("size NACA", array.shape)
    
    # add Jukovski
    
    array_juko = np.load("./data/juko_array.npy")
    CL_juko = np.load("./data/juko_CLCD.npy")
    CL_juko = np.array([CL_juko[:,1]]).T
    array = np.append(array, array_juko[::2], axis=0)
    CL = np.append(CL, CL_juko[::2], axis=0)

    print("size All", array.shape, CL.shape)

    X_mean = np.mean(array, axis=0).reshape(1,-1)
    X_std = np.std(array, axis=0).reshape(1,-1)

    y_mean = np.mean(CL, axis=0).reshape(1,-1)
    y_std = np.std(CL, axis=0).reshape(1,-1)
    X_std[X_std==0.0] = 1.0
    y_std[y_std==0.0] = 1.0
    array = (array - X_mean)/X_std
    CL = (CL - y_mean)/y_std
    
    np.save("X_mean.npy",X_mean)
    np.save("X_std.npy",X_std)
    
    # Make Train and Test data
    X_train, X_test, y_train, y_test = train_test_split(array, CL, train_size=0.9)
    
    #X_train = array
    #X_test = array[0,:]
    #y_train = CL
    #y_test = CL[0,:]
    

    n = len(X_train)
    n_iter = int(n / model.batch_size)
    print(X_train.shape)
    
    # logger
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    logger = create_log(save_path+"log")
    logger.info("train: data size(%i), batch num(%i), batch size(%i)" % (n, n_iter, model.batch_size))
    result = []
    # Initializing the tensor flow variables
    model.sess.run(tf.compat.v1.global_variables_initializer())
    loss_hist=[]
    for _e in range(epoch):
        _result = []
        perm = np.random.permutation(n)
        for _b in range(n_iter):
            # train
            _x = X_train[perm[_b*model.batch_size: (_b+1)*model.batch_size]]
            _y = y_train[perm[_b*model.batch_size: (_b+1)*model.batch_size]]
            _x = shape_1d(_x, model.batch_size) if input_image else _x

            if mode in ["conditional", "unsupervised"]:  # conditional unsupervised model
                feed_val = [model.summary, model.loss, model.re_loss, model.latent_loss, model.train]
                feed_dict = {model.x: _x, model.y: _y} if mode == "conditional" else {model.x: _x}
                summary, loss, re_loss, latent_loss, _ = model.sess.run(feed_val, feed_dict=feed_dict)
                __result = [loss, re_loss, latent_loss]
            elif mode == "supervised":  # supervised model
                feed_val = [model.summary, model.loss, model.accuracy, model.train]
                feed_dict = {model.x: _x, model.y: _y, model.is_training: True}
                summary, loss, acc, _ = model.sess.run(feed_val, feed_dict=feed_dict)
                __result = [loss, acc]
            else:
                sys.exit("unknown mode !")
            _result.append(__result)
            model.writer.add_summary(summary, int(_b + _e * model.batch_size))

        # validation
        if mode == "supervised":  # supervised model
            _x = X_test
            _y = y_test
            feed_dict = {model.x: _x, model.y: _y, model.is_training: False}
            loss, acc = model.sess.run([model.loss, model.accuracy], feed_dict=feed_dict)
            _result = np.append(np.mean(_result, 0), [loss, acc])
            logger.info("epoch %i: acc %0.3f, loss %0.3f, train acc %0.3f, train loss %0.3f"
                        % (_e, acc, loss, _result[1], _result[0]))
        else:
            _result = np.mean(_result, 0)
            logger.info("epoch %i: loss %0.3f, re loss %0.3f, latent loss %0.3f"
                        % (_e, _result[0], _result[1], _result[2]))
            loss_hist.append(_result[1])

        result.append(_result)
        if _e % 50 == 0:
            model.saver.save(model.sess, "%s/progress-%i-model.ckpt" % (save_path, _e))
            #np.savez("%s/progress-%i-acc.npz" % (save_path, _e), loss=np.array(result),learning_rate=model.learning_rate, epoch=epoch, batch_size=model.batch_size,clip=model.max_grad_norm)
    model.saver.save(model.sess, "%s/model.ckpt" % save_path)
    np.savez("%s/acc.npz" % save_path, loss=np.array(result), learning_rate=model.learning_rate, epoch=epoch,batch_size=model.batch_size, clip=model.max_grad_norm)

    rec_train = np.zeros(X_train.shape)
    #rec_test = np.zeros(X_test.shape)
    for i in range( np.int(X_train.shape[0] / model.batch_size) ):
        rec_train[i*model.batch_size:(i+1)*model.batch_size] = model.reconstruct(X_train[i*model.batch_size:(i+1)*model.batch_size], y_train[i*model.batch_size:(i+1)*model.batch_size])
    #for i in range( np.int(X_test.shape[0] / model.batch_size) ):
    #    rec_test[i*model.batch_size:(i+1)*model.batch_size] = model.reconstruct(X_test[i*model.batch_size:(i+1)*model.batch_size], y_test[i*model.batch_size:(i+1)*model.batch_size])

    print("shape", X_train.shape, y_train.shape)
    ##  z_var
    c = model.encode3(X_train, y_train)
    #c = c/np.linalg.norm(c, axis=1, ord=2).reshape(-1,1)
    np.savetxt("LatentDist3_{}_train.csv".format(lat), np.append(c, y_train*y_std+y_mean, axis=1 ), delimiter=",")

    ##  z after sampling
    c = model.encode(X_train, y_train)
    c = c/np.linalg.norm(c, axis=1, ord=2).reshape(-1,1)
    np.savetxt("LatentDist_{}_train.csv".format(lat), np.append(c.reshape([-1, lat]), y_train*y_std+y_mean, axis=1 ), delimiter=",")
    ##  z_mean
    c = model.encode2(X_train, y_train)
    c = c/np.linalg.norm(c, axis=1, ord=2).reshape(-1,1)
    np.savetxt("LatentDist2_{}_train.csv".format(lat), np.append(c.reshape([-1, lat]), y_train*y_std+y_mean, axis=1 ), delimiter=",")
    
    c_naca_center= c[0:1,:].mean(axis=0)
    c_naca_center = c_naca_center/ np.linalg.norm(c_naca_center, ord=2)
    c_juko_center= c[0:,:].mean(axis=0)
    c_juko_center = c_juko_center/ np.linalg.norm(c_juko_center, ord=2)
    print(c_naca_center, c_juko_center)
    np.savetxt("LatentDist_{}_train_center.csv".format(lat), np.append(c_naca_center.reshape([-1, lat]), c_juko_center.reshape([-1, lat]), axis=0 ), delimiter=",")
    #import tensorflow_probability as tfp
    #c_sample = tfp.distributions.VonMisesFisher(mean_direction=c, concentration = 1000).sample()
    #c_sample = model.sess.run(c_sample)
    
    
    #model = pickle.load("model_16.pickle")
    
    numLabel = 1
    #y_gen = np.arange(-1,1,0.05)

    y_gen = np.arange(0.5,1.199,0.007).reshape(-1,1)
    y_gen_stded = (y_gen-y_mean)/y_std
    #y_gen = np.random.randn(100,numLabel)
        
    for trial in range(0):
        z = np.random.randn(1,lat).astype(np.float32)
        z = np.tile(z, (100,1))
        z = z/ np.linalg.norm(z, axis=1, ord=2).reshape(100,1)
        b = model.decode(y_gen_stded, z=z)
        #y_gen = (y_gen*y_std)+y_std
        b = (b*X_std)+X_mean
        np.savetxt("genLabel_{}_zrand_tr{}.csv".format(lat, trial), y_gen, delimiter=",")
        np.savetxt("genLatent_{}_zrand_tr{}.csv".format(lat, trial), z[0], delimiter=",")
        np.savetxt("generated_{}_zrand_tr{}.csv".format(lat, trial), b.reshape([-1,248]), delimiter=",")
    
    y_uniqueC = 0.0*np.ones([100,numLabel], dtype='float32')    
    aa = np.array([-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    XX, YY = np.meshgrid(aa, aa)
    XX = XX.reshape(-1,1)
    YY = YY.reshape(-1,1)
    XX = np.append(XX,YY, axis=1)
    XX = np.append(XX, np.zeros([100,lat-2], dtype='float32'), axis=1)
    b_uniqueC = model.decode(y_uniqueC, XX)

    y_uniqueL = np.array([-1.6, -1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2, 1.6, 2.0], dtype='float32').reshape([-1,1])
    #y_uniqueL = np.append(y_uniqueL, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape([-1,1]), axis=1)   
    #print(y_uniqueL.shape)
    y_uniqueL = np.tile(y_uniqueL, (10,1))
    #print(y_uniqueL.shape)
    #y_uniqueL = y_uniqueL.reshape(100,numLabel)
    #y_uniqueL = (y_uniqueL-y_mean)/y_std
    XX = np.zeros([100,lat], dtype='float32')    
    b_uniqueL = model.decode(y_uniqueL, XX)

    rec_train = (rec_train*X_std)+X_mean
    rec_test = (rec_test*X_std)+X_mean
    b = (b*X_std)+X_mean
    b_uniqueC = (b_uniqueC*X_std)+X_mean
    b_uniqueL = (b_uniqueL*X_std)+X_mean
    X_test = (X_test*X_std)+X_mean
    X_train = (X_train*X_std)+X_mean

    np.savetxt("loss_hist.csv", np.array(loss_hist), delimiter=",")

    np.savetxt("reconstructed_{}_train.csv".format(lat), rec_train.reshape([-1,248]), delimiter=",")
    np.savetxt("reconstructed_test_{}.csv".format(lat), rec_test.reshape([-1,248]), delimiter=",")
    np.savetxt("genSameLabel_{}.csv".format(lat), b_uniqueC.reshape([-1,248]), delimiter=",")
    np.savetxt("genSameLabel_Label_{}.csv".format(lat), y_uniqueC*y_std+y_mean, delimiter=",")
    np.savetxt("generated_{}.csv".format(lat), b.reshape([-1,248]), delimiter=",")
    np.savetxt("latent_mean_{}.csv".format(lat), c.reshape([-1,4]), delimiter=",")
    np.savetxt("latent_{}.csv".format(lat), c_sample.reshape([-1,4]), delimiter=",")
    np.savetxt("testData_{}.csv".format(lat), X_test.reshape([-1,248]), delimiter=",")
    np.savetxt("testLabel_{}.csv".format(lat), y_test*y_std+y_mean, delimiter=",")
    np.savetxt("genLabel_{}.csv".format(lat), y_gen*y_std+y_mean, delimiter=",")
    np.savetxt("trainData_{}.csv".format(lat), X_train.reshape([-1,248]), delimiter=",")
    np.savetxt("trainLabel_{}.csv".format(lat), y_train*y_std+y_mean, delimiter=",")
    np.savetxt("CLCD_{}.csv".format(lat), CLCD, delimiter=",")

    print(y_mean, y_std)
