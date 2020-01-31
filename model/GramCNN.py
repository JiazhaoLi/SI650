import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
import os
from util.cnn import fc_layer as fc
import vs_multilayer
from dataset import TestingDataSet
from dataset import TrainingDataSet
from util.cnn import conv_layer as conv
from util.cnn import conv_relu_layer as conv_relu
from util.cnn import pooling_layer as pool
from util.cnn import fc_relu_layer as fc_relu
import pickle 


class GramCNN_Model(object):
    def __init__(self, batch_size, pool_size, train_csv_path, test_csv_path, test_visual_feature_dir, train_visual_feature_dir):
        """ init all hyperparameters here"""
        self.batch_size = batch_size
        self.test_batch_size = 1
        self.vs_lr = 0.0001
        self.lambda_regression = 0.01
        self.alpha = 1.0/batch_size
        #self.alpha=0.06
        self.pool_size = pool_size
        self.semantic_size = 1024 
        self.sentence_embedding_size = 4800
        self.visual_feature_dim = 4096
        
        if not os.path.isfile('../output/train_set.pkl'):
            self.train_set=TrainingDataSet(train_visual_feature_dir, train_csv_path, self.batch_size)
            pickle.dump(self.train_set, open('../output/train_set.pkl','wb'))
        else:
            with open('../output/train_set.pkl','rb') as f:
                self.train_set = pickle.load(f)
        if not os.path.isfile('../output/test_set.pkl'):
            self.test_set=TestingDataSet(test_visual_feature_dir, test_csv_path, self.test_batch_size)
            pickle.dump(self.test_set, open('../output/test_set.pkl','wb'))
        else:
            with open('../output/test_set.pkl','rb') as f:
                self.test_set = pickle.load(f)
        self.context_num = 3
        self.kernel_num = 1024
    

    def initial_weights(self):
        """define all weights here"""
        kernel_initializer = dict()
        glorot = np.sqrt(2/self.semantic_size)
        kernel_initializer['unigram'] = tf.Variable(np.random.normal(loc=0.0, scale=glorot, size=(1,self.semantic_size)), dtype=np.float32)
        kernel_initializer['bigram'] = tf.Variable(np.random.normal(loc=0.0, scale=glorot, size=(3,self.semantic_size)), dtype=np.float32)
        kernel_initializer['trigram'] = tf.Variable(np.random.normal(loc=0.0, scale=glorot, size=(5,self.semantic_size)), dtype=np.float32)
        kernel_initializer['quagram'] = tf.Variable(np.random.normal(loc=0.0, scale=glorot, size=(7,self.semantic_size)), dtype=np.float32)
        # attention weights
        attention_weights = dict()
        glorot = np.sqrt(2/self.kernel_num)
        attention_weights['unigram'] = tf.Variable(np.random.normal(loc=0.0, scale=glorot, size=(1,self.kernel_num)), dtype=np.float32)
        attention_weights['bigram'] = tf.Variable(np.random.normal(loc=0.0, scale=glorot, size=(1,self.kernel_num)), dtype=np.float32)
        attention_weights['trigram'] = tf.Variable(np.random.normal(loc=0.0, scale=glorot, size=(1,self.kernel_num)), dtype=np.float32)
        attention_weights['quagram'] = tf.Variable(np.random.normal(loc=0.0, scale=glorot, size=(1,self.kernel_num)), dtype=np.float32)

        ###
        visual_featmap_ph_train = tf.placeholder(tf.float32, shape=(self.batch_size, self.visual_feature_dim,2*self.context_num+1)) # input feature: current clip, pre-contex, and post contex
        sentence_ph_train = tf.placeholder(tf.float32, shape=(self.batch_size, self.sentence_embedding_size))
        offset_ph = tf.placeholder(tf.float32, shape=(self.batch_size,2))
        visual_featmap_ph_test = tf.placeholder(tf.float32, shape=(self.test_batch_size, self.visual_feature_dim,2*self.context_num+1))  #input feature: current clip, pre-contex, and post contex
        sentence_ph_test = tf.placeholder(tf.float32, shape=(self.test_batch_size, self.sentence_embedding_size))
        return visual_featmap_ph_train,sentence_ph_train,offset_ph,visual_featmap_ph_test,sentence_ph_test, kernel_initializer, attention_weights

    '''
    used in training alignment model, CTRL(aln)
    '''
    def fill_feed_dict_train(self):
        image_batch,sentence_batch,offset_batch = self.train_set.next_batch()
        input_feed = {
                self.visual_featmap_ph_train: image_batch,
                self.sentence_ph_train: sentence_batch,
                self.offset_ph: offset_batch
        }

        return input_feed

    '''
    used in training alignment+regression model, CTRL(reg)
    '''
    def fill_feed_dict_train_reg(self):
        image_batch, sentence_batch, offset_batch = self.train_set.next_batch_iou()
        input_feed = {
                self.visual_featmap_ph_train: image_batch,
                self.sentence_ph_train: sentence_batch,
                self.offset_ph: offset_batch
        }

        return input_feed



    def cross_modal_comb(self, visual_feat, sentence_embed, batch_size):
        vv_feature = tf.reshape(tf.tile(visual_feat, [batch_size, 1]),[batch_size, batch_size, self.kernel_num])
        ss_feature = tf.reshape(tf.tile(sentence_embed,[1, batch_size]),[batch_size, batch_size, self.kernel_num])
        vv_feature1=tf.reshape(vv_feature,[batch_size,batch_size,-1,1])
        ss_feature1=tf.reshape(ss_feature,[batch_size,batch_size,-1,1])
        pool_vv=tf.nn.avg_pool(vv_feature1,ksize=[1,1,self.pool_size,1],strides=[1,1,self.pool_size,1],padding='SAME')
        pool_ss=tf.nn.avg_pool(ss_feature1,ksize=[1,1,self.pool_size,1],strides=[1,1,self.pool_size,1],padding='SAME')
        shape_vv=pool_vv.get_shape().as_list()
        shape_ss=pool_ss.get_shape().as_list()
        vv=tf.reshape(pool_vv,[batch_size*batch_size,shape_vv[2],1])
        ss=tf.reshape(pool_ss,[batch_size*batch_size,1,shape_ss[2]]) #batchx batch, fea
        #print(vv.shape, ss.shape)
        concat_feature=tf.matmul(vv,ss) #batch*batch, 1024*1024
        #print(concat_feature.shape)
        concat_feature=tf.reshape(concat_feature,[batch_size,batch_size,-1])
        comb_feature = tf.reshape(tf.concat([vv_feature, ss_feature, concat_feature],2),[1, batch_size, batch_size,-1 ])
        
        return comb_feature

    def construct_model(self):
        # initialize the placeholder
        self.visual_featmap_ph_train, self.sentence_ph_train, self.offset_ph, self.visual_featmap_ph_test, self.sentence_ph_test, self.kernel_initializer, self.attention_weights=self.initial_weights()
        # build inference network
        sim_reg_mat, sim_reg_mat_test = self.visual_semantic_infer(self.visual_featmap_ph_train, self.sentence_ph_train, self.visual_featmap_ph_test, self.sentence_ph_test, self.kernel_initializer, self.attention_weights)
        # compute loss
        self.loss_align_reg, offset_pred, loss_reg = self.compute_loss_reg(sim_reg_mat, self.offset_ph)
        # optimize
        self.vs_train_op = self.training(self.loss_align_reg)
        return self.loss_align_reg, self.vs_train_op, sim_reg_mat_test, offset_pred, loss_reg

    '''
    visual semantic inference, including visual semantic alignment and clip location regression
    '''
    def visual_semantic_infer(self, visual_feature_train, sentence_embed_train, visual_feature_test, sentence_embed_test, kernel_initializer, attention_weights):
        name="GCNN_Model"
        with tf.variable_scope(name):
            print("Building training network...............................\n")
            """ embedding into common space dim 1024"""
            visual_feature_train = tf.transpose(visual_feature_train,[0,2,1]) # batch num fea
            inputs = tf.reshape(visual_feature_train,[-1,self.visual_feature_dim]) #batch x num,fe
            transformed_clip_train = fc('v2s_lt', inputs, output_dim=self.semantic_size) # batch x num, embed
            transformed_clip_train = tf.reshape(transformed_clip_train,[self.batch_size,2*self.context_num+1,self.semantic_size])#batch num embe
            transformed_sentence_train = fc('s2s_lt', sentence_embed_train, output_dim=self.kernel_num) # batch, embed
            # print transformed_sentence_train.get_shape
            # exit()
            #### G_CNN 
            concat_feature = 0
            uni_batch = tf.reshape(transformed_clip_train[:,self.context_num,:],[self.batch_size,1,-1,1])
            bi_batch = tf.reshape(transformed_clip_train[:,self.context_num-1:self.context_num+2,:],[self.batch_size,3,-1,1])
            tri_batch = tf.reshape(transformed_clip_train[:,self.context_num-2:self.context_num+3,:],[self.batch_size,5,-1,1])
            qua_batch = tf.reshape(transformed_clip_train[:,self.context_num-3:self.context_num+4,:],[self.batch_size,7,-1,1])
           
            unigram = tf.layers.conv2d(inputs=uni_batch, filters=self.kernel_num, activation=tf.nn.relu,kernel_size=[1,self.semantic_size],padding='valid',kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1), name='unigram')
            bigram = tf.layers.conv2d(inputs=bi_batch,filters=self.kernel_num,activation=tf.nn.relu,kernel_size=[3,self.semantic_size],padding='valid',kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),name='bigram')
            trigram = tf.layers.conv2d(inputs=tri_batch,filters=self.kernel_num,activation=tf.nn.relu,kernel_size=[5,self.semantic_size],padding='valid',kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),name='trigram')
            quagram = tf.layers.conv2d(inputs=qua_batch,filters=self.kernel_num,activation=tf.nn.relu,kernel_size=[7,self.semantic_size],padding='valid',kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),name='quagram')
           
            # attention
            unigram_score = tf.layers.dense(inputs=unigram,units=1,activation=tf.nn.relu,kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))
            bigram_score = tf.layers.dense(inputs=bigram,units=1,activation=tf.nn.relu,kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))
            trigram_score = tf.layers.dense(inputs=trigram,units=1,activation=tf.nn.relu,kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))
            quagram_score = tf.layers.dense(inputs=quagram,units=1,activation=tf.nn.relu,kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))

            unigram_score = tf.tanh(unigram_score)
            bigram_score = tf.tanh(bigram_score)
            trigram_score = tf.tanh(trigram_score)
            quagram_score = tf.tanh(quagram_score)

            unigram_score = tf.reshape(unigram_score,[self.batch_size,1])
            bigram_score = tf.reshape(bigram_score,[self.batch_size,1])
            trigram_score = tf.reshape(trigram_score,[self.batch_size,1])
            quagram_score = tf.reshape(quagram_score,[self.batch_size,1])

            concat_score = tf.concat([unigram_score,bigram_score,trigram_score,quagram_score],1)
            alpha = tf.reshape(tf.nn.softmax(concat_score),[self.batch_size,4,1])
            
            # reconpute the representation
            unigram = tf.reshape(unigram,[self.batch_size,self.kernel_num,1])
            bigram = tf.reshape(bigram,[self.batch_size,self.kernel_num,1])
            trigram = tf.reshape(trigram,[self.batch_size,self.kernel_num,1])
            quagram = tf.reshape(quagram,[self.batch_size,self.kernel_num,1])

            concat_feature = tf.concat([unigram,bigram,trigram,quagram],2)
            input_vision = tf.reshape(tf.matmul(concat_feature,alpha),[self.batch_size,self.kernel_num])

            transformed_clip_train_norm = tf.nn.l2_normalize(input_vision, axis=1)
            transformed_sentence_train_norm = tf.nn.l2_normalize(transformed_sentence_train, axis=1)
            
            cross_modal_vec_train = self.cross_modal_comb(transformed_clip_train_norm,transformed_sentence_train_norm, self.batch_size) # batch batch 2*conmmon_space_dim
            #print cross_modal_vec_train.shape
            sim_score_mat_train = vs_multilayer.vs_multilayer(cross_modal_vec_train, "vs_multilayer_lt", middle_layer_dim=1000)
            #print sim_score_mat_train.shape
            sim_score_mat_train = tf.reshape(sim_score_mat_train,[self.batch_size, self.batch_size, 3])
            tf.get_variable_scope().reuse_variables()



            print("Building test network...............................\n")
            visual_feature_test=tf.transpose(visual_feature_test,[0,2,1]) # batch num fea
            inputs=tf.reshape(visual_feature_test,[-1,self.visual_feature_dim]) #batch x num,fe
            transformed_clip_test = fc('v2s_lt', inputs, output_dim=self.semantic_size)
            transformed_clip_test=tf.reshape(transformed_clip_test,[self.test_batch_size,2*self.context_num+1,self.semantic_size])#batch num embe
            transformed_sentence_test = fc('s2s_lt', sentence_embed_test, output_dim=self.kernel_num)
            # print transformed_sentence_train.get_shape
    
            #### G_CNN 
            concat_feature = 0
            uni_batch = tf.reshape(transformed_clip_test[:,self.context_num,:],[self.test_batch_size,1,-1,1])
            bi_batch = tf.reshape(transformed_clip_test[:,self.context_num-1:self.context_num+2,:],[self.test_batch_size,3,-1,1])
            tri_batch = tf.reshape(transformed_clip_test[:,self.context_num-2:self.context_num+3,:],[self.test_batch_size,5,-1,1])
            qua_batch = tf.reshape(transformed_clip_test[:,self.context_num-3:self.context_num+4,:],[self.test_batch_size,7,-1,1])
           
            unigram = tf.layers.conv2d(inputs=uni_batch, filters=self.kernel_num, activation=tf.nn.relu,kernel_size=[1,self.semantic_size],padding='valid',kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1), reuse=True,name='unigram')
            bigram = tf.layers.conv2d(inputs=bi_batch,filters=self.kernel_num,activation=tf.nn.relu,kernel_size=[3,self.semantic_size],padding='valid',kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),reuse=True,name='bigram')
            trigram = tf.layers.conv2d(inputs=tri_batch,filters=self.kernel_num,activation=tf.nn.relu,kernel_size=[5,self.semantic_size],padding='valid',kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),reuse=True,name='trigram')
            quagram = tf.layers.conv2d(inputs=qua_batch,filters=self.kernel_num,activation=tf.nn.relu,kernel_size=[7,self.semantic_size],padding='valid',kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),reuse=True,name='quagram')
           
            # attention
            unigram_score = tf.layers.dense(inputs=unigram,units=1,activation=tf.nn.relu,reuse=True,kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))
            bigram_score = tf.layers.dense(inputs=bigram,units=1,activation=tf.nn.relu,reuse=True,kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))
            trigram_score = tf.layers.dense(inputs=trigram,units=1,activation=tf.nn.relu,reuse=True,kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))
            quagram_score = tf.layers.dense(inputs=quagram,units=1,activation=tf.nn.relu,reuse=True,kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))

            unigram_score = tf.tanh(unigram_score)
            bigram_score = tf.tanh(bigram_score)
            trigram_score = tf.tanh(trigram_score)
            quagram_score = tf.tanh(quagram_score)

            unigram_score = tf.reshape(unigram_score,[self.test_batch_size,1])
            bigram_score = tf.reshape(bigram_score,[self.test_batch_size,1])
            trigram_score = tf.reshape(trigram_score,[self.test_batch_size,1])
            quagram_score = tf.reshape(quagram_score,[self.test_batch_size,1])

            concat_score = tf.concat([unigram_score,bigram_score,trigram_score,quagram_score],1)
            alpha = tf.reshape(tf.nn.softmax(concat_score),[self.test_batch_size,4,1])
            
            # reconpute the representation
            unigram = tf.reshape(unigram,[self.test_batch_size,self.kernel_num,1])
            bigram = tf.reshape(bigram,[self.test_batch_size,self.kernel_num,1])
            trigram = tf.reshape(trigram,[self.test_batch_size,self.kernel_num,1])
            quagram = tf.reshape(quagram,[self.test_batch_size,self.kernel_num,1])

            concat_feature = tf.concat([unigram,bigram,trigram,quagram],2)
            input_vision = tf.reshape(tf.matmul(concat_feature,alpha),[self.test_batch_size,self.kernel_num])

            transformed_clip_test_norm = tf.nn.l2_normalize(input_vision, axis=1)
            transformed_sentence_test_norm = tf.nn.l2_normalize(transformed_sentence_test, axis=1)
            
            cross_modal_vec_test = self.cross_modal_comb(transformed_clip_test_norm,transformed_sentence_test_norm, self.test_batch_size) # batch batch 2*conmmon_space_dim
            #print cross_modal_vec_test.shape
            sim_score_mat_test = vs_multilayer.vs_multilayer(cross_modal_vec_test, "vs_multilayer_lt", reuse=True, middle_layer_dim=1000)
            sim_score_mat_test = tf.reshape(sim_score_mat_test,[self.test_batch_size, self.test_batch_size, 3])
            tf.get_variable_scope().reuse_variables()

            return sim_score_mat_train, sim_score_mat_test


    def compute_loss_reg(self, sim_reg_mat, offset_label):

        
        sim_score_mat, p_reg_mat, l_reg_mat = tf.split(sim_reg_mat,3)
        sim_score_mat = tf.reshape(sim_score_mat, [self.batch_size, self.batch_size])
        l_reg_mat = tf.reshape(l_reg_mat, [self.batch_size, self.batch_size])
        p_reg_mat = tf.reshape(p_reg_mat, [self.batch_size, self.batch_size])
        
        # unit matrix with -2
        I_2 = tf.diag(tf.constant(-2.0, shape=[self.batch_size]))
        all1 = tf.constant(1.0, shape=[self.batch_size, self.batch_size])
        #               | -1  1   1...   |

        #   mask_mat =  | 1  -1  -1...   |

        #               | 1   1  -1 ...  |
        mask_mat = tf.add(I_2, all1)
        # loss cls, not considering iou
        I = tf.diag(tf.constant(1.0, shape=[self.batch_size]))
        I_half = tf.diag(tf.constant(0.5, shape=[self.batch_size]))
        batch_para_mat = tf.constant(self.alpha, shape=[self.batch_size, self.batch_size])
        para_mat = tf.add(I,batch_para_mat)
        loss_mat = tf.log(tf.add(all1, tf.exp(tf.multiply(mask_mat, sim_score_mat))))
        loss_mat = tf.multiply(loss_mat, para_mat)
        loss_align = tf.reduce_mean(loss_mat)
        # regression loss
        l_reg_diag = tf.matmul(tf.multiply(l_reg_mat, I), tf.constant(1.0, shape=[self.batch_size, 1]))
        p_reg_diag = tf.matmul(tf.multiply(p_reg_mat, I), tf.constant(1.0, shape=[self.batch_size, 1]))
        offset_pred = tf.concat((p_reg_diag, l_reg_diag),1)
        loss_reg = tf.reduce_mean(tf.abs(tf.subtract(offset_pred, offset_label)))

        loss=tf.add(tf.multiply(self.lambda_regression, loss_reg), loss_align)
        return loss, offset_pred, loss_reg

    """
    compute alignment and regression loss
    """

    def get_variables_by_name(self,name_list):
        v_list = tf.trainable_variables()
        v_dict = {}
        for name in name_list:
            v_dict[name] = []
        for v in v_list:
            for name in name_list:
                if name in v.name: v_dict[name].append(v)

        for name in name_list:
            print("Variables of <"+name+">")
            for v in v_dict[name]:
                print("    "+v.name)
        return v_dict

    def training(self, loss):
        v_dict = self.get_variables_by_name(["lt"])
        vs_optimizer = tf.train.AdamOptimizer(self.vs_lr, name='vs_adam')
        vs_train_op = vs_optimizer.minimize(loss, var_list=v_dict["lt"])
        return vs_train_op
