import tensorflow as tf

class f1_score(tf.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(f1_score, self).__init__(name=name, **kwargs)
        self.tp = self.add_weight(dtype=tf.int32,name='f1_tp', initializer='zeros')
        self.tn = self.add_weight(dtype=tf.int32,name='f1_tn', initializer='zeros')
        self.fp = self.add_weight(dtype=tf.int32,name='f1_fp', initializer='zeros')
        self.fn = self.add_weight(dtype=tf.int32,name='f1_fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        pred = tf.argmax(y_pred,axis=1)
        pred = tf.cast(pred, tf.int32)
        #true = tf.argmax(label,axis=1)
        confusion = tf.math.confusion_matrix(y_true,pred,num_classes=2)

        self.tp.assign_add(confusion[0,0])
        self.tn.assign_add(confusion[1,1])
        self.fp.assign_add(confusion[1,0])
        self.fn.assign_add(confusion[0,1])

    def result(self):
        prec = tf.cast(self.tp,tf.float32 )/ tf.cast(self.tp+self.fp,tf.float32 )
        rec = tf.cast(self.tp,tf.float32 )/ tf.cast(self.tp+self.fn,tf.float32 )
        mul1 = tf.math.scalar_mul(prec,rec,name='mul1')
        mul2 = tf.math.scalar_mul(tf.constant(2.0),mul1, name='mul2')
        #mul2_con = tf.cast(mul2)
        return tf.cond(tf.equal((prec+rec),tf.constant(0.0)),\
            true_fn=lambda: tf.constant(-1.0), \
            false_fn=lambda: tf.math.divide(mul2,(prec+rec),name='div'))
        
        #res = tf.math.divide(mul2,(prec+rec),name='div')
        #return res

class confusion_metric(tf.metrics.Metric):
    def __init__(self, name='confusion_metric', **kwargs):
        super(confusion_metric, self).__init__(name=name, **kwargs)
        self.tp = self.add_weight(dtype=tf.int32,name='f1_tp', initializer='zeros')
        self.tn = self.add_weight(dtype=tf.int32,name='f1_tn', initializer='zeros')
        self.fp = self.add_weight(dtype=tf.int32,name='f1_fp', initializer='zeros')
        self.fn = self.add_weight(dtype=tf.int32,name='f1_fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        pred = tf.argmax(y_pred,axis=1)
        pred = tf.cast(pred, tf.int32)
        #true = tf.argmax(label,axis=1)
        confusion = tf.math.confusion_matrix(y_true,pred,num_classes=2)

        self.tp.assign_add(confusion[0,0])
        self.tn.assign_add(confusion[1,1])
        self.fp.assign_add(confusion[1,0])
        self.fn.assign_add(confusion[0,1])

    def result(self):
        return tf.constant([self.tp,self.fn,self.fp,self.tn])