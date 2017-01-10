# This is a snippet added to TFLearn's objectives.py as a custom loss function.
# This code will not function by itself, and is for reference only.

def neg_ln_with_total_variance(y_pred, y_true):
        # Removes the bw layer from the predicted results. This is due to a 
        # TFLearn bug that requires that the training data shape is the same
        # as the predicted data shape. This allows more flexibility for loss
        # formula experimentation.
        y_pred = tf.unpack(y_pred, axis=3)
        y_pred.pop(0)
        y_pred = tf.pack(y_pred, axis=3)

        ln_loss = tf.reduce_mean(-tf.log(y_pred+1e-8))
        tv_loss = total_variance(y_pred)
        loss = ln_loss + tv_loss

        return loss

def total_variance(y_pred):
        tv_weight = .1e-3

        batch_h_list = tf.unpack(tf.squeeze(y_pred))
        batch_h_res = []
        for batch_h in batch_h_list:
            batch_h = tf.unpack(batch_h)
            left = list(batch_h)
            right = list(batch_h)
            right.insert(0, tf.zeros_like(batch_h[0]))
            right.pop()
            batch_h = tf.pack(right) - tf.pack(left)
            batch_h_res.append(batch_h)
        batch_h_res = tf.pack(batch_h_res)
        hor_smooth = tf.reduce_mean(tf.square(batch_h_res))

        batch_w_list = tf.unpack(tf.transpose(tf.squeeze(y_pred), [0, 2, 1, 3]))
        batch_w_res = []
        for batch_w in batch_w_list:
            batch_w = tf.unpack(batch_w)
            left = list(batch_w)
            right = list(batch_w)
            right.insert(0, tf.zeros_like(batch_w[0]))
            right.pop()
            batch_w = tf.pack(right) - tf.pack(left)
            batch_w_res.append(batch_w)
        batch_w_res = tf.pack(batch_w_res)
        ver_smooth = tf.reduce_mean(tf.square(batch_w_res))

        tv_loss = tf.reduce_mean(hor_smooth + ver_smooth)

        return tv_weight * tf.reduce_mean(tv_loss)

