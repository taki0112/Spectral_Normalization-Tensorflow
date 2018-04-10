import tensorflow as tf

def l2_norm(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)


def spectral_norm(w, iteration=1, update_collection=tf.GraphKeys.UPDATE_OPS):
    first_call = not tf.get_variable_scope().reuse

    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u_hat, v_hat = None, None

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)
    u_iter = u
    for i in range(iteration) :
        """
        
        power iteration
        Usually iteration = 1 will be enough
        
        """
        v_ = tf.matmul(u_iter, tf.transpose(w))
        v_hat = l2_norm(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = l2_norm(u_)
        u_iter = u_hat

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    w_norm = w / sigma

    if update_collection is None:
        """
        
        Setting update_collection to None will make u being updated every W execution. This maybe undesirable
        Please consider using a update collection instead.
        default update_collection is tf.GraphKeys.UPDATE_OPS
        
        """
        with tf.control_dependencies([u.assign(u_hat)]):
            w_norm = tf.reshape(w_norm, w_shape)
    else:
        if first_call:
            """

            if GAN : 
                Discriminator and Generator will call for real image
                If fake image, it can not pass here

            """
            tf.add_to_collection(update_collection, u.assign(u_hat))

        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm