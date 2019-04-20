def superpixel_to_seg(unsigned long H, unsigned long W, unsigned long K,
                      float [:,:,:] m_k,
                      unsigned long[:,:] superindexes,
                      float[:,:] superpixel_pred):

    cdef unsigned long h
    cdef unsigned long w
    cdef unsigned long k
    cdef float value
    cdef unsigned long super_index

    for h in range(H):
        for w in range(W):
            for k in range(K):
                super_index = superindexes[h,w]
                value = superpixel_pred[k, super_index]
                m_k[h,w,k] = value
