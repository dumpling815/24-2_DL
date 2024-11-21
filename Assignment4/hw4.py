"""

HW4

"""

import torch
import torch.nn as nn #Own code needed -> I personally added to control warning msgs.
import torch.nn.functional as F
import numbers
import numpy as np
import math


"""

function view_as_windows

"""

def view_as_windows(arr_in, window_shape, step=1):
    """Rolling window view of the input n-dimensional array.
    Windows are overlapping views of the input array, with adjacent windows
    shifted by a single row or column (or an index of a higher dimension).
    Parameters
    ----------
    arr_in : Pytorch tensor
        N-d Pytorch tensor.
    window_shape : integer or tuple of length arr_in.ndim
        Defines the shape of the elementary n-dimensional orthotope
        (better know as hyperrectangle [1]_) of the rolling window view.
        If an integer is given, the shape will be a hypercube of
        sidelength given by its value.
    step : integer or tuple of length arr_in.ndim
        Indicates step size at which extraction shall be performed.
        If integer is given, then the step is uniform in all dimensions.
    Returns
    -------
    arr_out : ndarray
        (rolling) window view of the input array.
    Notes
    -----
    One should be very careful with rolling views when it comes to
    memory usage.  Indeed, although a 'view' has the same memory
    footprint as its base array, the actual array that emerges when this
    'view' is used in a computation is generally a (much) larger array
    than the original, especially for 2-dimensional arrays and above.
    For example, let us consider a 3 dimensional array of size (100,
    100, 100) of ``float64``. This array takes about 8*100**3 Bytes for
    storage which is just 8 MB. If one decides to build a rolling view
    on this array with a window of (3, 3, 3) the hypothetical size of
    the rolling view (if one was to reshape the view for example) would
    be 8*(100-3+1)**3*3**3 which is about 203 MB! The scaling becomes
    ( 3,3,3 ) 짜리  window가 97*3개 존재하게 되기 때문! => 차원의 크기에 따라 필요한 메모리가 지수적으로 증가함!
    even worse as the dimension of the input array becomes larger.
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Hyperrectangle
    Examples
    --------
    >>> import torch
    >>> A = torch.arange(4*4).reshape(4,4)
    >>> A
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])
    >>> window_shape = (2, 2)
    >>> B = view_as_windows(A, window_shape)
    >>> B[0, 0]
    array([[0, 1],
           [4, 5]])
    >>> B[0, 1]
    array([[1, 2],
           [5, 6]])
    >>> A = torch.arange(10)
    >>> A
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> window_shape = (3,)
    >>> B = view_as_windows(A, window_shape)
    >>> B.shape
    (8, 3)
    >>> B
    array([[0, 1, 2],
           [1, 2, 3],
           [2, 3, 4],
           [3, 4, 5],
           [4, 5, 6],
           [5, 6, 7],
           [6, 7, 8],
           [7, 8, 9]])
    >>> A = torch.arange(5*4).reshape(5, 4)
    >>> A
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15],
           [16, 17, 18, 19]])
    >>> window_shape = (4, 3)
    >>> B = view_as_windows(A, window_shape)
    >>> B.shape
    (2, 2, 4, 3)
    >>> B  # doctest: +NORMALIZE_WHITESPACE
    array([[[[ 0,  1,  2],
             [ 4,  5,  6],
             [ 8,  9, 10],
             [12, 13, 14]],
            [[ 1,  2,  3],
             [ 5,  6,  7],
             [ 9, 10, 11],
             [13, 14, 15]]],
           [[[ 4,  5,  6],
             [ 8,  9, 10],
             [12, 13, 14],
             [16, 17, 18]],
            [[ 5,  6,  7],
             [ 9, 10, 11],
             [13, 14, 15],
             [17, 18, 19]]]])
    """

    # -- basic checks on arguments
    if not torch.is_tensor(arr_in):
        raise TypeError("`arr_in` must be a pytorch tensor")

    ndim = arr_in.ndim


    if isinstance(window_shape, numbers.Number): # window shape가 요소가 하나인 경우.
        window_shape = (window_shape,) * ndim
        print('debug')
    if not (len(window_shape) == ndim): # window shape와 input tensor의 차원 갯수가 안맞을 경우 오류 발생.
        raise ValueError("`window_shape` is incompatible with `arr_in.shape`")

    if isinstance(step, numbers.Number):
        if step < 1:
            raise ValueError("`step` must be >= 1")
        step = (step,) * ndim
    if len(step) != ndim:
        raise ValueError("`step` is incompatible with `arr_in.shape`")

    arr_shape = torch.tensor(arr_in.shape)
    window_shape = torch.tensor(window_shape, dtype=arr_shape.dtype)

    if ((arr_shape - window_shape) < 0).any():
        raise ValueError("`window_shape` is too large")

    if ((window_shape - 1) < 0).any():
        raise ValueError("`window_shape` is too small")

    # -- build rolling window view
    slices = tuple(slice(None, None, st) for st in step)
    # window_strides = torch.tensor(arr_in.stride())
    window_strides = arr_in.stride()

    indexing_strides = arr_in[slices].stride()

    win_indices_shape = torch.div(arr_shape - window_shape
                          , torch.tensor(step), rounding_mode = 'floor') + 1
    
    new_shape = tuple(list(win_indices_shape) + list(window_shape))
    strides = tuple(list(indexing_strides) + list(window_strides))

    arr_out = torch.as_strided(arr_in, size=new_shape, stride=strides)
    return arr_out

#######
# if necessary, you can define additional functions which help your implementation,
# and import proper libraries for those functions.
#######

class nn_convolutional_layer:

    def __init__(self, f_height, f_width, input_size, in_ch_size, out_ch_size):
        
        # Xavier init
        self.W = torch.normal(0, 1 / math.sqrt((in_ch_size * f_height * f_width / 2)),
                                  size=(out_ch_size, in_ch_size, f_height, f_width))
        self.b = 0.01 + torch.zeros(size=(1, out_ch_size, 1, 1))

        self.W.requires_grad = True
        self.b.requires_grad = True

        self.input_size = input_size

    def update_weights(self, dW, db):
        self.W += dW
        self.b += db

    def get_weights(self):
        return self.W.clone().detach(), self.b.clone().detach()

    def set_weights(self, W, b):
        self.W = W.clone().detach()
        self.b = b.clone().detach()

    #######
    # Q1. Complete this method
    #######
    def forward(self, x):
        # forward function will be performed without padding, with stride 1.
        # That is, input width(height): N, filter size: F, output width: N-F+1
        batch_size = x.shape[0]
        num_filters = self.W.shape[0]
        in_ch_size = self.W.shape[1]
        filter_height = self.W.shape[2]
        filter_width = self.W.shape[3]

        output_height = x.shape[2] - filter_height + 1
        output_width = x.shape[3] - filter_width + 1

        out = torch.zeros(size=(x.shape[0],  
                                num_filters, 
                                output_height, 
                                output_width))

            
        windows = view_as_windows(x,(1,in_ch_size,filter_height,filter_width)).squeeze()
        # windows shape : (batch size, N-F+1, N-F+1, in_ch_size, filter_height, filter_width)
        # In this problem's first forward, (8,30,30,3,3,3) 
        for filter_index in range(num_filters):
            filter = self.W[filter_index]
            out[:,filter_index,:,:] = (windows * filter).sum(dim=(-3,-2,-1)) #마지막 3차원 합산, 
        out += self.b
            # (windows * filter).sum(dim=(-3,-2,-1)) 에서 windows * filters를 하면 windows size와 같이 나옴.
            # 거기서 마지막 3차원 (filter size)를 모두 더하면 (batch size, N-F+1, N-F+1)
            # 이걸 [:,filter_index,:,:]를 통해 최종 out과 차원 맞춤.
        
        # view_as_windows의 특징은, input tensor의 차원이 증가하면 필요한 메모리가 지수적으로 증가하게 됨.
        # 따라서, for문을 더 사용해서 windows에 들어가는 input의 차원을 줄이면 시간 복잡도는 증가, 공간 복잡도는 감소
        # 그러나 우리가 고려해아하는건 공간 복잡도가 지나치게 커진다면 결국은 OS의 memory caching 성능이 감소하게 됨
        # 즉, 알고리즘 자체의 시간 복잡도 for문을 적게 사용하는 것이 유리할 수도 있으나, 전체 컴퓨팅 성능을 고려했을 때 메모리 자원의 병목으로
        # 컴퓨팅 시간이 길어질 수도 있음을 명심해야함.

        """ # 여기서 부터 새로운 implementation
        for data in range(batch_size):
            window = view_as_windows(x[data],(in_ch_size,filter_height,filter_width))
            for filter_index in range(num_filters):
                filter = self.W[filter_index]
                out[data,filter_index,:,:] = (window * filter).sum(dim=(-3,-2,-1))
        out += self.b"""

        return out

    

    #######
    ## If necessary, you can define additional class methods here
    #######


class nn_max_pooling_layer:
    def __init__(self, pool_size, stride):
        self.stride = stride
        self.pool_size = pool_size
        #######
        ## If necessary, you can define additional class variables here
        #######

    #######
    # Q2. Complete this method
    #######
    def forward(self, x):
        batch_size = x.shape[0]
        in_ch_size = x.shape[1]
        output_height = (x.shape[2] - self.pool_size) // self.stride + 1
        output_width = (x.shape[3] - self.pool_size) // self.stride + 1

        out = torch.zeros(size=(batch_size,in_ch_size,output_width,output_height))

        for data in range(batch_size):
            for depth in range(in_ch_size):
                windows = view_as_windows(x[data][depth],(self.pool_size,self.pool_size),stride)
                for row in range(len(windows)):
                    for col in range(len(windows[0])):
                        window = windows[row][col]
                        out[data,depth,row,col] = window.max()
        return out
    #######
    ## If necessary, you can define additional class methods here
    #######

"""
TESTING 
"""

if __name__ == "__main__":

    # data sizes
    batch_size = 8
    input_size = 32
    filter_width = 3
    filter_height = filter_width
    in_ch_size = 3
    num_filters = 8

    std = 1e0
    dt = 1e-3

    # number of test loops
    num_test = 50

    # error parameters
    err_fwd = 0
    err_pool = 0


    # for reproducibility
    # torch.manual_seed(0)

    # set default type to float64
    torch.set_default_dtype(torch.float64)

    print('conv test')
    for i in range(num_test):
        # create convolutional layer object
        cnv = nn_convolutional_layer(filter_height, filter_width, input_size,
                                   in_ch_size, num_filters)

        # test conv layer from torch.nn for reference
        test_conv_layer = torch.nn.Conv2d(in_channels=in_ch_size, out_channels=num_filters,
                                kernel_size = (filter_height, filter_width))
        
        # test input
        x = torch.normal(0, 1, (batch_size, in_ch_size, input_size, input_size))
        
        with torch.no_grad():
            
            out = cnv.forward(x)
            W,b = cnv.get_weights()
            test_conv_layer.weight = nn.Parameter(W)
            test_conv_layer.bias = nn.Parameter(torch.squeeze(b))
            test_out = test_conv_layer(x)
            
            err=torch.norm(test_out - out)/torch.norm(test_out)
            err_fwd+= err
    
    stride = 2
    pool_size = 2
    
    print('pooling test')
    for i in range(num_test):
        # create pooling layer object
        mpl = nn_max_pooling_layer(pool_size=pool_size, stride=stride)
        
        # test pooling layer from torch.nn for reference
        test_pooling_layer = nn.MaxPool2d(kernel_size=(pool_size,pool_size), stride=stride)
        
        # test input
        x = torch.normal(0, 1, (batch_size, in_ch_size, input_size, input_size))
        
        with torch.no_grad():
            out = mpl.forward(x)
            test_out = test_pooling_layer(x)
            
            err=torch.norm(test_out - out)/torch.norm(test_out)
            err_pool+= err
    
    # reporting accuracy results.
    print('accuracy results')
    print('forward accuracy', 100 - err_fwd/num_test*100, '%')
    print('pooling accuracy', 100 - err_pool/num_test*100, '%')