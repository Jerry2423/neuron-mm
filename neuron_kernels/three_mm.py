"""
AWS Neuron Optimized Three Matrix Multiplication Kernels
Author: Jierui (Jerry) Xu (UW Madison)
Date: 2025.07.06

This file implements several optimized kernels for fused three matrix 
multiplication operations (A×B×C) targeting AWS NeuronCore hardware. 
The kernels progressively demonstrate optimization techniques including:
- Fusion of multiple matrix multiplications
- Memory hierarchy management
- Tiling and blocking strategies
- Data reuse optimization
- Loop order manipulation for cache efficiency
"""
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
from neuronxcc.nki.language import par_dim
import neuronxcc.nki.isa as nisa
import numpy as np
import argparse
import neuronxcc.nki.typing as nt
from scipy.special import softmax
from two_mm import nki_matmul_tiled_basic_natural_lhs, nki_matmul_tiled_basic, nki_matmul_fully_optimized_
import math
from utils import load_tensor_block, matmul_block_T, load_tensor_block_T


# Input: X^T, U_ref, V_ref 
# Output: (XUV)^T
@nki.jit
def fused_three_mm_XTUV_transpose(
    X_ref,  # Shape: (M, K) - stored as transpose
    U_ref,  # Shape: (M, r)
    V_ref,  # Shape: (r, N)
    mixed_precision=True,
    r_tiles_in_block=8,
    K_tiles_in_block=2,
    N_tiles_in_block=4,
    M_tiles_in_block=4
):
    # Use X_ref dtype as the intermediate tensor dtype
    # Assume all IO tensors have the same dtype
    kernel_dtype = X_ref.dtype
    pe_in_dt = nl.bfloat16 if mixed_precision else np.float32
    assert X_ref.dtype == U_ref.dtype == V_ref.dtype

    # Shape checking - X@U@V computation
    M, K = X_ref.shape # M - 7168
    M_U, r = U_ref.shape
    r_V, N = V_ref.shape # N - 18432
    assert tuple(U_ref.shape) == (M_U, r), "Input shape mismatch!"
    assert tuple(V_ref.shape) == (r_V, N), "Input shape mismatch!"
    assert M == M_U, "M dimension must match between X and U!"
    assert r == r_V, "r dimension must match between U and V!"

    out_ref = nl.ndarray((N, K), dtype=V_ref.dtype, buffer=nl.hbm)


    # Tiling configuration

    M_tile_size = 128
    M_block_size = M_tile_size * M_tiles_in_block
    M_n_blocks = (M + M_block_size - 1) // M_block_size

    r_tile_size = 128
    r_block_size = r_tile_size * r_tiles_in_block
    r_n_blocks = (r + r_block_size - 1) // r_block_size

    K_tile_size = 512
    K_block_size = int(K_tile_size * K_tiles_in_block)
    if K_block_size < K_tile_size:
        K_tile_size = K_block_size
        K_tiles_in_block = 1
    
    if K < K_block_size:
        K_tile_size = K
        K_tiles_in_block = 1
        K_block_size = K_tile_size * K_tiles_in_block
    
    K_n_blocks = (K + K_block_size - 1) // K_block_size

    N_tile_size = 128
    N_block_size = N_tile_size * N_tiles_in_block
    N_n_blocks = (N + N_block_size - 1) // N_block_size

    # Index patterns
    ip_X = nl.arange(M_tile_size)[:, None]
    if_X_tile = nl.arange(K_tile_size)[None, :]
    if_X_block = nl.arange(K_block_size)[None, :]

    ip_V = nl.arange(r_tile_size)[:, None]
    if_V_tile = nl.arange(N_tile_size)[None, :]
    if_V_block = nl.arange(N_block_size)[None, :]

    ip_U = nl.arange(M_tile_size)[:, None]
    if_U_tile = nl.arange(r_tile_size)[None, :]
    if_U_block = nl.arange(r_block_size)[None, :]

    # Main computation loops
    for i_K_block in nl.affine_range(K_n_blocks):  # Loop over K dimension blocks
        # Buffer for intermediate result XU (K x r)
        XU_result_buf = nl.zeros(
            (r_n_blocks, r_tiles_in_block, par_dim(r_tile_size), K_block_size), dtype=kernel_dtype
        )
        # Loop over r dimension blocks
        for i_M_block in nl.sequential_range(M_n_blocks):
            # X_cache = nl.ndarray((par_dim(M_tile_size), M_tiles_in_block, K_block_size), dtype=pe_in_dt)
            X_cache = load_tensor_block(
                X_ref,
                (i_M_block * M_block_size, i_K_block * K_block_size),
                (M_tiles_in_block, M_tile_size, K_block_size),
            )
            for i_r_block in nl.affine_range(r_n_blocks):

                # U_cache = nl.ndarray((par_dim(M_tile_size), M_tiles_in_block, r_block_size), dtype=pe_in_dt)
                U_cache = load_tensor_block(
                    U_ref,
                    (i_M_block * M_block_size, i_r_block * r_block_size),
                    (M_tiles_in_block, M_tile_size, r_block_size),
                )

                for ib_K_tile in nl.affine_range(K_tiles_in_block):
                    for ib_r_tile in nl.affine_range(r_tiles_in_block):
                        # PSUM buffer for X @ U
                        XU_psum = nl.zeros((par_dim(r_tile_size), K_tile_size), dtype=np.float32, buffer=nl.psum)

                        # Index patterns for result
                        if_XU = nl.arange(K_tile_size)[None, :]
                        ip_XU = nl.arange(r_tile_size)[:, None]

                        # Contract over M dimension
                        for ib_M_tile in nl.affine_range(M_tiles_in_block):
                            # Compute X^T @ U (since X is stored transposed)
                            XU_psum[ip_XU, if_XU] += nisa.nc_matmul(
                                moving=X_cache[ip_X, ib_M_tile, if_X_tile + ib_K_tile * K_tile_size],
                                stationary=U_cache[ip_U, ib_M_tile, if_U_tile + ib_r_tile * r_tile_size],
                            )

                        XU_result_buf[i_r_block, ib_r_tile, ip_XU, if_XU + ib_K_tile * K_tile_size] += XU_psum[
                            ip_XU, if_XU
                        ]

        # Loop over N dimension blocks for final result
        for i_N_block in nl.affine_range(N_n_blocks):

            # TODO: Create a final result buffer
            final_result_buf = nl.zeros((N_tiles_in_block, par_dim(N_tile_size), K_block_size), dtype=kernel_dtype)

            if_out = nl.arange(K_block_size)[None, :]
            ip_out = nl.arange(N_tile_size)[:, None]

            for i_r_block in nl.sequential_range(r_n_blocks):
                # Compute (XU) @ V for current blocks

                # V_cache = nl.ndarray((par_dim(r_tile_size), r_tiles_in_block, N_block_size), dtype=pe_in_dt)
                V_cache = load_tensor_block(
                    V_ref,
                    (i_r_block * r_block_size, i_N_block * N_block_size),
                    (r_tiles_in_block, r_tile_size, N_block_size),
                )
                for ib_K_tile in nl.affine_range(K_tiles_in_block):

                    for ib_N_tile in nl.affine_range(N_tiles_in_block):
                        # PSUM buffer for final result
                        XUV_psum = nl.zeros((par_dim(N_tile_size), K_tile_size), dtype=np.float32, buffer=nl.psum)

                        ip_XU_t = nl.arange(r_tile_size)[:, None]
                        if_XU_t = nl.arange(K_tile_size)[None, :]

                        ip_XUV = nl.arange(N_tile_size)[:, None]
                        if_XUV = nl.arange(K_tile_size)[None, :]
                        
                        # Contract over r dimension
                        for ib_r_tile in nl.affine_range(r_tiles_in_block):
                            # Compute XU @ V
                            ip_V_t = nl.arange(r_tile_size)[:, None]
                            if_V_t = nl.arange(N_tile_size)[None, :]

                            XUV_psum[ip_XUV, if_XUV] += nisa.nc_matmul(
                                moving=XU_result_buf[i_r_block, ib_r_tile, ip_XU_t, if_XU_t + ib_K_tile * K_tile_size],
                                stationary=V_cache[ip_V_t, ib_r_tile, if_V_t + ib_N_tile * N_tile_size],
                            )

                        final_result_buf[ib_N_tile, ip_XUV, ib_K_tile * K_tile_size + if_XUV] += XUV_psum[
                            ip_XUV, if_XUV
                        ]
            for ib_N_tile in nl.affine_range(N_tiles_in_block):
                # Store the final result for the current N block
                nl.store(
                    out_ref[
                        i_N_block * N_block_size + ib_N_tile * N_tile_size + ip_out,
                        i_K_block * K_block_size + if_out,
                    ],
                    value=final_result_buf[ib_N_tile, ip_out, if_out],
                    mask=(i_N_block * N_block_size + ib_N_tile * N_tile_size + ip_out < N) & (i_K_block * K_block_size + if_out < K)
                )
    return out_ref


#Input: X, U, V
# Output: (XUV)^T
@nki.jit
def fused_three_mm_XUV_transpose(
    X_ref,  # Shape: (K, M) 
    U_ref,  # Shape: (M, r)
    V_ref,  # Shape: (r, N)
    mixed_precision=True,
    r_tiles_in_block=8,
    K_tiles_in_block=2,
    N_tiles_in_block=4,
    M_tiles_in_block=4
):
    # Use X_ref dtype as the intermediate tensor dtype
    # Assume all IO tensors have the same dtype
    kernel_dtype = X_ref.dtype
    pe_in_dt = nl.bfloat16 if mixed_precision else np.float32
    assert X_ref.dtype == U_ref.dtype == V_ref.dtype

    # Shape checking - X@U@V computation
    K, M = X_ref.shape # M - 7168
    M_U, r = U_ref.shape
    r_V, N = V_ref.shape # N - 18432
    assert tuple(U_ref.shape) == (M_U, r), "Input shape mismatch!"
    assert tuple(V_ref.shape) == (r_V, N), "Input shape mismatch!"
    assert M == M_U, "M dimension must match between X and U!"
    assert r == r_V, "r dimension must match between U and V!"

    out_ref = nl.ndarray((N, K), dtype=V_ref.dtype, buffer=nl.hbm)


    # Tiling configuration

    M_tile_size = 128
    M_block_size = M_tile_size * M_tiles_in_block
    M_n_blocks = (M + M_block_size - 1) // M_block_size

    r_tile_size = 128
    r_block_size = r_tile_size * r_tiles_in_block
    r_n_blocks = (r + r_block_size - 1) // r_block_size

    K_tile_size = 512
    K_block_size = int(K_tile_size * K_tiles_in_block)
    if K_block_size < K_tile_size:
        K_tile_size = K_block_size
        K_tiles_in_block = 1
    
    if K < K_block_size:
        K_tile_size = K
        K_tiles_in_block = 1
        K_block_size = K_tile_size * K_tiles_in_block

    
    K_n_blocks = (K + K_block_size - 1) // K_block_size

    N_tile_size = 128
    N_block_size = N_tile_size * N_tiles_in_block
    N_n_blocks = (N + N_block_size - 1) // N_block_size

    # Index patterns
    ip_X = nl.arange(M_tile_size)[:, None]
    if_X_tile = nl.arange(K_tile_size)[None, :]
    if_X_block = nl.arange(K_block_size)[None, :]

    ip_V = nl.arange(r_tile_size)[:, None]
    if_V_tile = nl.arange(N_tile_size)[None, :]
    if_V_block = nl.arange(N_block_size)[None, :]

    ip_U = nl.arange(M_tile_size)[:, None]
    if_U_tile = nl.arange(r_tile_size)[None, :]
    if_U_block = nl.arange(r_block_size)[None, :]

    # Main computation loops
    for i_K_block in nl.affine_range(K_n_blocks):  # Loop over K dimension blocks
        # Buffer for intermediate result XU (K x r)
        XU_result_buf = nl.ndarray(
            (r_n_blocks, r_tiles_in_block, par_dim(r_tile_size), K_block_size), dtype=kernel_dtype
        )
        # Loop over r dimension blocks
        for i_M_block in nl.sequential_range(M_n_blocks):
            # X_cache = nl.ndarray((par_dim(M_tile_size), M_tiles_in_block, K_block_size), dtype=pe_in_dt)
            X_cache = load_tensor_block_T(
                X_ref,
                (i_K_block * K_block_size, i_M_block * M_block_size),
                (M_tiles_in_block, M_tile_size, K_block_size),
            )
            for i_r_block in nl.affine_range(r_n_blocks):

                # U_cache = nl.ndarray((par_dim(M_tile_size), M_tiles_in_block, r_block_size), dtype=pe_in_dt)
                U_cache = load_tensor_block(
                    U_ref,
                    (i_M_block * M_block_size, i_r_block * r_block_size),
                    (M_tiles_in_block, M_tile_size, r_block_size),
                )

                for ib_K_tile in nl.affine_range(K_tiles_in_block):
                    for ib_r_tile in nl.affine_range(r_tiles_in_block):
                        # PSUM buffer for X @ U
                        XU_psum = nl.zeros((par_dim(r_tile_size), K_tile_size), dtype=np.float32, buffer=nl.psum)

                        # Index patterns for result
                        if_XU = nl.arange(K_tile_size)[None, :]
                        ip_XU = nl.arange(r_tile_size)[:, None]

                        # Contract over M dimension
                        for ib_M_tile in nl.affine_range(M_tiles_in_block):
                            # Compute X^T @ U (since X is stored transposed)
                            XU_psum[ip_XU, if_XU] += nisa.nc_matmul(
                                moving=X_cache[ip_X, ib_M_tile, if_X_tile + ib_K_tile * K_tile_size],
                                stationary=U_cache[ip_U, ib_M_tile, if_U_tile + ib_r_tile * r_tile_size],
                            )

                        XU_result_buf[i_r_block, ib_r_tile, ip_XU, if_XU + ib_K_tile * K_tile_size] += XU_psum[
                            ip_XU, if_XU
                        ]

        # Loop over N dimension blocks for final result
        for i_N_block in nl.affine_range(N_n_blocks):

            # TODO: Create a final result buffer
            final_result_buf = nl.zeros((N_tiles_in_block, par_dim(N_tile_size), K_block_size), dtype=kernel_dtype)

            if_out = nl.arange(K_block_size)[None, :]
            ip_out = nl.arange(N_tile_size)[:, None]

            for i_r_block in nl.sequential_range(r_n_blocks):
                # Compute (XU) @ V for current blocks

                # V_cache = nl.ndarray((par_dim(r_tile_size), r_tiles_in_block, N_block_size), dtype=pe_in_dt)
                V_cache = load_tensor_block(
                    V_ref,
                    (i_r_block * r_block_size, i_N_block * N_block_size),
                    (r_tiles_in_block, r_tile_size, N_block_size),
                )
                for ib_K_tile in nl.affine_range(K_tiles_in_block):

                    for ib_N_tile in nl.affine_range(N_tiles_in_block):
                        # PSUM buffer for final result
                        XUV_psum = nl.zeros((par_dim(N_tile_size), K_tile_size), dtype=np.float32, buffer=nl.psum)

                        ip_XU_t = nl.arange(r_tile_size)[:, None]
                        if_XU_t = nl.arange(K_tile_size)[None, :]

                        ip_XUV = nl.arange(N_tile_size)[:, None]
                        if_XUV = nl.arange(K_tile_size)[None, :]
                        
                        # Contract over r dimension
                        for ib_r_tile in nl.affine_range(r_tiles_in_block):
                            # Compute XU @ V
                            ip_V_t = nl.arange(r_tile_size)[:, None]
                            if_V_t = nl.arange(N_tile_size)[None, :]

                            XUV_psum[ip_XUV, if_XUV] += nisa.nc_matmul(
                                moving=XU_result_buf[i_r_block, ib_r_tile, ip_XU_t, if_XU_t + ib_K_tile * K_tile_size],
                                stationary=V_cache[ip_V_t, ib_r_tile, if_V_t + ib_N_tile * N_tile_size],
                            )

                        final_result_buf[ib_N_tile, ip_XUV, ib_K_tile * K_tile_size + if_XUV] += XUV_psum[
                            ip_XUV, if_XUV
                        ]
            for ib_N_tile in nl.affine_range(N_tiles_in_block):
                # Store the final result for the current N block
                nl.store(
                    out_ref[
                        i_N_block * N_block_size + ib_N_tile * N_tile_size + ip_out,
                        i_K_block * K_block_size + if_out,
                    ],
                    value=final_result_buf[ib_N_tile, ip_out, if_out],
                    mask=(i_N_block * N_block_size + ib_N_tile * N_tile_size + ip_out < N) &
                          (i_K_block * K_block_size + if_out < K)
                )
    return out_ref


# For mlp down projection
#Input: X^T, U, V
#Ouput: XUV
@nki.jit
def fused_three_mm_XTUV(
    X_ref,  # Shape: (M, K) - stored as transpose
    U_ref,  # Shape: (M, r)
    V_ref,  # Shape: (r, N)
    mixed_precision=True,
    r_tiles_in_block=8,
    K_tiles_in_block=8,
    N_tiles_in_block=2,
    M_tiles_in_block=4
):
    # Use X_ref dtype as the intermediate tensor dtype
    # Assume all IO tensors have the same dtype
    kernel_dtype = X_ref.dtype
    pe_in_dt = nl.bfloat16 if mixed_precision else np.float32
    assert X_ref.dtype == U_ref.dtype == V_ref.dtype

    # Shape checking - X@U@V computation
    M, K = X_ref.shape # M - 7168
    M_U, r = U_ref.shape
    r_V, N = V_ref.shape # N - 18432
    assert tuple(U_ref.shape) == (M_U, r), "Input shape mismatch!"
    assert tuple(V_ref.shape) == (r_V, N), "Input shape mismatch!"
    assert M == M_U, "M dimension must match between X and U!"
    assert r == r_V, "r dimension must match between U and V!"

    out_ref = nl.ndarray((K, N), dtype=V_ref.dtype, buffer=nl.hbm)


    # Tiling configuration

    M_tile_size = 128
    M_block_size = M_tile_size * M_tiles_in_block

    if M < M_block_size:
        M_tile_size = min(M, M_tile_size)
        M_tiles_in_block = 1
        M_block_size = M_tile_size * M_tiles_in_block
    
    M_n_blocks = (M + M_block_size - 1) // M_block_size

    r_tile_size = 128
    r_block_size = r_tile_size * r_tiles_in_block
    r_n_blocks = (r + r_block_size - 1) // r_block_size

    K_tile_size = 128
    K_block_size = int(K_tile_size * K_tiles_in_block)
    if K_block_size < K_tile_size:
        K_tile_size = K_block_size
        K_tiles_in_block = 1
    
    if K < K_block_size:
        K_tile_size = min(K, K_tile_size)
        K_tiles_in_block = 1
        K_block_size = K_tile_size * K_tiles_in_block    

    K_n_blocks = (K + K_block_size - 1) // K_block_size


    N_tile_size = 512
    N_block_size = N_tile_size * N_tiles_in_block

    if N < N_block_size:
        N_tile_size = min(N, N_tile_size)
        N_tiles_in_block = 1
        N_block_size = N_tile_size * N_tiles_in_block

    N_n_blocks = (N + N_block_size - 1) // N_block_size

    # Index patterns
    ip_X = nl.arange(M_tile_size)[:, None]
    if_X_tile = nl.arange(K_tile_size)[None, :]
    if_X_block = nl.arange(K_block_size)[None, :]

    ip_V = nl.arange(r_tile_size)[:, None]
    if_V_tile = nl.arange(N_tile_size)[None, :]
    if_V_block = nl.arange(N_block_size)[None, :]

    ip_U = nl.arange(M_tile_size)[:, None]
    if_U_tile = nl.arange(r_tile_size)[None, :]
    if_U_block = nl.arange(r_block_size)[None, :]

    # Main computation loops
    for i_K_block in nl.affine_range(K_n_blocks):  # Loop over K dimension blocks
        # Buffer for intermediate result XU (K x r)
        XU_result_buf = nl.zeros(
            (r_n_blocks, r_tiles_in_block, par_dim(r_tile_size), K_block_size), dtype=kernel_dtype
        )
        # Loop over r dimension blocks
        for i_M_block in nl.sequential_range(M_n_blocks):
            # X_cache = nl.ndarray((par_dim(M_tile_size), M_tiles_in_block, K_block_size), dtype=pe_in_dt)
            X_cache = load_tensor_block(
                X_ref,
                (i_M_block * M_block_size, i_K_block * K_block_size),
                (M_tiles_in_block, M_tile_size, K_block_size),
            )
            for i_r_block in nl.affine_range(r_n_blocks):

                # U_cache = nl.ndarray((par_dim(M_tile_size), M_tiles_in_block, r_block_size), dtype=pe_in_dt)
                U_cache = load_tensor_block(
                    U_ref,
                    (i_M_block * M_block_size, i_r_block * r_block_size),
                    (M_tiles_in_block, M_tile_size, r_block_size),
                )

                for ib_K_tile in nl.affine_range(K_tiles_in_block):
                    for ib_r_tile in nl.affine_range(r_tiles_in_block):
                        # PSUM buffer for X @ U
                        XU_psum = nl.zeros((par_dim(r_tile_size), K_tile_size), dtype=np.float32, buffer=nl.psum)

                        # Index patterns for result
                        if_XU = nl.arange(K_tile_size)[None, :]
                        ip_XU = nl.arange(r_tile_size)[:, None]

                        # Contract over M dimension
                        for ib_M_tile in nl.affine_range(M_tiles_in_block):
                            # Compute X^T @ U (since X is stored transposed)
                            XU_psum[ip_XU, if_XU] += nisa.nc_matmul(
                                moving=X_cache[ip_X, ib_M_tile, if_X_tile + ib_K_tile * K_tile_size],
                                stationary=U_cache[ip_U, ib_M_tile, if_U_tile + ib_r_tile * r_tile_size],
                            )

                        XU_result_buf[i_r_block, ib_r_tile, ip_XU, if_XU + ib_K_tile * K_tile_size] += XU_psum[
                            ip_XU, if_XU
                        ]

        # Loop over N dimension blocks for final result
        for i_N_block in nl.affine_range(N_n_blocks):

            # TODO: Create a final result buffer
            final_result_buf = nl.zeros((K_tiles_in_block, par_dim(K_tile_size), N_block_size), dtype=kernel_dtype)

            if_out = nl.arange(N_block_size)[None, :]
            ip_out = nl.arange(K_tile_size)[:, None]

            for i_r_block in nl.sequential_range(r_n_blocks):
                # Compute (XU) @ V for current blocks

                # V_cache = nl.ndarray((par_dim(r_tile_size), r_tiles_in_block, N_block_size), dtype=pe_in_dt)
                V_cache = load_tensor_block(
                    V_ref,
                    (i_r_block * r_block_size, i_N_block * N_block_size),
                    (r_tiles_in_block, r_tile_size, N_block_size),
                )
                for ib_K_tile in nl.affine_range(K_tiles_in_block):

                    for ib_N_tile in nl.affine_range(N_tiles_in_block):
                        # PSUM buffer for final result
                        XUV_psum = nl.zeros((par_dim(K_tile_size), N_tile_size), dtype=np.float32, buffer=nl.psum)

                        ip_XU_t = nl.arange(r_tile_size)[:, None]
                        if_XU_t = nl.arange(K_tile_size)[None, :]

                        ip_XUV = nl.arange(K_tile_size)[:, None]
                        if_XUV = nl.arange(N_tile_size)[None, :]
                        
                        # Contract over r dimension
                        for ib_r_tile in nl.affine_range(r_tiles_in_block):
                            # Compute XU @ V
                            ip_V_t = nl.arange(r_tile_size)[:, None]
                            if_V_t = nl.arange(N_tile_size)[None, :]

                            XUV_psum[ip_XUV, if_XUV] += nisa.nc_matmul(
                                moving=V_cache[ip_V_t, ib_r_tile, if_V_t + ib_N_tile * N_tile_size],
                                stationary=XU_result_buf[i_r_block, ib_r_tile, ip_XU_t, if_XU_t + ib_K_tile * K_tile_size],
                            )

                        final_result_buf[ib_K_tile, ip_XUV, ib_N_tile * N_tile_size + if_XUV] += XUV_psum[
                            ip_XUV, if_XUV
                        ]
            for ib_K_tile in nl.affine_range(K_tiles_in_block):
                # Store the final result for the current N block
                nl.store(
                    out_ref[
                        i_K_block * K_block_size + ib_K_tile * K_tile_size + ip_out,
                        i_N_block * N_block_size + if_out,
                    ],
                    value=final_result_buf[ib_K_tile, ip_out, if_out],
                    mask=(i_K_block * K_block_size + ib_K_tile * K_tile_size + ip_out < K) & (i_N_block * N_block_size + if_out < N)
                )
    return out_ref

#Input: X, U, V
#Output: XUV
#For decode stage where the sequence length is 1: use Parameters: {'M_tiles_in_block': 4, 'r_tiles_in_block': 4, 'K_tiles_in_block': 2, 'N_tiles_in_block': 2, 'mixed_precision': True}
#For MLP in LLama 1b (non-decoding stage), use Parameters: {'M_tiles_in_block': 8, 'r_tiles_in_block': 2, 'K_tiles_in_block': 4, 'N_tiles_in_block': 4, 'mixed_precision': True}
@nki.jit
def fused_three_mm_XUV(
    X_ref,  # Shape: (M, K)
    U_ref,  # Shape: (M, r)
    V_ref,  # Shape: (r, N)
    mixed_precision=True,
    is_activation=False,
    r_tiles_in_block=8,
    K_tiles_in_block=8,
    N_tiles_in_block=2,
    M_tiles_in_block=4
):
    # Use X_ref dtype as the intermediate tensor dtype
    # Assume all IO tensors have the same dtype
    kernel_dtype = X_ref.dtype
    pe_in_dt = nl.bfloat16 if mixed_precision else np.float32
    assert X_ref.dtype == U_ref.dtype == V_ref.dtype

    # Shape checking - X@U@V computation
    K, M = X_ref.shape # M - 7168
    M_U, r = U_ref.shape
    r_V, N = V_ref.shape # N - 18432
    assert tuple(U_ref.shape) == (M_U, r), "Input shape mismatch!"
    assert tuple(V_ref.shape) == (r_V, N), "Input shape mismatch!"
    assert M == M_U, "M dimension must match between X and U!"
    assert r == r_V, "r dimension must match between U and V!"

    out_ref = nl.ndarray((K, N), dtype=V_ref.dtype, buffer=nl.hbm)


    # Tiling configuration

    M_tile_size = 128
    M_block_size = M_tile_size * M_tiles_in_block
    M_n_blocks = (M + M_block_size - 1) // M_block_size

    r_tile_size = 128
    r_block_size = r_tile_size * r_tiles_in_block
    r_n_blocks = (r + r_block_size - 1) // r_block_size

    K_tile_size = 128
    K_block_size = int(K_tile_size * K_tiles_in_block)
    if K_block_size < K_tile_size:
        K_tile_size = K_block_size
        K_tiles_in_block = 1
    

    K_n_blocks = (K + K_block_size - 1) // K_block_size


    N_tile_size = 512
    N_block_size = N_tile_size * N_tiles_in_block

    if N < N_block_size:
        N_tile_size = N
        N_tiles_in_block = 1
        N_block_size = N_tile_size * N_tiles_in_block

    N_n_blocks = (N + N_block_size - 1) // N_block_size

    # Index patterns
    ip_X = nl.arange(M_tile_size)[:, None]
    if_X_tile = nl.arange(K_tile_size)[None, :]
    if_X_block = nl.arange(K_block_size)[None, :]

    ip_V = nl.arange(r_tile_size)[:, None]
    if_V_tile = nl.arange(N_tile_size)[None, :]
    if_V_block = nl.arange(N_block_size)[None, :]

    ip_U = nl.arange(M_tile_size)[:, None]
    if_U_tile = nl.arange(r_tile_size)[None, :]
    if_U_block = nl.arange(r_block_size)[None, :]

    # Main computation loops
    for i_K_block in nl.affine_range(K_n_blocks):  # Loop over K dimension blocks
        # Buffer for intermediate result XU (K x r)
        XU_result_buf = nl.zeros(
            (r_n_blocks, r_tiles_in_block, par_dim(r_tile_size), K_block_size), dtype=kernel_dtype
        )
        # Loop over r dimension blocks
        for i_M_block in nl.sequential_range(M_n_blocks):
            # X_cache = nl.ndarray((par_dim(M_tile_size), M_tiles_in_block, K_block_size), dtype=pe_in_dt)
            X_cache = load_tensor_block_T(
                X_ref,
                (i_K_block * K_block_size, i_M_block * M_block_size),
                (M_tiles_in_block, M_tile_size, K_block_size),
            )
            for i_r_block in nl.affine_range(r_n_blocks):

                # U_cache = nl.ndarray((par_dim(M_tile_size), M_tiles_in_block, r_block_size), dtype=pe_in_dt)
                U_cache = load_tensor_block(
                    U_ref,
                    (i_M_block * M_block_size, i_r_block * r_block_size),
                    (M_tiles_in_block, M_tile_size, r_block_size),
                )

                for ib_K_tile in nl.affine_range(K_tiles_in_block):
                    for ib_r_tile in nl.affine_range(r_tiles_in_block):
                        # PSUM buffer for X @ U
                        XU_psum = nl.zeros((par_dim(r_tile_size), K_tile_size), dtype=np.float32, buffer=nl.psum)

                        # Index patterns for result
                        if_XU = nl.arange(K_tile_size)[None, :]
                        ip_XU = nl.arange(r_tile_size)[:, None]

                        # Contract over M dimension
                        for ib_M_tile in nl.affine_range(M_tiles_in_block):
                            # Compute X^T @ U (since X is stored transposed)
                            XU_psum[ip_XU, if_XU] += nisa.nc_matmul(
                                moving=X_cache[ip_X, ib_M_tile, if_X_tile + ib_K_tile * K_tile_size],
                                stationary=U_cache[ip_U, ib_M_tile, if_U_tile + ib_r_tile * r_tile_size],
                            )

                        XU_result_buf[i_r_block, ib_r_tile, ip_XU, if_XU + ib_K_tile * K_tile_size] += XU_psum[
                            ip_XU, if_XU
                        ]

        # Loop over N dimension blocks for final result
        for i_N_block in nl.affine_range(N_n_blocks):

            # TODO: Create a final result buffer
            final_result_buf = nl.zeros((K_tiles_in_block, par_dim(K_tile_size), N_block_size), dtype=kernel_dtype)

            if_out = nl.arange(N_block_size)[None, :]
            ip_out = nl.arange(K_tile_size)[:, None]

            for i_r_block in nl.sequential_range(r_n_blocks):
                # Compute (XU) @ V for current blocks

                # V_cache = nl.ndarray((par_dim(r_tile_size), r_tiles_in_block, N_block_size), dtype=pe_in_dt)
                V_cache = load_tensor_block(
                    V_ref,
                    (i_r_block * r_block_size, i_N_block * N_block_size),
                    (r_tiles_in_block, r_tile_size, N_block_size),
                )
                for ib_K_tile in nl.affine_range(K_tiles_in_block):

                    for ib_N_tile in nl.affine_range(N_tiles_in_block):
                        # PSUM buffer for final result
                        XUV_psum = nl.zeros((par_dim(K_tile_size), N_tile_size), dtype=np.float32, buffer=nl.psum)

                        ip_XU_t = nl.arange(r_tile_size)[:, None]
                        if_XU_t = nl.arange(K_tile_size)[None, :]

                        ip_XUV = nl.arange(K_tile_size)[:, None]
                        if_XUV = nl.arange(N_tile_size)[None, :]
                        
                        # Contract over r dimension
                        for ib_r_tile in nl.affine_range(r_tiles_in_block):
                            # Compute XU @ V
                            ip_V_t = nl.arange(r_tile_size)[:, None]
                            if_V_t = nl.arange(N_tile_size)[None, :]

                            XUV_psum[ip_XUV, if_XUV] += nisa.nc_matmul(
                                moving=V_cache[ip_V_t, ib_r_tile, if_V_t + ib_N_tile * N_tile_size],
                                stationary=XU_result_buf[i_r_block, ib_r_tile, ip_XU_t, if_XU_t + ib_K_tile * K_tile_size],
                            )

                        final_result_buf[ib_K_tile, ip_XUV, ib_N_tile * N_tile_size + if_XUV] += XUV_psum[
                            ip_XUV, if_XUV
                        ]
            for ib_K_tile in nl.affine_range(K_tiles_in_block):
                # Store the final result for the current N block
                nl.store(
                    out_ref[
                        i_K_block * K_block_size + ib_K_tile * K_tile_size + ip_out,
                        i_N_block * N_block_size + if_out,
                    ],
                    value=final_result_buf[ib_K_tile, ip_out, if_out] if not is_activation else nl.silu(
                        final_result_buf[ib_K_tile, ip_out, if_out]
                    ),
                    mask=(i_K_block * K_block_size + ib_K_tile * K_tile_size + ip_out < K) & (i_N_block * N_block_size + if_out < N)
                )
    return out_ref


#Input: X, U, V
# Output: (XUV)^T
@nki.jit
def fused_mlp_up_T(
    X_ref,  # Shape: (K, M) 
    U_ref,  # Shape: (M, r)
    V_ref,  # Shape: (r, N)
    U_ref_1, # Shape: (M, r)
    V_ref_1, # Shape: (r, N)
    mixed_precision=True,
    r_tiles_in_block=8,
    K_tiles_in_block=2,
    N_tiles_in_block=4,
    M_tiles_in_block=4
):
    # Use X_ref dtype as the intermediate tensor dtype
    # Assume all IO tensors have the same dtype
    kernel_dtype = X_ref.dtype
    pe_in_dt = nl.bfloat16 if mixed_precision else np.float32
    assert X_ref.dtype == U_ref.dtype == V_ref.dtype

    # Shape checking - X@U@V computation
    K, M = X_ref.shape # M - 7168
    M_U, r = U_ref.shape
    r_V, N = V_ref.shape # N - 18432
    assert tuple(U_ref.shape) == (M_U, r), "Input shape mismatch!"
    assert tuple(V_ref.shape) == (r_V, N), "Input shape mismatch!"
    assert M == M_U, "M dimension must match between X and U!"
    assert r == r_V, "r dimension must match between U and V!"

    out_ref = nl.ndarray((N, K), dtype=V_ref.dtype, buffer=nl.hbm)


    # Tiling configuration

    M_tile_size = 128
    M_block_size = M_tile_size * M_tiles_in_block
    M_n_blocks = (M + M_block_size - 1) // M_block_size

    r_tile_size = 128
    r_block_size = r_tile_size * r_tiles_in_block
    r_n_blocks = (r + r_block_size - 1) // r_block_size

    K_tile_size = 512
    K_block_size = int(K_tile_size * K_tiles_in_block)
    if K_block_size < K_tile_size:
        K_tile_size = K_block_size
        K_tiles_in_block = 1
    
    if K < K_block_size:
        K_tile_size = min(K, K_tile_size)
        K_tiles_in_block = 1
        K_block_size = K_tile_size * K_tiles_in_block

    
    K_n_blocks = (K + K_block_size - 1) // K_block_size

    N_tile_size = 128
    N_block_size = N_tile_size * N_tiles_in_block
    N_n_blocks = (N + N_block_size - 1) // N_block_size

    # Index patterns
    ip_X = nl.arange(M_tile_size)[:, None]
    if_X_tile = nl.arange(K_tile_size)[None, :]
    if_X_block = nl.arange(K_block_size)[None, :]

    ip_V = nl.arange(r_tile_size)[:, None]
    if_V_tile = nl.arange(N_tile_size)[None, :]
    if_V_block = nl.arange(N_block_size)[None, :]

    ip_U = nl.arange(M_tile_size)[:, None]
    if_U_tile = nl.arange(r_tile_size)[None, :]
    if_U_block = nl.arange(r_block_size)[None, :]

    # Main computation loops
    for i_K_block in nl.affine_range(K_n_blocks):  # Loop over K dimension blocks
        # Buffer for intermediate result XU (K x r)
        XU_result_buf = nl.zeros(
            (r_n_blocks, r_tiles_in_block, par_dim(r_tile_size), K_block_size), dtype=kernel_dtype
        )

        XU_result_buf_1 = nl.zeros(
            (r_n_blocks, r_tiles_in_block, par_dim(r_tile_size), K_block_size), dtype=kernel_dtype
        )
        # Loop over r dimension blocks
        for i_M_block in nl.sequential_range(M_n_blocks):
            # X_cache = nl.ndarray((par_dim(M_tile_size), M_tiles_in_block, K_block_size), dtype=pe_in_dt)
            X_cache = load_tensor_block_T(
                X_ref,
                (i_K_block * K_block_size, i_M_block * M_block_size),
                (M_tiles_in_block, M_tile_size, K_block_size),
            )
            for i_r_block in nl.affine_range(r_n_blocks):

                # U_cache = nl.ndarray((par_dim(M_tile_size), M_tiles_in_block, r_block_size), dtype=pe_in_dt)
                U_cache = load_tensor_block(
                    U_ref,
                    (i_M_block * M_block_size, i_r_block * r_block_size),
                    (M_tiles_in_block, M_tile_size, r_block_size),
                )

                U_cache_1 = load_tensor_block(
                    U_ref_1,
                    (i_M_block * M_block_size, i_r_block * r_block_size),
                    (M_tiles_in_block, M_tile_size, r_block_size),
                )

                for ib_K_tile in nl.affine_range(K_tiles_in_block):
                    for ib_r_tile in nl.affine_range(r_tiles_in_block):
                        # PSUM buffer for X @ U
                        XU_psum = nl.zeros((par_dim(r_tile_size), K_tile_size), dtype=np.float32, buffer=nl.psum)

                        XU_psum_1 = nl.zeros((par_dim(r_tile_size), K_tile_size), dtype=np.float32, buffer=nl.psum)

                        # Index patterns for result
                        if_XU = nl.arange(K_tile_size)[None, :]
                        ip_XU = nl.arange(r_tile_size)[:, None]

                        # Contract over M dimension
                        for ib_M_tile in nl.affine_range(M_tiles_in_block):
                            # Compute X^T @ U (since X is stored transposed)
                            XU_psum[ip_XU, if_XU] += nisa.nc_matmul(
                                moving=X_cache[ip_X, ib_M_tile, if_X_tile + ib_K_tile * K_tile_size],
                                stationary=U_cache[ip_U, ib_M_tile, if_U_tile + ib_r_tile * r_tile_size],
                            )
                            
                            XU_psum_1[ip_XU, if_XU] += nisa.nc_matmul(
                                moving=X_cache[ip_X, ib_M_tile, if_X_tile + ib_K_tile * K_tile_size],
                                stationary=U_cache_1[ip_U, ib_M_tile, if_U_tile + ib_r_tile * r_tile_size],
                            )

                        XU_result_buf[i_r_block, ib_r_tile, ip_XU, if_XU + ib_K_tile * K_tile_size] += XU_psum[
                            ip_XU, if_XU
                        ]

                        XU_result_buf_1[i_r_block, ib_r_tile, ip_XU, if_XU + ib_K_tile * K_tile_size] += XU_psum_1[
                            ip_XU, if_XU
                        ]

        # Loop over N dimension blocks for final result
        for i_N_block in nl.affine_range(N_n_blocks):

            # TODO: Create a final result buffer
            final_result_buf = nl.zeros((N_tiles_in_block, par_dim(N_tile_size), K_block_size), dtype=kernel_dtype)

            final_result_buf_1 = nl.zeros((N_tiles_in_block, par_dim(N_tile_size), K_block_size), dtype=kernel_dtype)

            if_out = nl.arange(K_block_size)[None, :]
            ip_out = nl.arange(N_tile_size)[:, None]

            for i_r_block in nl.sequential_range(r_n_blocks):
                # Compute (XU) @ V for current blocks

                # V_cache = nl.ndarray((par_dim(r_tile_size), r_tiles_in_block, N_block_size), dtype=pe_in_dt)
                V_cache = load_tensor_block(
                    V_ref,
                    (i_r_block * r_block_size, i_N_block * N_block_size),
                    (r_tiles_in_block, r_tile_size, N_block_size),
                )
                
                V_cache_1 = load_tensor_block(
                    V_ref_1,
                    (i_r_block * r_block_size, i_N_block * N_block_size),
                    (r_tiles_in_block, r_tile_size, N_block_size),
                )
                
                for ib_K_tile in nl.affine_range(K_tiles_in_block):

                    for ib_N_tile in nl.affine_range(N_tiles_in_block):
                        # PSUM buffer for final result
                        XUV_psum = nl.zeros((par_dim(N_tile_size), K_tile_size), dtype=np.float32, buffer=nl.psum)

                        XUV_psum_1 = nl.zeros((par_dim(N_tile_size), K_tile_size), dtype=np.float32, buffer=nl.psum)

                        ip_XU_t = nl.arange(r_tile_size)[:, None]
                        if_XU_t = nl.arange(K_tile_size)[None, :]

                        ip_XUV = nl.arange(N_tile_size)[:, None]
                        if_XUV = nl.arange(K_tile_size)[None, :]
                        
                        # Contract over r dimension
                        for ib_r_tile in nl.affine_range(r_tiles_in_block):
                            # Compute XU @ V
                            ip_V_t = nl.arange(r_tile_size)[:, None]
                            if_V_t = nl.arange(N_tile_size)[None, :]

                            XUV_psum[ip_XUV, if_XUV] += nisa.nc_matmul(
                                moving=XU_result_buf[i_r_block, ib_r_tile, ip_XU_t, if_XU_t + ib_K_tile * K_tile_size],
                                stationary=V_cache[ip_V_t, ib_r_tile, if_V_t + ib_N_tile * N_tile_size],
                            )

                            XUV_psum_1[ip_XUV, if_XUV] += nisa.nc_matmul(
                                moving=XU_result_buf_1[i_r_block, ib_r_tile, ip_XU_t, if_XU_t + ib_K_tile * K_tile_size],
                                stationary=V_cache_1[ip_V_t, ib_r_tile, if_V_t + ib_N_tile * N_tile_size],
                            )

                        final_result_buf[ib_N_tile, ip_XUV, ib_K_tile * K_tile_size + if_XUV] += XUV_psum[
                            ip_XUV, if_XUV
                        ]
                        final_result_buf_1[ib_N_tile, ip_XUV, ib_K_tile * K_tile_size + if_XUV] += XUV_psum_1[
                            ip_XUV, if_XUV
                        ]
            for ib_N_tile in nl.affine_range(N_tiles_in_block):
                # Store the final result for the current N block
                nl.store(
                    out_ref[
                        i_N_block * N_block_size + ib_N_tile * N_tile_size + ip_out,
                        i_K_block * K_block_size + if_out,
                    ],
                    value=nl.multiply(nl.silu(final_result_buf[ib_N_tile, ip_out, if_out]), final_result_buf_1[ib_N_tile, ip_out, if_out]),
                    mask=(i_N_block * N_block_size + ib_N_tile * N_tile_size + ip_out < N) &
                          (i_K_block * K_block_size + if_out < K)
                )
    return out_ref



# Main execution block can remain the same as the previous version.
if __name__ == "__main__":

    
    M = 2048
    r = 1280
    N = 8192 

    K = 4096 



    # q_ref = nt.tensor[[d_head, seq_q], nl.bfloat16]
    q_ref_t = nt.tensor[[K, M], nl.bfloat16]
    k_ref = nt.tensor[[M, r], nl.bfloat16]
    v_ref = nt.tensor[[r, N], nl.bfloat16]

    k_ref_1 = nt.tensor[[N, r], nl.bfloat16]
    v_ref_1 = nt.tensor[[r, M], nl.bfloat16]

    k_ref_2 = nt.tensor[[N, r], nl.bfloat16]
    v_ref_2 = nt.tensor[[r, M], nl.bfloat16]


    def benchmark_nki(nki_func, a, b, c, b1=None, c1=None, **kwargs):
        bench_func = nki.benchmark(warmup=5, iters=10)(nki_func)
        if b1 is not None and c1 is not None:
          bench_func(a, b, c, b1, c1, **kwargs)
        else:
          bench_func(a, b, c, True, M_tiles_in_block=8, r_tiles_in_block=2, K_tiles_in_block=4, N_tiles_in_block=4)
        latency_res = bench_func.benchmark_result.nc_latency
        p99 = latency_res.get_latency_percentile(99)
        print("Latency: {:.2f} ms (P99)".format(p99 / 1000.0))
    
    def benchmark_mlp(nki_func, x, u, v, u1, v1, u2, v2, **kwargs):
        bench_func = nki.benchmark(warmup=5, iters=10)(nki_func)
        bench_func(x, u, v, u1, v1, u2, v2, **kwargs)
        latency_res = bench_func.benchmark_result.nc_latency
        p99 = latency_res.get_latency_percentile(99)
        print("Latency: {:.2f} ms (P99)".format(p99 / 1000.0))
    
    args = {"M_tiles_in_block": 8, "r_tiles_in_block": 2, "K_tiles_in_block": 4, "N_tiles_in_block": 4}
    benchmark_mlp(fused_mlp_, q_ref_t, k_ref, v_ref, k_ref_1, v_ref_1, k_ref_2, v_ref_2, **args)
    # benchmark_nki(fused_three_mm_qkv_activation_transpose_block, q_ref, k_ref, v_ref, k_ref, v_ref)
    # benchmark_nki(fused_three_mm_XUV_transpose_block, q_ref, k_ref, v_ref)
    # benchmark_nki(fused_three_mm_XUV_block, q_ref, k_ref, v_ref)
    # benchmark_nki(fused_three_mm_XUV_transpose_block_natural, q_ref_t, k_ref, v_ref)
    benchmark_nki(fused_three_mm_XUV, q_ref_t, k_ref, v_ref)
    # benchmark_nki(fused_three_mm_qkv, q_ref, k_ref, v_ref)
    # benchmark_nki(fused_three_mm_qkv_transpose_block, q_ref, k_ref, v_ref)
    # benchmark_nki(fused_three_mm_qkv_transpose, q_ref, k_ref, v_ref)
    # print("Benchmarking nki_matmul_fully_optimized")
    # # benchmark_nki(fused_qkv_fully_tiled)
    # # benchmark_nki(three_mm_unfused)
    # # benchmark_nki(three_mm_block_fused_1)
    # benchmark_nki(fused_qkv_fully_blocked)
