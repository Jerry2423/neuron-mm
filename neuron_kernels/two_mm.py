"""
AWS Neuron Matrix Multiplication Kernels
Author: Jierui (Jerry) Xu (UW Madison)
Date: 2025.07.06

This file implements progressively optimized matrix multiplication kernels
for AWS NeuronCore hardware with different performance characteristics:
- Basic tiled implementation (nki_matmul_tiled_basic)
- Natural LHS version that handles non-transposed inputs (nki_matmul_tiled_basic_natural_lhs)
- Fully optimized version with multi-level blocking (nki_matmul_fully_optimized_)
- Loop-order optimized version to avoid compiler errors (nki_matmul_fixed_loop_order_)
"""
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import numpy as np
import torch
from torch_xla.core import xla_model as xm
import math



# This kernel should be used as baseline for the NKI matmul
@nki.jit
def nki_matmul_tiled_basic(lhsT, rhs):
  """
  A basic NKI matrix multiplication kernel that uses tiling.

  This kernel can handle large matrices that satisfy specific size
  multiple requirements.
  - lhsT: K and M dimensions must be multiples of 128.
  - rhs: N dimension must be a multiple of 512.

  Args:
      lhsT: The left-hand side operand, which is the transpose of A,
            with shape [K, M].
      rhs: The right-hand side operand, B, with shape [K, N].
  Returns:
      result: The result matrix D, with shape [M, N].
  """
  # --- 1. SETUP PHASE: Define tile sizes and get matrix dimensions ---
  K, M = lhsT.shape
  K_rhs, N = rhs.shape
  assert K == K_rhs, "The contraction dimension K must match for LHS and RHS"

  # Define the size of a "Tile", the basic unit processed by the hardware.
  # This corresponds to the required size multiples for the dimensions.
  TILE_M = 128
  TILE_K = 128
  TILE_N = 512

  # Check if input dimensions meet the requirements
  assert M % TILE_M == 0, f"Dimension M({M}) must be a multiple of {TILE_M}"
  assert K % TILE_K == 0, f"Dimension K({K}) must be a multiple of {TILE_K}"
  assert N % TILE_N == 0, f"Dimension N({N}) must be a multiple of {TILE_N}"

  # Calculate the number of tiles in each dimension
  NUM_TILES_M = M // TILE_M
  NUM_TILES_K = K // TILE_K
  NUM_TILES_N = N // TILE_N

  # Define the final output tensor in the main memory (HBM)
  result = nl.zeros((M, N), dtype=lhsT.dtype, buffer=nl.hbm)

  # --- 2. TILING LOOPS: M -> N -> K ---
  # Iterate over each tile of the output matrix
  for m_tile_idx in nl.affine_range(NUM_TILES_M):
    for n_tile_idx in nl.affine_range(NUM_TILES_N):

      # Create an accumulator for the current output tile (result[m, n]).
      # It resides in PSUM for efficient accumulation operations.
      result_tile_psum = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)

      # Reduction loop: Iterate over all tiles along the K dimension.
      for k_tile_idx in nl.sequential_range(NUM_TILES_K):
        
        # Calculate the offset for the current tile in the large matrix
        m_offset = m_tile_idx * TILE_M
        n_offset = n_tile_idx * TILE_N
        k_offset = k_tile_idx * TILE_K

        # a. Load one tile of LHS from HBM to SBUF
        lhs_tile_sbuf = nl.load(
            lhsT[k_offset : k_offset + TILE_K,
                 m_offset : m_offset + TILE_M]
        )

        # b. Load one tile of RHS from HBM to SBUF
        rhs_tile_sbuf = nl.load(
            rhs[k_offset : k_offset + TILE_K,
                n_offset : n_offset + TILE_N]
        )

        # c. Perform matmul on the currently loaded tiles and accumulate the result
        #    into the PSUM accumulator.
        result_tile_psum[...] += nl.matmul(lhs_tile_sbuf, rhs_tile_sbuf, transpose_x=True)

      # d. After the reduction loop over K finishes, the PSUM accumulator
      #    holds the final result for this tile.
      # Write this final tile result back to the correct location in HBM.
      nl.store(
          result[m_tile_idx * TILE_M : (m_tile_idx + 1) * TILE_M,
                 n_tile_idx * TILE_N : (n_tile_idx + 1) * TILE_N],
          result_tile_psum
      )

  return result


@nki.jit
def nki_matmul_tiled_basic_natural_lhs(lhs, rhs):
  """
  A basic, tiled NKI matmul kernel that accepts a standard (non-transposed)
  LHS matrix.

  This kernel handles the required transpose operation internally, which is
  less optimal than providing a pre-transposed LHS.

  Args:
      lhs: The left-hand side operand A, with shape [M, K].
      rhs: The right-hand side operand B, with shape [K, N].
  Returns:
      result: The result matrix D, with shape [M, N].
  """
  # --- 1. SETUP PHASE: Define tile sizes and get matrix dimensions ---
  M, K = lhs.shape
  K_rhs, N = rhs.shape
  assert K == K_rhs, "The contraction dimension K must match for LHS and RHS"

  # Define the size of a "Tile", the basic unit processed by the hardware.
  TILE_M = 128
  TILE_K = 128
  TILE_N = 512

  # Check if input dimensions are multiples of tile sizes
  assert M % TILE_M == 0, f"Dimension M({M}) must be a multiple of {TILE_M}"
  assert K % TILE_K == 0, f"Dimension K({K}) must be a multiple of {TILE_K}"
  assert N % TILE_N == 0, f"Dimension N({N}) must be a multiple of {TILE_N}"

  # Calculate the number of tiles in each dimension
  NUM_TILES_M = M // TILE_M
  NUM_TILES_K = K // TILE_K
  NUM_TILES_N = N // TILE_N

  # Define the final output tensor in the main memory (HBM)
  result = nl.zeros((M, N), dtype=lhs.dtype, buffer=nl.hbm)

  # --- 2. TILING LOOPS: M -> N -> K ---
  # Iterate over each tile of the output matrix
  for m_tile_idx in nl.affine_range(NUM_TILES_M):
    for n_tile_idx in nl.affine_range(NUM_TILES_N):

      # Create an accumulator for the current output tile in PSUM
      result_tile_psum = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)

      # Reduction loop: Iterate over all tiles along the K dimension.
      for k_tile_idx in nl.sequential_range(NUM_TILES_K):
        
        # Calculate the offset for the current tile
        m_offset = m_tile_idx * TILE_M
        n_offset = n_tile_idx * TILE_N
        k_offset = k_tile_idx * TILE_K

        # a. Load one tile of LHS (shape TILE_M, TILE_K) from HBM.
        lhs_tile_sbuf = nl.load(
            lhs[m_offset : m_offset + TILE_M,
                k_offset : k_offset + TILE_K]
        )
        

        # b. Load one tile of RHS from HBM to SBUF
        rhs_tile_sbuf = nl.load(
            rhs[k_offset : k_offset + TILE_K,
                n_offset : n_offset + TILE_N]
        )


        # c. Perform matmul with the correctly transposed LHS tile
        result_tile_psum[...] += nl.matmul(lhs_tile_sbuf , rhs_tile_sbuf)

      # d. After the reduction loop, store the final tile result
      nl.store(
          result[m_offset : m_offset + TILE_M,
                 n_offset : n_offset + TILE_N],
          result_tile_psum
      )

  return result


# NKI kernel from the NKI tutorial, SBUF will spill if M dimension is greater than 8192 
@nki.jit
def nki_matmul_fully_optimized_(
    lhsT,
    rhs,
    # Meta-parameters
    TILES_IN_BLOCK_M=8,
    TILES_IN_BLOCK_N=2,
    TILES_IN_BLOCK_K=8,
):
  """NKI kernel to compute a large matrix multiplication efficiently by
     blocking all dimensions and doing layout optimization.

  Args:
      lhsT: an input tensor of shape [K,M], where K is a multiple of 128 *
        TILES_IN_BLOCK_K and M is a multiple of 128 * TILES_IN_BLOCK_M.  It is the
        left-hand-side argument of the matrix multiplication, delivered transposed
        for optimal performance.
      rhs: an input tensor of shape [K,N],  where K is a multiple of 128 *
        TILES_IN_BLOCK_K and N is a multiple of 512 * TILES_IN_BLOCK_N.  It is
        the right-hand-side argument of the matrix multiplication.
      TILES_IN_BLOCK_*: meta parameters to control blocking dimensions
  Returns:
      result: the resulting output tensor of shape [M,N]
  """

  K, M = lhsT.shape
  K_, N = rhs.shape
  assert K == K_, "lhsT and rhs must have the same contraction dimension"
  result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

  TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
  TILE_K = nl.tile_size.pmax  # 128
  TILE_N = nl.tile_size.gemm_moving_fmax  # 512

  if M < TILE_M:
    TILE_M = M
    TILES_IN_BLOCK_M = 1
  else:
    TILES_IN_BLOCK_M = min(TILES_IN_BLOCK_M, M // TILE_M)
  
  if N < TILE_N:
    TILE_N = N
    TILES_IN_BLOCK_N = 1
  else:
    TILES_IN_BLOCK_N = min(TILES_IN_BLOCK_N, N // TILE_N)
  
  if K < TILE_K:
    TILE_K = K
    TILES_IN_BLOCK_K = 1
  else:
    TILES_IN_BLOCK_K = min(TILES_IN_BLOCK_K, K // TILE_K)

  BLOCK_M = TILE_M * TILES_IN_BLOCK_M
  BLOCK_N = TILE_N * TILES_IN_BLOCK_N
  BLOCK_K = TILE_K * TILES_IN_BLOCK_K

  NUM_BLOCK_M = (M + BLOCK_M - 1) // BLOCK_M
  NUM_BLOCK_N = (N + BLOCK_N - 1) // BLOCK_N
  NUM_BLOCK_K = (K + BLOCK_K - 1) // BLOCK_K

  # Blocking N dimension (the RHS free dimension)
  for n in nl.affine_range(NUM_BLOCK_N):
    result_tiles = nl.zeros((NUM_BLOCK_M, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N,
                             nl.par_dim(TILE_M), TILE_N),
                            dtype=lhsT.dtype,
                            buffer=nl.sbuf)

    # Blocking K dimension (the contraction dimension)
    for k in nl.sequential_range(NUM_BLOCK_K):
      # Loading tiles from rhs
      i_rhs = nl.mgrid[0:TILE_K, 0:BLOCK_N]
      rhs_tiles = nl.zeros((TILES_IN_BLOCK_K, nl.par_dim(TILE_K), BLOCK_N),
                           dtype=rhs.dtype,
                           buffer=nl.sbuf)

      for bk_r in nl.affine_range(TILES_IN_BLOCK_K):
        k_index = (k * TILES_IN_BLOCK_K + bk_r) * TILE_K
        rhs_tiles[bk_r, i_rhs.p, i_rhs.x] = nl.load(
            rhs[k_index + i_rhs.p, BLOCK_N * n + i_rhs.x], 
            mask=(k_index + i_rhs.p < K) & (BLOCK_N * n + i_rhs.x < N))

      # Blocking M dimension (the LHS free dimension)
      for m in nl.affine_range(NUM_BLOCK_M):
        # Loading tiles from lhsT
        i_lhsT = nl.mgrid[0:TILE_K, 0:BLOCK_M]
        lhsT_tiles = nl.zeros((TILES_IN_BLOCK_K, nl.par_dim(TILE_K), BLOCK_M),
                              dtype=lhsT.dtype,
                              buffer=nl.sbuf)
        
        for bk_l in nl.affine_range(TILES_IN_BLOCK_K):
          k_index = (k * TILES_IN_BLOCK_K + bk_l) * TILE_K
          lhsT_tiles[bk_l, i_lhsT.p, i_lhsT.x] = nl.load(
              lhsT[k_index + i_lhsT.p, BLOCK_M * m + i_lhsT.x], 
              mask=(k_index + i_lhsT.p < K) & (BLOCK_M * m + i_lhsT.x < M))

        # Do matmul with all tiles in the blocks
        i_lhsT_mm = nl.mgrid[0:TILE_K, 0:TILE_M]
        i_rhs_mm = nl.mgrid[0:TILE_K, 0:TILE_N]
        i_res_mm = nl.mgrid[0:TILE_M, 0:TILE_N]
        
        for bn in nl.affine_range(TILES_IN_BLOCK_N):
          for bm in nl.affine_range(TILES_IN_BLOCK_M):
            res_tile = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)

            for bk in nl.affine_range(TILES_IN_BLOCK_K):
              k_index = (k * TILES_IN_BLOCK_K + bk) * TILE_K
              res_tile[...] += nisa.nc_matmul(
                  lhsT_tiles[bk, i_lhsT_mm.p, bm * TILE_M + i_lhsT_mm.x],
                  rhs_tiles[bk, i_rhs_mm.p, bn * TILE_N + i_rhs_mm.x])

            # Accumulate on corresponding SBUF tile
            result_tiles[m, bm, bn, i_res_mm.p,
                         i_res_mm.x] += res_tile[i_res_mm.p, i_res_mm.x]

    # Copying the result from SBUF to HBM
    for m in nl.affine_range(NUM_BLOCK_M):
      for bm in nl.affine_range(TILES_IN_BLOCK_M):
        i_res = nl.mgrid[0:TILE_M, 0:TILE_N]
        i_res_packed = nl.mgrid[0:TILE_M, 0:BLOCK_N]
        result_packed = nl.zeros((TILE_M, BLOCK_N),
                                 dtype=result_tiles.dtype,
                                 buffer=nl.sbuf)

        # coalesce result tiles for better DMA performance
        for bn in nl.affine_range(TILES_IN_BLOCK_N):
          result_packed[i_res.p,
                        bn * TILE_N + i_res.x] = nl.copy(result_tiles[m, bm, bn,
                                                                      i_res.p,
                                                                      i_res.x])
        nl.store(result[(TILES_IN_BLOCK_M * m + bm) * TILE_M + i_res_packed.p,
                        BLOCK_N * n + i_res_packed.x],
                 value=result_packed[i_res_packed.p, i_res_packed.x], 
                 mask=((TILES_IN_BLOCK_M * m + bm) * TILE_M + i_res_packed.p < M) & 
                      (BLOCK_N * n + i_res_packed.x < N))

  return result


# Compute a complete block of output matrix (instead of a partial sum block)
@nki.jit
def nki_matmul_fixed_loop_order_(
    lhsT,
    rhs,
    # Meta-parameters
    TILES_IN_BLOCK_M=16,
    TILES_IN_BLOCK_N=2,
    TILES_IN_BLOCK_K=8,
    TILE_M=nl.tile_size.gemm_stationary_fmax,
    TILE_K=nl.tile_size.pmax,
    TILE_N=nl.tile_size.gemm_moving_fmax, 
):
  """
  NKI kernel to compute a large matrix multiplication with a corrected
  n -> m -> k loop order. This structure separates the reduction loop (k)
  from the memory store operations, resolving the `TEN404` compiler error.

  Args:
      lhsT: an input tensor of shape [K,M]. It is the left-hand-side
        argument of the matrix multiplication, delivered transposed.
      rhs: an input tensor of shape [K,N]. It is the right-hand-side
        argument of the matrix multiplication.
      TILES_IN_BLOCK_*: meta parameters to control blocking dimensions.
  Returns:
      result: the resulting output tensor of shape [M,N].
  """

  K, M = lhsT.shape
  K_, N = rhs.shape
  assert K == K_, "lhsT and rhs must have the same contraction dimension"
  result = nl.zeros((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

  # TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
  # TILE_K = nl.tile_size.pmax  # 128
  # TILE_N = nl.tile_size.gemm_moving_fmax  # 512

  BLOCK_M = TILE_M * TILES_IN_BLOCK_M
  BLOCK_N = TILE_N * TILES_IN_BLOCK_N
  BLOCK_K = TILE_K * TILES_IN_BLOCK_K

  # the size has to be multiple of block size
  assert M % BLOCK_M == 0
  assert N % BLOCK_N == 0
  assert K % BLOCK_K == 0

  NUM_BLOCK_M = M // BLOCK_M
  NUM_BLOCK_N = N // BLOCK_N
  NUM_BLOCK_K = K // BLOCK_K

  # Correct loop order: n -> m -> k
  # This ensures the reduction (k-loop) is isolated.
  for n in nl.affine_range(NUM_BLOCK_N):
    for m in nl.affine_range(NUM_BLOCK_M):
      # === SETUP PHASE ===
      # Create an SBUF accumulator for the entire (m, n) block.
      # It will be used to sum the results over the K dimension.
      res_accumulator = nl.zeros((TILES_IN_BLOCK_M, TILES_IN_BLOCK_N,
                                  nl.par_dim(TILE_M), TILE_N),
                                  dtype=lhsT.dtype,
                                  buffer=nl.sbuf)

      # === REDUCTION PHASE ===
      # This is now a pure reduction loop over the K dimension.
      # No storing to HBM happens inside this loop.
      for k in nl.sequential_range(NUM_BLOCK_K):
        # Load a block of rhs [k, n] into SBUF
        i_rhs = nl.mgrid[0:TILE_K, 0:BLOCK_N]
        rhs_block_sbuf = nl.ndarray((TILES_IN_BLOCK_K, nl.par_dim(TILE_K), BLOCK_N),
                                    dtype=rhs.dtype,
                                    buffer=nl.sbuf)
        for bk_r in nl.affine_range(TILES_IN_BLOCK_K):
          rhs_block_sbuf[bk_r, i_rhs.p, i_rhs.x] = nl.load(
              rhs[(k * TILES_IN_BLOCK_K + bk_r) * TILE_K + i_rhs.p,
                  n * BLOCK_N + i_rhs.x])

        # Load a block of lhsT [k, m] into SBUF
        i_lhsT = nl.mgrid[0:TILE_K, 0:BLOCK_M]
        lhsT_block_sbuf = nl.ndarray((TILES_IN_BLOCK_K, nl.par_dim(TILE_K), BLOCK_M),
                                     dtype=lhsT.dtype,
                                     buffer=nl.sbuf)
        for bk_l in nl.affine_range(TILES_IN_BLOCK_K):
          lhsT_block_sbuf[bk_l, i_lhsT.p, i_lhsT.x] = nl.load(
              lhsT[(k * TILES_IN_BLOCK_K + bk_l) * TILE_K + i_lhsT.p,
                   m * BLOCK_M + i_lhsT.x])

        # Perform matmul on tiles and accumulate in SBUF
        i_lhsT_mm = nl.mgrid[0:TILE_K, 0:TILE_M]
        i_rhs_mm = nl.mgrid[0:TILE_K, 0:TILE_N]
        i_res_mm = nl.mgrid[0:TILE_M, 0:TILE_N]
        
        for bn in nl.affine_range(TILES_IN_BLOCK_N):
          for bm in nl.affine_range(TILES_IN_BLOCK_M):
            res_tile = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
            for bk in nl.affine_range(TILES_IN_BLOCK_K):
              res_tile[...] += nisa.nc_matmul(
                  lhsT_block_sbuf[bk, i_lhsT_mm.p, bm * TILE_M + i_lhsT_mm.x],
                  rhs_block_sbuf[bk, i_rhs_mm.p, bn * TILE_N + i_rhs_mm.x])

            # Accumulate on the main SBUF accumulator
            res_accumulator[bm, bn, i_res_mm.p, i_res_mm.x] += res_tile[i_res_mm.p, i_res_mm.x]

      # === STORE PHASE ===
      # The k-loop is complete. Now, store the final result for the (m, n) block
      # from the SBUF accumulator back to HBM.
      i_packed = nl.mgrid[0:TILE_M, 0:TILE_N]
      i_store_slice = nl.mgrid[0:TILE_M, 0:BLOCK_N]
      
      for bm in nl.affine_range(TILES_IN_BLOCK_M):
        # Create a temporary contiguous buffer for one row-of-tiles
        result_packed = nl.ndarray((TILE_M, BLOCK_N),
                                   dtype=result.dtype,
                                   buffer=nl.sbuf)
        
        # Pack tiles from the accumulator into the contiguous buffer
        for bn in nl.affine_range(TILES_IN_BLOCK_N):
          result_packed[i_packed.p, bn * TILE_N + i_packed.x] = nl.copy(
              res_accumulator[bm, bn, i_packed.p, i_packed.x])
        
        # Store the packed row-of-tiles to the correct HBM location
        nl.store(
            result[(m * BLOCK_M + bm * TILE_M) + i_store_slice.p,
                   n * BLOCK_N + i_store_slice.x],
            value=result_packed[i_store_slice.p, i_store_slice.x]
        )

  return result


@nki.jit
def nki_matmul_tiled_basic_masked(lhsT, rhs):
  """
  A basic NKI matrix multiplication kernel that uses tiling.

  This kernel can handle large matrices that satisfy specific size
  multiple requirements.
  - lhsT: K and M dimensions must be multiples of 128.
  - rhs: N dimension must be a multiple of 512.

  Args:
      lhsT: The left-hand side operand, which is the transpose of A,
            with shape [K, M].
      rhs: The right-hand side operand, B, with shape [K, N].
  Returns:
      result: The result matrix D, with shape [M, N].
  """
  # --- 1. SETUP PHASE: Define tile sizes and get matrix dimensions ---
  K, M = lhsT.shape
  K_rhs, N = rhs.shape
  assert K == K_rhs, "The contraction dimension K must match for LHS and RHS"

  # Define the size of a "Tile", the basic unit processed by the hardware.
  # This corresponds to the required size multiples for the dimensions.
  TILE_M = 128
  TILE_K = 128
  TILE_N = 512


  # Calculate the number of tiles in each dimension
  NUM_TILES_M = math.ceil(M / TILE_M)
  NUM_TILES_K = math.ceil(K / TILE_K)
  NUM_TILES_N = math.ceil(N / TILE_N)

  # Define the final output tensor in the main memory (HBM)
  result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.hbm)

  # --- 2. TILING LOOPS: M -> N -> K ---
  # Iterate over each tile of the output matrix
  for m_tile_idx in nl.affine_range(NUM_TILES_M):
    for n_tile_idx in nl.affine_range(NUM_TILES_N):

      # Create an accumulator for the current output tile (result[m, n]).
      # It resides in PSUM for efficient accumulation operations.
      result_tile_psum = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)

      # Reduction loop: Iterate over all tiles along the K dimension.
      for k_tile_idx in nl.sequential_range(NUM_TILES_K):
        
        # Calculate the offset for the current tile in the large matrix
        m_offset = m_tile_idx * TILE_M
        n_offset = n_tile_idx * TILE_N
        k_offset = k_tile_idx * TILE_K

        # a. Load one tile of LHS from HBM to SBUF
        i_p_lhs, i_f_lhs = nl.mgrid[0:TILE_K, 0:TILE_M]
        lhs_tile_sbuf = nl.load(
            lhsT[k_offset+i_p_lhs,
                 m_offset+i_f_lhs], mask= (k_offset+i_p_lhs < K) & (m_offset+i_f_lhs < M)
        )

        # b. Load one tile of RHS from HBM to SBUF
        i_p_rhs, i_f_rhs = nl.mgrid[0:TILE_K, 0:TILE_N]
        rhs_tile_sbuf = nl.load(
            rhs[k_offset + i_p_rhs,
                n_offset + i_f_rhs], mask=(k_offset + i_p_rhs < K) & (n_offset + i_f_rhs < N)
        )

        # c. Perform matmul on the currently loaded tiles and accumulate the result
        #    into the PSUM accumulator.
        result_tile_psum[...] += nl.matmul(lhs_tile_sbuf, rhs_tile_sbuf, transpose_x=True)

      # d. After the reduction loop over K finishes, the PSUM accumulator
      #    holds the final result for this tile.
      # Write this final tile result back to the correct location in HBM.
      i_p_result, i_f_result = nl.mgrid[0:TILE_M, 0:TILE_N]
      result_m_offset = m_tile_idx * TILE_M
      result_n_offset = n_tile_idx * TILE_N
      nl.store(
          result[result_m_offset + i_p_result,
                 result_n_offset + i_f_result],
          result_tile_psum, mask=(result_m_offset + i_p_result < M) & (result_n_offset + i_f_result < N)
      )

  return result

@nki.jit
def nki_matmul_tiled_basic_natural_lhs_masked(lhs, rhs):
  """
  A basic, tiled NKI matmul kernel that accepts a standard (non-transposed)
  LHS matrix.

  This kernel handles the required transpose operation internally, which is
  less optimal than providing a pre-transposed LHS.

  Args:
      lhs: The left-hand side operand A, with shape [M, K].
      rhs: The right-hand side operand B, with shape [K, N].
  Returns:
      result: The result matrix D, with shape [M, N].
  """
  # --- 1. SETUP PHASE: Define tile sizes and get matrix dimensions ---
  M, K = lhs.shape
  K_rhs, N = rhs.shape
  assert K == K_rhs, "The contraction dimension K must match for LHS and RHS"

  # Define the size of a "Tile", the basic unit processed by the hardware.
  TILE_M = 128
  TILE_K = 128
  TILE_N = 512

  # Calculate the number of tiles in each dimension
  NUM_TILES_M = math.ceil(M / TILE_M)
  NUM_TILES_K = math.ceil(K / TILE_K)
  NUM_TILES_N = math.ceil(N / TILE_N)

  # Define the final output tensor in the main memory (HBM)
  result = nl.ndarray((M, N), dtype=lhs.dtype, buffer=nl.hbm)

  # --- 2. TILING LOOPS: M -> N -> K ---
  # Iterate over each tile of the output matrix
  for m_tile_idx in nl.affine_range(NUM_TILES_M):
    for n_tile_idx in nl.affine_range(NUM_TILES_N):

      # Create an accumulator for the current output tile in PSUM
      result_tile_psum = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
      m_offset = m_tile_idx * TILE_M
      n_offset = n_tile_idx * TILE_N

      # Reduction loop: Iterate over all tiles along the K dimension.
      for k_tile_idx in nl.sequential_range(NUM_TILES_K):
        
        # Calculate the offset for the current tile
        k_offset = k_tile_idx * TILE_K

        # a. Load one tile of LHS (shape TILE_M, TILE_K) from HBM.
        i_p_lhs, i_f_lhs = nl.mgrid[0:TILE_M, 0:TILE_K]
        lhs_tile_sbuf = nl.zeros((TILE_M, TILE_K), dtype=lhs.dtype, buffer=nl.sbuf)
        lhs_tile_sbuf[i_p_lhs, i_f_lhs] = nl.load(
            lhs[m_offset + i_p_lhs,
                k_offset + i_f_lhs], mask=(m_offset + i_p_lhs < M) & (k_offset + i_f_lhs < K)
        )
        

        # b. Load one tile of RHS from HBM to SBUF
        i_p_rhs, i_f_rhs = nl.mgrid[0:TILE_K, 0:TILE_N]
        rhs_tile_sbuf = nl.zeros((TILE_K, TILE_N), dtype=rhs.dtype, buffer=nl.sbuf)
        rhs_tile_sbuf[i_p_rhs, i_f_rhs] = nl.load(
            rhs[k_offset + i_p_rhs,
                n_offset + i_f_rhs], mask=(k_offset + i_p_rhs < K) & (n_offset + i_f_rhs < N)
        )


        # c. Perform matmul with the correctly transposed LHS tile
        result_tile_psum[...] += nl.matmul(lhs_tile_sbuf , rhs_tile_sbuf)

      # d. After the reduction loop, store the final tile result
      i_p_result, i_f_result = nl.mgrid[0:TILE_M, 0:TILE_N]
      nl.store(
          result[m_offset + i_p_result,
                 n_offset + i_f_result],
          result_tile_psum, mask=(m_offset + i_p_result < M) & (n_offset + i_f_result < N)
      )

  return result

# NKI_EXAMPLE_23_BEGIN
if __name__ == "__main__":
  # Benchmarking with large matrices to show the differences more clearly
  lhsT = nt.tensor[[1280, 64], nl.bfloat16]
  rhs = nt.tensor[[1280, 512], nl.bfloat16]

  # lhsT = np.random.randn(8192, 4096)
  # lhsT = lhsT.astype(np.float16)
  # rhs = np.random.randn(8192, 8192)
  # rhs = rhs.astype(np.float16)

  # device = xm.xla_device()
  # lhsT = torch.randn((8192, 4096), dtype=torch.bfloat16, device=device)
  # rhs = torch.randn((8192, 8192), dtype=torch.bfloat16, device=device) 

  # nki.simulate_kernel(nki_matmul_fully_optimized_, lhsT, rhs)
  # nki_matmul_fully_optimized_(lhsT, rhs)

  def benchmark_nki(nki_func, a, b):
    bench_func = nki.benchmark(warmup=5, iters=10)(nki_func)
    bench_func(a, b)
    latency_res = bench_func.benchmark_result.nc_latency
    p99 = latency_res.get_latency_percentile(99)
    print("Latency: {:.2f} ms (P99)".format(p99 / 1000.0))

  
  print("Benchmarking nki_matmul_fully_optimized")
  benchmark_nki(nki_matmul_fully_optimized_, lhsT, rhs)
  # NKI_EXAMPLE_23_END
