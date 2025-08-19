from typing import Tuple

import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import numpy as np

def load_tensor_block(input_tensor, ofs: Tuple[int, int], load_shape: Tuple[int, nl.par_dim, int]):
    """
    Load a 2D rectangle region from the input HBM tensor to SBUF.
    The location of the 2D region is offset by (ofs[0], ofs[1]) at its upper left corner.
    The size of the 2D region to load into SBUF is (block_size * par_size, free_size).
    Load the input HBM tensor by (par_size, free_size) tiles in parallel in the block dimension.
    Output SBUF tensor has a shape of (block_size, par_size, free_size).

    +------------------+
    |                  |
    |    +--------+    |  ← Starting at (ofs[0], ofs[1])
    |    |Tile 0  |    |
    |    |Tile 1  |    |  Each tile is (par_size * free_size)
    |    |  ...   |    |
    |    |Tile N-1|    |  N = block_size
    |    +--------+    |
    |                  |
    +------------------+

    Args:
        input_tensor: the input 2D HBM tensor
        ofs: location offsets in the 2D HBM tensor dimensions
        load_shape: (par_dim(par_size), block_sizeblock_size, free_size)

    Returns:
        Loaded tiles in SBUF in the shape of load_shape
    """
    assert len(ofs) == 2, f"'ofs' expects (ofs_0, ofs_1). Received {ofs}."
    assert len(load_shape) == 3, f"'load_shape' expects (block, par, free). Received {load_shape}."
    max_rows, max_cols = input_tensor.shape
    load_block_size, load_par_size, load_free_size = load_shape
    tile_index = nl.mgrid[0:load_par_size, 0:load_free_size]
    loaded_tensor = nl.zeros(
        (nl.par_dim(load_par_size), load_block_size, load_free_size), dtype=input_tensor.dtype, buffer=nl.sbuf
    )
    for block_id in nl.affine_range(load_block_size):
        row_indices = ofs[0] + block_id * load_par_size + tile_index.p
        col_indices = ofs[1] + tile_index.x
        loaded_tensor[tile_index.p, block_id, tile_index.x] = nl.load(
            input_tensor[row_indices, col_indices], mask=(row_indices < max_rows) & (col_indices < max_cols)
        )
    return loaded_tensor

def load_tensor_block_T(input_tensor, ofs: Tuple[int, int], load_shape: Tuple[int, nl.par_dim, int]):
    """
    Load a 2D rectangle region from the input HBM tensor to SBUF.
    The location of the 2D region is offset by (ofs[0], ofs[1]) at its upper left corner.
    The size of the 2D region to load into SBUF is (block_size * par_size, free_size).
    Load the input HBM tensor by (par_size, free_size) tiles in parallel in the block dimension.
    Output SBUF tensor has a shape of (block_size, par_size, free_size).

    +------------------+
    |                  |
    |    +--------+    |  ← Starting at (ofs[0], ofs[1])
    |    |Tile 0  |    |
    |    |Tile 1  |    |  Each tile is (par_size * free_size)
    |    |  ...   |    |
    |    |Tile N-1|    |  N = block_size
    |    +--------+    |
    |                  |
    +------------------+

    Args:
        input_tensor: the input 2D HBM tensor
        ofs: location offsets in the 2D HBM tensor dimensions
        load_shape: (par_dim(par_size), block_sizeblock_size, free_size)

    Returns:
        Loaded tiles in SBUF in the shape of load_shape
    """
    assert len(ofs) == 2, f"'ofs' expects (ofs_0, ofs_1). Received {ofs}."
    assert len(load_shape) == 3, f"'load_shape' expects (block, par, free). Received {load_shape}."
    max_rows, max_cols = input_tensor.shape
    load_block_size, load_par_size, load_free_size = load_shape
    tile_index = nl.mgrid[0:load_par_size, 0:load_free_size]
    tile_index_T = nl.mgrid[0:load_free_size, 0:load_par_size]
    loaded_tensor = nl.zeros(
        (nl.par_dim(load_par_size), load_block_size, load_free_size), dtype=input_tensor.dtype, buffer=nl.sbuf
    )
    for block_id in nl.affine_range(load_block_size):
        row_indices = ofs[0]  + tile_index_T.p
        col_indices = ofs[1] + tile_index_T.x + block_id * load_par_size
        loaded_tensor[tile_index.p, block_id, tile_index.x] = nl.load_transpose2d(
            input_tensor[row_indices, col_indices], mask=(row_indices < max_rows) & (col_indices < max_cols)
        )
    return loaded_tensor

def matmul_block_T(lhsT_block, rhs_block, result_block):
    """
    Accumulate matmul result tiles between lhsT and rhs into result_block

    Args:
    lhsT_block: TILES_IN_BLOCK_K, TILE_K, BLOCK_M
    rhs_block: TILES_IN_BLOCK_K, TILE_K, BLOCK_N
    result_block : , TILES_IN_BLOCK_N, TILE_N, BLOCK_M
    """
    TILES_IN_BLOCK_K, TILE_K, BLOCK_M = lhsT_block.shape
    _TILES_IN_BLOCK_K, _TILE_K, BLOCK_N = rhs_block.shape
    TILES_IN_BLOCK_M, TILES_IN_BLOCK_N, TILE_M, TILE_N = result_block.shape
    # XU_result_buf = nl.ndarray((par_dim(r_tile_size), r_n_tiles, K_block_size), dtype=kernel_dtype)

    # Data checks
    assert (
        TILES_IN_BLOCK_K == _TILES_IN_BLOCK_K and TILE_K == _TILE_K
    ), f"lhsT_block {lhsT_block.shape} does not match with rhs_block {rhs_block.shape}"
    assert (
        BLOCK_M == TILES_IN_BLOCK_M * TILE_M and BLOCK_N == TILES_IN_BLOCK_N * TILE_N
    ), f"lhsT_block {lhsT_block.shape} does not match with result_block {result_block.shape}"

    idx_lhsT = nl.mgrid[0:TILE_K, 0:TILE_M]
    idx_rhs = nl.mgrid[0:TILE_K, 0:TILE_N]
    idx_res = nl.mgrid[0:TILE_N, 0:TILE_M]
    for tile_id_M in nl.affine_range(TILES_IN_BLOCK_M):
        for tile_id_N in nl.affine_range(TILES_IN_BLOCK_N):
            result_tile = nl.zeros((TILE_N, TILE_M), dtype=nl.float32, buffer=nl.psum)
            """
            Use PSUM buffer to accumulate into a single hardware tile
            """
            for tile_id_K in nl.affine_range(TILES_IN_BLOCK_K):
                result_tile += nisa.nc_matmul(
                    rhs_block[tile_id_K, idx_rhs.p, tile_id_N * TILE_N + idx_rhs.x],
                    lhsT_block[tile_id_K, idx_lhsT.p, tile_id_M * TILE_M + idx_lhsT.x]
                )

            result_block[tile_id_M, tile_id_N, idx_res.p, idx_res.x] += result_tile[idx_res.p, idx_res.x]



def save_result_dma(result, result_blocks, block_id_M: int):
    M, N = result.shape
    TILE_M, NUM_BLOCK_N, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N, TILE_N = result_blocks.shape
    idx_res = nl.mgrid[0:TILE_M, 0:TILE_N]
    idx_res_packed = nl.mgrid[0:TILE_M, 0:N]
    for tile_id_M in nl.affine_range(TILES_IN_BLOCK_M):
        m_ofs = (block_id_M * TILES_IN_BLOCK_M + tile_id_M) * TILE_M
        result_blocks_packed = nl.ndarray((TILE_M, N), dtype=result_blocks.dtype, buffer=nl.sbuf)
        for block_id_N in nl.affine_range(NUM_BLOCK_N):
            for tile_id_N in nl.affine_range(TILES_IN_BLOCK_N):
                n_ofs = (block_id_N * TILES_IN_BLOCK_N + tile_id_N) * TILE_N
                result_blocks_packed[idx_res.p, n_ofs + idx_res.x] = nl.copy(
                    result_blocks[idx_res.p, block_id_N, tile_id_M, tile_id_N, idx_res.x]
                )
        nl.store(result[m_ofs + idx_res_packed.p, idx_res_packed.x], value=result_blocks_packed)