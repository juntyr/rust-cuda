#[cfg(all(feature = "device", not(doc)))]
use core::arch::nvptx;

pub struct Thread {
    _private: (),
}

#[expect(clippy::module_name_repetitions)]
pub struct ThreadBlock {
    _private: (),
}

#[expect(clippy::module_name_repetitions)]
pub struct ThreadBlockGrid {
    _private: (),
}

impl Thread {
    #[must_use]
    #[expect(clippy::inline_always)]
    #[inline(always)]
    pub const fn this() -> Self {
        Self { _private: () }
    }

    #[must_use]
    #[expect(clippy::inline_always)]
    #[inline(always)]
    pub fn index(&self) -> usize {
        let block = self.block();
        let grid = block.grid();

        let block_id = block.idx().as_id(&grid.dim());
        let thread_id = self.idx().as_id(&block.dim());

        block_id * block.dim().size() + thread_id
    }

    #[must_use]
    #[expect(clippy::inline_always)]
    #[inline(always)]
    pub fn idx(&self) -> Idx3 {
        #[expect(clippy::cast_sign_loss)]
        unsafe {
            Idx3 {
                x: nvptx::_thread_idx_x() as u32,
                y: nvptx::_thread_idx_y() as u32,
                z: nvptx::_thread_idx_z() as u32,
            }
        }
    }

    #[must_use]
    #[expect(clippy::inline_always)]
    #[inline(always)]
    pub const fn block(&self) -> ThreadBlock {
        ThreadBlock { _private: () }
    }
}

impl ThreadBlock {
    #[must_use]
    #[expect(clippy::inline_always)]
    #[inline(always)]
    pub fn dim(&self) -> Dim3 {
        #[expect(clippy::cast_sign_loss)]
        unsafe {
            Dim3 {
                x: nvptx::_block_dim_x() as u32,
                y: nvptx::_block_dim_y() as u32,
                z: nvptx::_block_dim_z() as u32,
            }
        }
    }

    #[must_use]
    #[expect(clippy::inline_always)]
    #[inline(always)]
    pub fn idx(&self) -> Idx3 {
        #[expect(clippy::cast_sign_loss)]
        unsafe {
            Idx3 {
                x: nvptx::_block_idx_x() as u32,
                y: nvptx::_block_idx_y() as u32,
                z: nvptx::_block_idx_z() as u32,
            }
        }
    }

    #[must_use]
    #[expect(clippy::inline_always)]
    #[inline(always)]
    pub const fn grid(&self) -> ThreadBlockGrid {
        ThreadBlockGrid { _private: () }
    }

    #[expect(clippy::inline_always)]
    #[inline(always)]
    pub fn synchronize(&self) {
        unsafe { nvptx::_syncthreads() }
    }
}

impl ThreadBlockGrid {
    #[must_use]
    #[expect(clippy::inline_always)]
    #[inline(always)]
    pub fn dim(&self) -> Dim3 {
        #[expect(clippy::cast_sign_loss)]
        unsafe {
            Dim3 {
                x: nvptx::_grid_dim_x() as u32,
                y: nvptx::_grid_dim_y() as u32,
                z: nvptx::_grid_dim_z() as u32,
            }
        }
    }
}

/// Dimension specified in kernel launching
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Dim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

/// Indices that the kernel code is running on
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Idx3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Dim3 {
    #[must_use]
    #[expect(clippy::inline_always)]
    #[inline(always)]
    pub const fn size(&self) -> usize {
        (self.x as usize) * (self.y as usize) * (self.z as usize)
    }
}

impl Idx3 {
    #[must_use]
    #[expect(clippy::inline_always)]
    #[inline(always)]
    pub const fn as_id(&self, dim: &Dim3) -> usize {
        (self.x as usize)
            + (self.y as usize) * (dim.x as usize)
            + (self.z as usize) * (dim.x as usize) * (dim.y as usize)
    }
}
