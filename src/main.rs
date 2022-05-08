// rustc 1.62.0-nightly needed to run
#![feature(portable_simd)]
use std::ops::{Index, IndexMut};
use std::simd::u32x16;

struct Matrix<T> {
    pub width: usize,
    pub height: usize,
    data: Vec<T>,
}

impl<T> Matrix<T> {
    fn new(width: usize, height: usize, data: Vec<T>) -> Self {
        Matrix {
            width,
            height,
            data,
        }
    }

    fn new_uninit(width: usize, height: usize) -> Self {
        let mut data = Vec::with_capacity(width * height);
        unsafe { data.set_len(width * height) };
        Matrix {
            width,
            height,
            data,
        }
    }
}

impl<T> Index<(usize, usize)> for Matrix<T> {
    type Output = T;
    fn index(&self, index: (usize, usize)) -> &T {
        &self.data[self.width * index.0 + index.1]
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut T {
        &mut self.data[self.width * index.0 + index.1]
    }
}

impl<T: Copy> Matrix<T> {
    fn transpose(&mut self) {
        let mut transpose = Matrix::new_uninit(self.height, self.width);
        for i in 0..self.height {
            for j in 0..self.width {
                transpose[(j, i)] = self[(i, j)];
            }
        }
        self.data = transpose.data;
        let height = self.height;
        self.height = self.width;
        self.width = height;
    }
}

impl<T> Index<usize> for Matrix<T> {
    type Output = [T];
    fn index(&self, index: usize) -> &[T] {
        &self.data[self.width*index..self.width*(index+1)]
    }
}

fn lindenmayer_a(l: u8, i: &mut isize, j: &mut isize, d: &mut isize, a: &Matrix<u32>, b: &mut Matrix<u32>, c: &mut Matrix<u32>) {
    if l==0 {
        let i = *i as usize;
        let j = *j as usize;
        let a_row = &a[i];
        let b_row = &b[j];
        let mut sum = u32x16::splat(0);
        for k in (0..a.width).step_by(16) {
            sum += u32x16::from_slice(&a_row[k..])*u32x16::from_slice(&b_row[k..]);
        }
        c[(i,j)] = sum.reduce_sum();
    } else {
        //*d = (*d+3)&3;
        *d = (*d+3)%4;
        lindenmayer_b(l-1, i, j, d, a, b, c);
        //*j += (*d-1)&1;
        *j += (*d-1)%2;
        *i += (*d-2)%2;
        *d = (*d+1)%4;
        lindenmayer_a(l-1, i, j, d, a, b, c);
        *j += (*d-1)%2;
        *i += (*d-2)%2;
        lindenmayer_a(l-1, i, j, d, a, b, c);
        *d = (*d+1)%4;
        *j += (*d-1)%2;
        *i += (*d-2)%2;
        lindenmayer_b(l-1, i, j, d, a, b, c);
        *d = (*d+3)%4;
    }
}

fn lindenmayer_b(l: u8, i: &mut isize, j: &mut isize, d: &mut isize, a: &Matrix<u32>, b: &mut Matrix<u32>, c: &mut Matrix<u32>) {
    if l==0 {
        let i = *i as usize;
        let j = *j as usize;
        let a_row = &a[i];
        let b_row = &b[j];
        let mut sum = u32x16::splat(0);
        for k in (0..a.width).step_by(16) {
            sum += u32x16::from_slice(&a_row[k..])*u32x16::from_slice(&b_row[k..]);
        }
        c[(i,j)] = sum.reduce_sum();
    } else {
        *d = (*d+1)%4;
        lindenmayer_a(l-1, i, j, d, a, b, c);
        *j += (*d-1)%2;
        *i += (*d-2)%2;
        *d = (*d+3)%4;
        lindenmayer_b(l-1, i, j, d, a, b, c);
        *j += (*d-1)%2;
        *i += (*d-2)%2;
        lindenmayer_b(l-1, i, j, d, a, b, c);
        *d = (*d+3)%4;
        *j += (*d-1)%2;
        *i += (*d-2)%2;
        lindenmayer_a(l-1, i, j, d, a, b, c);
        *d = (*d+1)%4;
    }
}

fn matmul(a: &Matrix<u32>, b: &mut Matrix<u32>, half_log_grid_size: u8) -> Matrix<u32> {
    b.transpose();
    let mut c = Matrix::new_uninit(b.width, a.height);
    lindenmayer_a(half_log_grid_size,&mut 0,&mut 0,&mut 3, &a, b, &mut c);
    c
}


fn main() {
    let half_log_grid_size = 11;
    let size = 1 << half_log_grid_size;

    let a = Matrix::new(size, size, vec![1u32; size * size]);
    let mut b = Matrix::new(size, size, vec![1u32; size * size]);
    let c = matmul(&a, &mut b, half_log_grid_size);
    assert_eq!(size,c.width);
}
