#[derive(Clone)]
pub struct Vector {
    values: Vec<f64>,
    n: usize,
}

pub struct Matrix {
    values: Vec<Vector>,
    n: usize,
    m: usize,
}

impl Vector {
    pub fn new(n: usize) -> Self {
        Vector { values: Vec::new(), n: n }
    }

    pub fn from_vec(values: Vec<f64>) -> Self {
        let n = values.len();
        Vector { values, n }
    }

    pub fn from_f64(values: Vec<f64>) -> Self {
        let n = values.len();
        Vector { values, n }
    }

    pub fn from_i32(values: Vec<i32>) -> Self {
        let n = values.len();
        Vector { values: values.iter().map(|&x| x as f64).collect(), n }
    }

    pub fn size(&self) -> usize {
        return self.n;
    }

    pub fn get(&self, idx: usize) -> f64 {
        if idx > self.n {
            panic!("Index out of bounds");
        }

        return self.values[idx];
    }

    pub fn set(&mut self, idx: usize, value: f64) {
        if idx > self.n {
            panic!("Index out of bounds");
        }

        self.values[idx] = value;
    }
}

impl Matrix {
    pub fn new(n: usize, m: usize) -> Self {
        Matrix { values: Vec::new(), n: n, m: m }
    }

    pub fn from_vec(values: Vec<Vec<f64>>) -> Self {
        let n = values.len();
        let m = values[0].len();
        Matrix { values: values.iter().map(|x| Vector::from_vec(x.to_vec())).collect(), n, m }
    }

    pub fn from_vector(values: Vec<Vector>) -> Self {
        let n = values.len();
        let m = values[0].size();
        Matrix { values, n, m }
    }

    pub fn get(&self, i: usize, j: usize) -> f64 {
        if i > self.n || j > self.m {
            panic!("Index out of bounds");
        }

        return self.values[i].get(j);
    }

    pub fn get_row(&self, i: usize) -> Vector {
        if i > self.n {
            panic!("Index out of bounds");
        }

        // return the ith value of each vector
        return Vector::from_vec(self.values.iter().map(|x| x.get(i)).collect());
    }

    pub fn get_col(&self, j: usize) -> Vector {
        if j > self.m {
            panic!("Index out of bounds");
        }

        return self.values[j].clone();
    }

    pub fn set(&mut self, i: usize, j: usize, value: f64) {
        if i > self.n || j > self.m {
            panic!("Index out of bounds");
        }

        self.values[i].set(j, value);
    }

    pub fn size(&self) -> (usize, usize) {
        return (self.n, self.m);
    }
}

pub fn add_vector(a: &Vector, b: &Vector) -> Vector {
    if a.n != b.n {
        panic!("Vector dimensions do not match");
    }

    let values = a.values.iter().zip(b.values.iter()).map(|(a, b)| a + b).collect();
    Vector { values, n: a.n }
}

pub fn sub_vector(a: &Vector, b: &Vector) -> Vector {
    if a.n != b.n {
        panic!("Vector dimensions do not match");
    }

    let values = a.values.iter().zip(b.values.iter()).map(|(a, b)| a - b).collect();
    Vector { values, n: a.n }
}

pub fn dot(a: &Vector, b: &Vector) -> f64 {
    if a.n != b.n {
        panic!("Vector dimensions do not match");
    }

    a.values.iter().zip(b.values.iter()).map(|(a, b)| a * b).sum()
}

pub fn scale_vector(a: &Vector, scalar: f64) -> Vector {
    let values = a.values.iter().map(|x| x * scalar).collect();
    Vector { values, n: a.n }
}

pub fn scale_matrix(a: &Matrix, scalar: f64) -> Matrix {
    let values = a.values.iter().map(|row| scale_vector(row, scalar)).collect();
    Matrix { values, n: a.n, m: a.m }
}

pub fn add_matrix(a: &Matrix, b: &Matrix) -> Matrix {
    if a.n != b.n || a.m != b.m {
        panic!("Matrix dimensions do not match");
    }

    let values = a.values.iter().zip(b.values.iter()).map(|(a, b)| add_vector(a, b)).collect();
    Matrix { values, n: a.n, m: a.m }
}

pub fn sub_matrix(a: &Matrix, b: &Matrix) -> Matrix {
    if a.n != b.n || a.m != b.m {
        panic!("Matrix dimensions do not match");
    }

    let values = a.values.iter().zip(b.values.iter()).map(|(a, b)| sub_vector(a, b)).collect();
    Matrix { values, n: a.n, m: a.m }
}

pub fn transpose_vector(a: &Vector) -> Matrix {
    let values = a.values.iter().map(|&x| Vector::from_vec(vec![x])).collect();
    Matrix { values, n: a.n, m: 1 }
}

pub fn transpose_matrix(a: &Matrix) -> Matrix {
    let values = (0..a.m).map(|j| {
        Vector::from_vec((0..a.n).map(|i| a.get(i, j)).collect())
    }).collect();

    Matrix { values, n: a.m, m: a.n }
}

pub fn matrix_matrix_product(a: &Matrix, b: &Matrix) -> Matrix {
    if a.m != b.n {
        panic!("Matrix dimensions do not match");
    }

    let values = (0..a.n).map(|i| {
        Vector::from_vec((0..b.m).map(|j| dot(&a.get_row(i), &b.get_col(j))).collect())
    }).collect();

    Matrix { values, n: a.n, m: b.m }
}

pub fn matrix_vector_product(mat: &Matrix, vec: &Vector) -> Vector {
    if mat.m != vec.n {
        panic!("Matrix and vector dimensions do not match");
    }

    let values = mat.values.iter().map(|row| dot(row, vec)).collect();

    Vector { values, n: mat.n }
}