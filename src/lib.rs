use std::fmt;

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
        if idx >= self.n  {
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
        if i >= self.n || j >= self.m {
            panic!("Index out of bounds");
        }

        return self.values[j].get(i);
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

        self.values[j].set(i, value);
    }

    pub fn size(&self) -> (usize, usize) {
        return (self.n, self.m);
    }

    pub fn det(&self) -> f64 {
        if self.n != self.m {
            panic!("Matrix is not square");
        }

        if self.n == 1 {
            return self.get(0, 0);
        }

        if self.n == 2 {
            return self.get(0, 0) * self.get(1, 1) - self.get(0, 1) * self.get(1, 0);
        }

        let cofactors = matrix_cofactors(self);
        let mut det = 0.0;
        for i in 0..self.n {
            let ai1 = self.get(i, 0);
            let ci1 = cofactors.get(i, 0);

            det += ai1 * ci1;
        }

        return det;
    }
}

impl fmt::Display for Vector {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let values = self.values.iter().map(|x| x.to_string()).collect::<Vec<String>>().join(", ");
        write!(f, "[{}]", values)
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let values = self.values.iter().map(|x| x.to_string()).collect::<Vec<String>>().join("\n");
        write!(f, "[{}]", values)
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
        Vector::from_vec((0..a.n).map(|i| a.get(j, i)).collect())
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

fn remove_row(mat: &Matrix, n: usize) -> Matrix {
    // for each Vector in mat, remove the nth element
    let values = mat.values.iter().map(|row| {
        let mut new_row = row.values.clone();
        new_row.remove(n);
        Vector::from_vec(new_row)
    }).collect();

    Matrix { values, n: mat.n, m: mat.m - 1 }
}

fn remove_col(mat: &Matrix, m: usize) -> Matrix {
    // remove the mth Vector in mat
    let values = mat.values.iter().enumerate().filter(|(i, _)| *i != m).map(|(_, row)| row.clone()).collect();
    Matrix { values, n: mat.n - 1, m: mat.m }
}

pub fn matrix_cofactors(mat: &Matrix)  -> Matrix{
    let values = (0..mat.m).map(|j| {
        (0..mat.n).map(|i| {
            let minor = remove_col(&remove_row(mat, i), j);
            let sign = if (i + j) % 2 == 0 { 1.0 } else { -1.0 };
            sign * minor.det()
        }).collect()
    }).map(|x| Vector::from_vec(x)).collect();

    Matrix { values, n: mat.n, m: mat.m }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_new() {
        let v = Vector::new(3);
        assert_eq!(v.size(), 3);
        assert_eq!(v.values.len(), 0);
    }

    #[test]
    fn test_vector_from_vec() {
        let v = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        assert_eq!(v.size(), 3);
        assert_eq!(v.get(0), 1.0);
        assert_eq!(v.get(1), 2.0);
        assert_eq!(v.get(2), 3.0);
    }

    #[test]
    fn test_vector_from_i32() {
        let v = Vector::from_i32(vec![1, 2, 3]);
        assert_eq!(v.size(), 3);
        assert_eq!(v.get(0), 1.0);
        assert_eq!(v.get(1), 2.0);
        assert_eq!(v.get(2), 3.0);
    }

    #[test]
    fn test_vector_set() {
        let mut v = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        v.set(1, 4.0);
        assert_eq!(v.get(1), 4.0);
    }

    #[test]
    #[should_panic(expected = "Index out of bounds")]
    fn test_vector_get_out_of_bounds() {
        let v = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        v.get(3);
    }

    #[test]
    fn test_matrix_new() {
        let m = Matrix::new(2, 3);
        assert_eq!(m.size(), (2, 3));
        assert_eq!(m.values.len(), 0);
    }

    #[test]
    fn test_matrix_from_vec() {
        let m = Matrix::from_vec(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        assert_eq!(m.size(), (2, 2));
        assert_eq!(m.get(0, 0), 1.0);
        assert_eq!(m.get(1, 1), 4.0);
    }

    #[test]
    fn test_matrix_set() {
        let mut m = Matrix::from_vec(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        m.set(0, 1, 5.0);
        assert_eq!(m.get(0, 1), 5.0);
    }

    #[test]
    fn test_add_vector() {
        let a = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Vector::from_vec(vec![4.0, 5.0, 6.0]);
        let c = add_vector(&a, &b);
        assert_eq!(c.values, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_sub_vector() {
        let a = Vector::from_vec(vec![4.0, 5.0, 6.0]);
        let b = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        let c = sub_vector(&a, &b);
        assert_eq!(c.values, vec![3.0, 3.0, 3.0]);
    }

    #[test]
    fn test_dot_product() {
        let a = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Vector::from_vec(vec![4.0, 5.0, 6.0]);
        let result = dot(&a, &b);
        assert_eq!(result, 32.0);
    }

    #[test]
    fn test_scale_vector() {
        let a = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        let result = scale_vector(&a, 2.0);
        assert_eq!(result.values, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_add_matrix() {
        let a = Matrix::from_vec(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let b = Matrix::from_vec(vec![vec![5.0, 6.0], vec![7.0, 8.0]]);
        let c = add_matrix(&a, &b);
        assert_eq!(c.values[0].values, vec![6.0, 8.0]);
        assert_eq!(c.values[1].values, vec![10.0, 12.0]);
    }

    #[test]
    fn test_sub_matrix() {
        let a = Matrix::from_vec(vec![vec![5.0, 6.0], vec![7.0, 8.0]]);
        let b = Matrix::from_vec(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let c = sub_matrix(&a, &b);
        assert_eq!(c.values[0].values, vec![4.0, 4.0]);
        assert_eq!(c.values[1].values, vec![4.0, 4.0]);
    }

    #[test]
    fn test_matrix_matrix_product() {
        let a = Matrix::from_vec(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let b = Matrix::from_vec(vec![vec![5.0, 6.0], vec![7.0, 8.0]]);
        let c = matrix_matrix_product(&a, &b);
        assert_eq!(c.values[0].values, vec![23.0, 31.0]);
        assert_eq!(c.values[1].values, vec![34.0, 46.0]);
    }

    #[test]
    fn test_matrix_vector_product() {
        let mat = Matrix::from_vec(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let vec = Vector::from_vec(vec![5.0, 6.0]);
        let result = matrix_vector_product(&mat, &vec);
        assert_eq!(result.values, vec![17.0, 39.0]);
    }

    #[test]
    fn test_transpose_matrix() {
        let a = Matrix::from_vec(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let b = transpose_matrix(&a);
        assert_eq!(b.values[0].values, vec![1.0, 3.0]);
        assert_eq!(b.values[1].values, vec![2.0, 4.0]);
    }

    #[test]
    fn test_matrix_determinant() {
        let a = Matrix::from_vec(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let det = a.det();
        assert_eq!(det, -2.0);
    }

    #[test]
    fn test_matrix_determinant_r3() {
        let a = Matrix::from_vec(vec![vec![6.0, 1.0, 1.0], vec![4.0, -2.0, 5.0], vec![2.0, 8.0, 7.0]]);
        let det = a.det();
        assert_eq!(det, -306.0);
    }

    #[test]
    fn test_matrix_cofactors() {
        let a = Matrix::from_vec(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let cofactors = matrix_cofactors(&a);
        assert_eq!(cofactors.values[0].values, vec![4.0, -3.0]);
        assert_eq!(cofactors.values[1].values, vec![-2.0, 1.0]);
    }

    #[test]
    fn printing_vector() {
        let v = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        assert_eq!(format!("{}", v), "[1, 2, 3]");
    }

    #[test]
    fn printing_matrix() {
        let m = Matrix::from_vec(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        assert_eq!(format!("{}", m), "[[1, 2]\n[3, 4]]");
    }
}
