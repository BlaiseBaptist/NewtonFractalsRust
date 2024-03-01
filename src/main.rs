use image::RgbImage;
use rand::prelude::*;
use rayon::iter::ParallelIterator;
use std::ops;
use std::fmt::Write;
#[derive(Copy, Clone, PartialEq)]
#[allow(non_camel_case_types)]
struct c64 {
    r: f64,
    i: f64,
}
impl c64 {
    fn new(r: f64, i: f64) -> c64 {
        c64 { r, i }
    }
    fn sum(&self) -> f64 {
        self.r + self.i
    }
}
impl From<f64> for c64 {
    fn from(item: f64) -> Self {
        c64 { r: item, i: 0.0 }
    }
}
impl From<i32> for c64 {
    fn from(item: i32) -> Self {
        c64 {
            r: item.into(),
            i: 0.0,
        }
    }
}
impl std::fmt::Display for c64 {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if self.i > 0.0 {
            return write!(f, "({:.2}+{:.2}i)", self.r, self.i);
        }
        write!(f, "({:.2}{:.2}i)", self.r, self.i)
    }
}
impl ops::Mul<&c64> for &c64 {
    type Output = c64;
    fn mul(self, other: &c64) -> c64 {
        c64 {
            r: (self.r * other.r) - (self.i * other.i),
            i: (self.i * other.r) + (self.r * other.i),
        }
    }
}
impl ops::Add<&c64> for &c64 {
    type Output = c64;
    fn add(self, other: &c64) -> c64 {
        c64 {
            r: self.r + other.r,
            i: self.i + other.i,
        }
    }
}

impl ops::Sub<&c64> for &c64 {
    type Output = c64;
    fn sub(self, other: &c64) -> c64 {
        c64 {
            r: self.r - other.r,
            i: self.i - other.i,
        }
    }
}
impl ops::Div<&c64> for &c64 {
    type Output = c64;
    fn div(self, other: &c64) -> c64 {
        c64 {
            r: (self.r * other.r + self.i * other.i) / (other.r * other.r + other.i * other.i),
            i: (self.i * other.r - self.r * other.i) / (other.r * other.r + other.i * other.i),
        }
    }
}

fn f(x: c64, roots: &[c64]) -> c64 {
    roots
        .iter()
        .fold(c64::new(1.0, 0.0), |acc, r| &acc * &(&x - r))
}
fn f_prime(x: c64, roots: &[c64]) -> c64 {
    let len = roots.len();
    if len == 1 {
        return c64::new(1.0, 0.0);
    }
    let left = &roots[0..len / 2];
    let right = &roots[len / 2..];
    &(&f(x, left) * &f_prime(x, right)) + &(&f_prime(x, left) * &f(x, right))
}
fn newton_iter(mut point: c64, roots: &[c64]) -> Option<usize> {
    for _ in 0..100 {
        point = &point - &(&f(point, roots) / &f_prime(point, roots));
        let diffs = roots.iter().map(|v| (&point - v).sum());
        for (spot, diff) in diffs.enumerate() {
            if diff.abs() <= 0.00000001 {
                return Some(spot);
            }
        }
    }
    None
}
fn make_random_im(nx: u32, ny: u32) {
    let y_sep = 1.0 / ny as f64;
    let x_sep = 1.0 / nx as f64;
    let x_start = -0.5;
    let y_start = -0.5;
    const LEN: usize = 4;
    let roots: [c64; LEN] =
        core::array::from_fn(|_| c64::new(random::<f64>() - 0.5, random::<f64>() - 0.5));
    let colors: [[u8; 3]; LEN] =
        core::array::from_fn(|_| [random::<u8>(), random::<u8>(), random::<u8>()]);
    let mut imgbuf: RgbImage = image::ImageBuffer::new(nx, ny);
    imgbuf.par_enumerate_pixels_mut().for_each(|(x, y, pixel)| {
        *pixel = image::Rgb(
            match newton_iter(
                c64::new((x as f64 * x_sep) + x_start, (y as f64 * y_sep) + y_start),
                &roots,
            ) {
                Some(i) => colors[i],
                None => [0, 0, 0],
            },
        )
    });
	let mut path =String::from("fractals/");
	for root in roots.iter(){
		write!(path,"{}",root).unwrap();	
	}
	path.push_str(".png");
    imgbuf.save(path).unwrap();
}
//TODO make the file names based on the roots
fn main() {
    let ny = 50000;
    let nx = 50000;
	for i in 0..100{
		make_random_im(nx, ny);
		println!("made image {}",i+1);
	}
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn simple_func() {
        let roots: [c64; 3] = [(-1).into(), (-2).into(), (-3).into()];
        assert_eq!(f(c64::new(1.0, 1.0), &roots), c64::new(15.0, 25.0));
        assert_eq!(
            &c64::new(1.0, 1.0) / &c64::new(1.0, 1.0),
            c64::new(1.0, 0.0)
        );
    }
    #[test]
    fn d_func() {
        let roots: [c64; 3] = [(-1).into(), (-2).into(), (-3).into()];
        assert_eq!(f_prime((-2).into(), &roots), (-1).into());
        assert_eq!(f_prime((0).into(), &roots), (11).into());
        assert_eq!(f_prime(c64::new(1.0, 1.0), &roots), c64::new(23.0, 18.0));
    }
    #[test]
    fn newton_test() {
        let roots: [c64; 3] = [(-1).into(), (-2).into(), (-3).into()];
        assert_eq!(newton_iter((-1).into(), &roots), Some(0));
        assert_eq!(newton_iter(c64::new(-10.0, -10.0), &roots), Some(2));
    }
}
