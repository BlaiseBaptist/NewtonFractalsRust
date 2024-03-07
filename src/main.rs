use clap::Parser;
use image::RgbImage;
use rand::random;
use rayon::iter::ParallelIterator;
use std::fmt::{Display, Formatter, Result};
use std::{fs, ops, path};
#[derive(Debug, PartialEq)]
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
impl Display for c64 {
    fn fmt(&self, f: &mut Formatter) -> Result {
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

fn f(x: &c64, roots: &[c64]) -> c64 {
    roots
        .iter()
        .fold(c64::new(1.0, 0.0), |acc, r| &acc * &(x - r))
}
fn f_prime(x: &c64, roots: &[c64]) -> c64 {
    let len = roots.len();
    if len == 1 {
        return c64::new(1.0, 0.0);
    }
    let left = &roots[0..len / 2];
    let right = &roots[len / 2..];
    &(&f(&x, left) * &f_prime(&x, right)) + &(&f_prime(&x, left) * &f(&x, right))
}
fn newton_iter(mut point: c64, roots: &[c64]) -> Option<[u8; 2]> {
    for step in 1..255 {
        point = &point - &(&f(&point, roots) / &f_prime(&point, roots));
        let diffs = roots.iter().map(|v| (&point - v).sum());
        for (spot, diff) in diffs.enumerate() {
            if diff.abs() <= 0.00000001 {
                return Some([spot as u8, step]);
            }
        }
    }
    None
}
fn make_random_im(nx: u32, ny: u32, len: usize, pattern: PatternFn, dir: &str) {
    let y_sep = 1.0 / ny as f64;
    let x_sep = 1.0 / nx as f64;
    let x_start = -0.5;
    let y_start = -0.5;
    let roots: Vec<c64> = (0..len)
        .map(|_| c64::new(random::<f64>() - 0.5, random::<f64>() - 0.5))
        .collect();
    let colors: Vec<Color> = (0..len)
        .map(|_| [random::<u8>(), random::<u8>(), random::<u8>()])
        .collect();
    let mut imgbuf: RgbImage = image::ImageBuffer::new(nx, ny);
    imgbuf.par_enumerate_pixels_mut().for_each(|(x, y, pixel)| {
        *pixel = image::Rgb(pattern(
            newton_iter(
                c64::new((x as f64 * x_sep) + x_start, (y as f64 * y_sep) + y_start),
                &roots,
            ),
            &colors,
        ))
    });
    let path: path::PathBuf = [
        dir.to_string(),
        roots.iter().take(19).map(|r| format!("{r}")).collect::<String>()+ ".png",
    ]
    .iter()
    .collect();

    imgbuf.save(path).unwrap();
}
fn shade(value: Option<[u8; 2]>, colors: &[Color]) -> Color {
    match value {
        Some(i) => colors[i[0] as usize]
            .iter()
            .map(|x| ((*x as f64) * 2.8 / ((i[1] + 1) as f64).powf(0.8)) as u8)
            .collect::<Vec<u8>>()
            .try_into()
            .unwrap(),
        None => [0, 0, 0],
    }
}
fn invert(value: Option<[u8; 2]>, colors: &[Color]) -> Color {
    match value {
        Some(i) => colors[i[0] as usize]
            .iter()
            .map(|x| (0.08 * *x as f64 * i[1] as f64) as u8)
            .collect::<Vec<u8>>()
            .try_into()
            .unwrap(),
        None => [0, 0, 0],
    }
}
fn flat(value: Option<[u8; 2]>, colors: &[Color]) -> Color {
    match value {
        Some(i) => colors[i[0] as usize],
        None => [0, 0, 0],
    }
}
type Color = [u8; 3];
type PatternFn = fn(Option<[u8; 2]>, &[Color]) -> Color;
#[derive(Parser, Debug)]
struct Args {
    #[arg(short, long)]
    cores: Option<usize>,
    #[arg(short, long, default_value_t = 10000)]
    size: u32,
    #[arg(short, long, default_value_t = 20)]
    len: usize,
    #[arg(short, long, default_value = "flat")]
    pattern: String,
    #[arg(short, long, default_value_t = 1)]
    number: usize,
}
fn main() {
    let args = Args::parse();
    println!("{:?}", args);
    if let Some(i) = args.cores {
        rayon::ThreadPoolBuilder::new()
            .num_threads(i)
            .build_global()
            .unwrap();
    }
    let pattern: PatternFn = match args.pattern.as_str() {
        "shade" => shade,
        "invert" => invert,
        _ => flat,
    };
    let path = match args.size {
        d if d < 10000 => "small",
        d if d > 10000 => "large",
        _ => "fract",
    };
	let _ = fs::create_dir(path);
	for i in 0..args.number {
        make_random_im(args.size, args.size, args.len, pattern, path);
        println!("made image {} in the {} folder", i + 1, path);
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn simple_func() {
        let roots = vec![(-1).into(), (-2).into(), (-3).into()];
        assert_eq!(f(c64::new(1.0, 1.0), &roots), c64::new(15.0, 25.0));
        assert_eq!(
            &c64::new(1.0, 1.0) / &c64::new(1.0, 1.0),
            c64::new(1.0, 0.0)
        );
    }
    #[test]
    fn d_func() {
        let roots = vec![(-1).into(), (-2).into(), (-3).into()];
        assert_eq!(f_prime((-2).into(), &roots), (-1).into());
        assert_eq!(f_prime((0).into(), &roots), (11).into());
        assert_eq!(f_prime(c64::new(1.0, 1.0), &roots), c64::new(23.0, 18.0));
    }
    #[test]
    fn newton_test() {
        let roots = vec![(-1).into(), (-2).into(), (-3).into()];
        assert_eq!(newton_iter((-1).into(), &roots), Some([0, 0]));
        assert_eq!(newton_iter(c64::new(-10.0, -10.0), &roots), Some([2, 15]));
    }
    #[test]
    fn colorize() {
        let roots = vec![(-1).into(), (-2).into(), (-3).into()];
        let colors = vec![[255, 0, 0], [0, 255, 0], [0, 0, 255]];
        assert_eq!(shade(newton_iter((-1).into(), &roots), &colors), colors[0]);
        assert_eq!(
            shade(newton_iter(c64::new(-10.0, -10.0), &roots), &colors),
            [0, 0, 15]
        );
    }
}
