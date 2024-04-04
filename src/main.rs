use clap::Parser;
use image::RgbImage;
use rand::random;
use rayon::iter::ParallelIterator;
use regex::Regex;
use std::fmt::{Display, Formatter, Result};
use std::{fs, ops, path, time};
#[derive(Debug, PartialEq, Clone)]
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
impl From<&str> for c64 {
    fn from(s: &str) -> Self {
        let pattern = Regex::new(r"(?ix)(?<r>-?[\d\.]+)?\+?(?:(?<i>-?[\d\.])i)?$").unwrap();
        let caps: c64 = match pattern.captures(s) {
            Some(cap) => c64 {
                r: match cap.name("r") {
                    Some(v) => v.as_str().parse().expect("bad regex"),
                    None => 0.0,
                },
                i: match cap.name("i") {
                    Some(v) => v.as_str().parse().expect("bad regex"),
                    None => 0.0,
                },
            },
            None => 0.0.into(),
        };
        return caps;
    }
}
impl Display for c64 {
    fn fmt(&self, f: &mut Formatter) -> Result {
        if self.i == 0.0 {
            return write!(f, "({:.2})", self.r);
        }

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
struct Fractal {
    image: RgbImage,
    path: path::PathBuf,
}
#[derive(Debug)]
struct FractalData {
    y_res: u32,
    x_res: u32,
    top_left: c64,
    bot_right: c64,
    pattern: PatternFn,
    roots: Vec<c64>,
    colors: Vec<Color>,
    path: path::PathBuf,
}
impl From<&Args> for FractalData {
    fn from(args: &Args) -> Self {
        let path: path::PathBuf = match args.size {
            d if d < 10000 => "small",
            d if d > 10000 => "large",
            _ => "fract",
        }
        .into();
        let _ = fs::create_dir(path.clone());
        FractalData {
            y_res: args.size,
            x_res: args.size,
            top_left: c64::new(-0.5, -0.5),
            bot_right: c64::new(0.5, 0.5),
            pattern: match args.pattern.as_str() {
                "shade" => shade,
                "invert" => invert,
                _ => flat,
            },
            colors: (0..args.len)
                .map(|_| [random::<u8>(), random::<u8>(), random::<u8>()])
                .collect::<Vec<Color>>(),
            roots: (0..args.len)
                .map(|_| c64::new(random::<f64>() - 0.5, random::<f64>() - 0.5))
                .collect::<Vec<c64>>(),
            path: path,
        }
    }
}
impl FractalData {
    fn new(args: &Args, data: (Vec<c64>, Vec<Color>)) -> Self {
        let path: path::PathBuf = match args.size {
            d if d < 10000 => "small",
            d if d > 10000 => "large",
            _ => "fract",
        }
        .into();
        let _ = fs::create_dir(path.clone());
        FractalData {
            y_res: args.size,
            x_res: args.size,
            top_left: c64::new(-0.5, -0.5),
            bot_right: c64::new(0.5, 0.5),
            pattern: match args.pattern.as_str() {
                "shade" => shade,
                "invert" => invert,
                _ => flat,
            },
            colors: data.1,
            roots: data.0,
            path: path,
        }
    }
}
#[derive(Parser, Debug)]
struct Args {
    #[arg(short, long, help = "Number of cores to use")]
    cores: Option<usize>,

    #[arg(short, long, default_value_t = 10000, help = "Number of pixels across")]
    size: u32,

    #[arg(short, long, default_value = "flat")]
    pattern: String,

    #[arg(short, long, default_value_t = 20, help = "How many roots")]
    len: usize,

    #[arg(short, long, default_value_t = 1, help = "Number of images to make")]
    number: usize,

    #[arg(short, long, help = "Make from csv", default_value = " ")]
    file: path::PathBuf,
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
fn make_im(data: &FractalData) -> Fractal {
    let y_sep = (data.bot_right.i - data.top_left.i) / (data.y_res as f64);
    let x_sep = (data.bot_right.r - data.top_left.r) / (data.x_res as f64);
    let x_start = data.top_left.r;
    let y_start = data.top_left.r;
    let mut imgbuf: RgbImage = image::ImageBuffer::new(data.x_res, data.y_res);
    imgbuf.par_enumerate_pixels_mut().for_each(|(x, y, pixel)| {
        *pixel = image::Rgb((data.pattern)(
            newton_iter(
                c64::new((x as f64 * x_sep) + x_start, (y as f64 * y_sep) + y_start),
                &data.roots,
            ),
            &data.colors,
        ))
    });
    Fractal {
        path: [
            data.path.display().to_string(),
            data.roots
                .iter()
                .take(19)
                .map(|r| format!("{r}"))
                .collect::<String>()
                + ".png",
        ]
        .iter()
        .collect(),
        image: imgbuf,
    }
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
fn decode_file(contents: String) -> (Vec<c64>, Vec<Color>) {
    let mut roots: Vec<c64> = Vec::new();
    let mut colors: Vec<Color> = Vec::new();
    for line in contents.lines().skip(1) {
        let mut values = line.split(";");
        let root: c64 = match values.next() {
            Some(r) => r.trim().into(),
            None => 0.0.into(),
        };
        let color = match values.next() {
            Some(c) => c
                .trim()
                .split(",")
                .cycle()
                .take(3)
                .map(|c| match c.parse::<u8>() {
                    Ok(val) => val,
                    Err(_) => 0,
                })
                .collect::<Vec<u8>>()
                .try_into()
                .unwrap(),
            None => [0, 0, 0],
        };
		println!("root: {}, color: {:?}",root,color);
        roots.push(root);
        colors.push(color);
    }
    return (roots, colors);
}
fn make_ims(args: Args) {
    let len = args.number;
    for i in 0..len {
        let data: FractalData = match fs::read_to_string(&args.file) {
            Ok(f) => FractalData::new(&args, decode_file(f)),
            Err(_) => (&args).into(),
        };
        let now = time::Instant::now();
        let fractal = make_im(&data);
        let later = now.elapsed().as_nanos();
        println!(
            "put im #{} in {} dir; took {}ns, {}ns/px",
            i + 1,
            data.path.display(),
            later,
            later / (data.x_res * data.y_res) as u128
        );
        let _ = fractal.image.save(fractal.path);
    }
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
	make_ims(args);
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
