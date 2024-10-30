
### Original program (v1) Rust: 
Asks for the user to input roots, the associated function for the roots, and the derivative of that function. Then it would launch something you could interact with to zoom in and move around within the image.
### v2 Python: 
Not interactive at all; would generate its own roots and take the derivative, and then generate an image and save it. It would repeat this process indefinitely.
### v3 Rust (redux): 
This version does both of the things that v1 and v2 can do: it can launch an interactive app like v1, and it will either ask you for roots or generate its own like v2; and it takes the derivative on its own. This version is also substantially faster because it can use more than one core on your computer.
### How v3 works:
1.  Reads the command line arguments given by the user to determine the behavior of the code. These include:
-   number of cores to use (default = all; it adapts to however many your computer has)
-   resolution of the image (default = 10,000 x 10,000 = 100,000,000 pixels)
-   whether or not it should be shaded (default = flat)
-   how many roots should be in the polynomial (default = 20)
-   how many images to generate (default = 1 image)
-   whether to generate it from a file (as opposed to randomly generating them) (default = does not read from a file
-   whether or not to launch the interactive viewer for the image upon completion (default = save it and do not launch the viewer)
2. Creates a directory to put the files in (if one doesn’t already exist).
3. Starts generating the images.
4. If not running interactively, it will save the image and then do the next one; otherwise, it will launch the interactive viewer.
## Some images
<img src="https://github.com/user-attachments/assets/ce25e3e6-6ba2-4ff4-a4ac-183acdbef062" width=200><img src="https://github.com/user-attachments/assets/8767dd68-2ccd-4973-9d5c-f2399816abc5" width=200><img src="https://github.com/user-attachments/assets/d4c2c2e7-946b-4109-a04e-1aff95bf07fc" width=200><img src="https://github.com/user-attachments/assets/206e8bc0-7274-44b4-bf4d-ccad1bfc3364" width=200>
# Steps to run
*might need github destop if on windows*
1. git clone
2. get rust
[installation methods](https://forge.rust-lang.org/infra/other-installation-methods.html)
3. cargo r -r -- -h
4. config args after the "cargo r -r --" [help with cargo run args can be found here](https://doc.rust-lang.org/cargo/commands/cargo-run.html)
