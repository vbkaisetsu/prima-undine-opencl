use glob::glob;
use std::env;
use std::fs;
use std::path::Path;

fn main() {
    let out_dir = env::var_os("OUT_DIR").unwrap();
    for entry in glob("src/kernels/*.cl").unwrap() {
        match entry {
            Ok(path) => {
                let kernel_name = path.file_stem().unwrap().to_str().unwrap();
                let dest_path =
                    Path::new(&out_dir).join("kernel_".to_string() + kernel_name + ".in");
                let kernel_src = fs::read(path).unwrap();
                fs::write(
                    &dest_path,
                    "vec![".to_string()
                        + &kernel_src
                            .iter()
                            .map(|x| x.to_string())
                            .collect::<Vec<String>>()
                            .join(",")
                        + "]",
                )
                .unwrap();
            }
            Err(e) => panic!(e),
        }
    }
    println!("cargo:rerun-if-changed=src/kernels");
}
