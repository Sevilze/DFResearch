fn main() {
    println!("cargo:rustc-link-search=native={}", std::env::var("LIBTORCH").unwrap() + "\\lib");
    println!("cargo:rustc-link-lib=torch");
    println!("cargo:rustc-link-lib=torch_cpu");
    println!("cargo:rustc-link-arg=/MTd");
}