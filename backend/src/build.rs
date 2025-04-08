fn main() {
    let libtorch_path = std::env::var("LIBTORCH").expect("LIBTORCH environment variable not set");
    println!("cargo:rustc-link-search=native={}/lib", libtorch_path);

    println!("cargo:rustc-link-lib=c10");
    println!("cargo:rustc-link-lib=torch_cpu");
    println!("cargo:rustc-link-lib=torch");
    println!("cargo:rustc-link-lib=torch_cuda");

    println!("cargo:rustc-link-lib=gomp");
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rustc-link-lib=pthread");

    println!("cargo:rerun-if-env-changed=LIBTORCH");
    println!("cargo:rerun-if-env-changed=USE_CUDA");
}
