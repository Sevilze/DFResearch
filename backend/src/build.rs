use pyo3_build_config::use_pyo3_cfgs;

fn main() {
    use_pyo3_cfgs();
    pyo3_build_config::add_extension_module_link_args();

    let libtorch_path = std::env::var("LIBTORCH").expect("LIBTORCH environment variable not set");
    println!("cargo:rustc-link-search=native={}/lib", libtorch_path);

    let python_lib_dir = std::env::var("PYTHON_LIB_DIR").expect("PYTHON_LIB_DIR environment variable not set");
    println!("cargo:rustc-link-search=native={}", python_lib_dir);
    println!("cargo:rustc-link-lib=python3.13");

    println!("cargo:rustc-link-lib=c10");
    println!("cargo:rustc-link-lib=torch_cpu");
    println!("cargo:rustc-link-lib=torch");
    println!("cargo:rustc-link-lib=torch_cuda");

    println!("cargo:rustc-link-lib=gomp");
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rustc-link-lib=pthread");

    println!("cargo:rerun-if-env-changed=LIBTORCH");
    println!("cargo:rerun-if-env-changed=PYTHON_LIB_DIR");
    println!("cargo:rerun-if-env-changed=USE_CUDA");
}
